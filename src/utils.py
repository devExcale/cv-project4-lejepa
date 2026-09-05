import glob
import json
import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.utils import save_image

from src.data import get_dataloaders, get_or_compute_stats
from src.globals import CONFIG, DATASETS, DEVICE, DIR_CHECKPOINTS, DIR_OUTPUT, set_seed
from src.network import AttentionEncoder, build_model


class GuidedBackprop:
	"""Guided Backpropagation hook manager for capturing fine pixel gradients."""

	def __init__(self, model: nn.Module):
		self.model = model
		self.hooks = []
		# In-place ReLUs break backward hooks; disable them across the entire model
		for module in self.model.modules():
			if isinstance(module, nn.ReLU):
				module.inplace = False

	def _register_hooks(self):
		def relu_backward_hook(module, grad_in, grad_out):
			if isinstance(grad_in[0], torch.Tensor):
				return (torch.clamp(grad_in[0], min=0.0),)

		for module in self.model.modules():
			if isinstance(module, nn.ReLU):
				self.hooks.append(module.register_full_backward_hook(relu_backward_hook))

	def generate_gradients(self, input_tensor: torch.Tensor, target_class: int) -> np.ndarray:
		self.model.eval()
		self._register_hooks()  # Attach hooks only for Guided Backprop pass

		input_tensor = input_tensor.clone().detach().requires_grad_(True)
		output = self.model(input_tensor)
		self.model.zero_grad()

		score = output[0, target_class]
		score.backward()

		gradients = input_tensor.grad.detach().cpu().squeeze(0).permute(1, 2, 0).numpy()

		self.remove_hooks()  # Detach hooks immediately so other passes run normally
		return gradients

	def remove_hooks(self):
		for hook in self.hooks:
			hook.remove()
		self.hooks = []


class GradCAM:
	"""
	Grad-CAM hook manager and saliency map generator for CNN backbones.
	"""

	def __init__(self, model: nn.Module, target_layer: nn.Module):
		self.model = model
		self.target_layer: nn.Module = target_layer
		self.activations: Optional[torch.Tensor] = None
		self.gradients: Optional[torch.Tensor] = None

		# Register forward and backward hooks
		self.forward_handle = self.target_layer.register_forward_hook(self._forward_hook)
		self.backward_handle = self.target_layer.register_full_backward_hook(self._backward_hook)

	def _forward_hook(
			self,
			_module: nn.Module,
			_inputs: Tuple[torch.Tensor, ...],
			output: torch.Tensor,
	) -> None:
		self.activations = output.detach()

	def _backward_hook(
			self,
			_module: nn.Module,
			_grad_input: tuple[torch.Tensor | None, ...] | torch.Tensor,
			grad_output: tuple[torch.Tensor | None, ...] | torch.Tensor,
	) -> tuple[torch.Tensor | None, ...] | torch.Tensor | None:
		if isinstance(grad_output, tuple) and grad_output:
			self.gradients = grad_output[0].detach() if grad_output[0] is not None else None
		elif isinstance(grad_output, torch.Tensor):
			self.gradients = grad_output.detach()
		return None

	def generate_cam(
			self,
			input_tensor: torch.Tensor,
			target_class: Optional[int] = None
	) -> torch.Tensor:
		"""
		Generate normalized Grad-CAM heatmaps for a single input tensor of shape [1, C, H, W].
		Returns a 2D float tensor of shape [H, W] normalized in [0, 1].
		"""
		self.model.eval()
		self.model.zero_grad()

		# Forward pass
		logits = self.model(input_tensor)  # [1, Num_Classes]

		if target_class is None:
			target_class = logits.argmax(dim=1).item()

		# Target score (pre-softmax)
		score = logits[0, target_class]
		score.backward(retain_graph=True)

		# Global average pooling of gradients: alpha_k = (1/Z) * sum(grad)
		# self.gradients: [1, C, H', W'], self.activations: [1, C, H', W']
		weights = torch.mean(self.gradients, dim=(2, 3), keepdim=True)  # [1, C, 1, 1]

		# Weighted combination of feature activation maps
		cam = torch.sum(weights * self.activations, dim=1, keepdim=True)  # [1, 1, H', W']
		cam = F.relu(cam)  # Apply ReLU to focus on features that have a positive influence

		# Upsample to original input resolution (32x32)
		_, _, h, w = input_tensor.shape
		cam = F.interpolate(cam, size=(h, w), mode="bilinear", align_corners=False)
		cam = cam.squeeze().cpu()

		# Normalize between 0 and 1
		cam_min, cam_max = cam.min(), cam.max()
		if cam_max > cam_min:
			cam = (cam - cam_min) / (cam_max - cam_min)
		else:
			cam = torch.zeros_like(cam)

		return cam

	def remove_hooks(self):
		"""Clean up PyTorch hooks."""
		self.forward_handle.remove()
		self.backward_handle.remove()

class GMAR:
	def __init__(self, model: nn.Module):
		self.model = model
		self.attn_modules = [
			module for module in self.model.modules()
			if isinstance(module, AttentionEncoder)
		]
		if not self.attn_modules:
			raise ValueError("GMAR requires a model containing AttentionEncoder modules")

	def _get_attention_matrices_and_grads(self, inputs: torch.Tensor, target_category: int = None):
		self.model.eval()
		self.model.zero_grad()

		with torch.enable_grad():
			logits = self.model(inputs, need_attn=True)
			if target_category is None:
				target_category = logits.argmax(dim=-1).item()
			logits[0, target_category].backward()

		attentions = []
		gradients = []
		for module in self.attn_modules:
			if module.attention is None or module.attention.grad is None:
				raise RuntimeError("Attention weights or gradients were not captured")
			attentions.append(module.attention.detach())
			gradients.append(module.attention.grad.detach())

		return attentions, gradients, target_category

	def compute_head_weights(self, gradients: list[torch.Tensor], norm_type: str = "l1"):
		layer_head_weights = []
		for grad in gradients:
			values = grad[0]
			if norm_type == "l1":
				norms = values.abs().sum(dim=(-2, -1))
			elif norm_type == "l2":
				norms = values.square().sum(dim=(-2, -1)).sqrt()
			else:
				raise ValueError("norm_type must be 'l1' or 'l2'")
			layer_head_weights.append(norms / norms.sum().clamp_min(1e-8))
		return layer_head_weights

	def attention_rollout(self, attentions, head_weights, residual_ratio: float = 0.25):
		batch_size, _, num_tokens, _ = attentions[0].shape
		device = attentions[0].device
		rollout = torch.eye(num_tokens, device=device).unsqueeze(0).expand(batch_size, -1, -1)

		for attention, weights in zip(attentions, head_weights):
			weighted = (attention * weights.view(1, -1, 1, 1)).sum(dim=1)
			identity = torch.eye(num_tokens, device=device).unsqueeze(0)
			combined = weighted + residual_ratio * identity
			combined = combined / combined.sum(dim=-1, keepdim=True).clamp_min(1e-8)
			rollout = combined @ rollout
		return rollout

	def generate_saliency_map(self, image_tensor: torch.Tensor, target_category: int = None,
							  image_size: tuple = (32, 32)) -> torch.Tensor:
		attentions, gradients, _ = self._get_attention_matrices_and_grads(
			image_tensor, target_category
		)
		head_weights = self.compute_head_weights(gradients)
		rollout = self.attention_rollout(attentions, head_weights)
		cls_rollout = rollout[0, 0, 1:]
		grid_size = int(cls_rollout.numel() ** 0.5)
		if grid_size * grid_size != cls_rollout.numel():
			raise ValueError("Number of patch tokens must form a square grid")
		saliency_map = cls_rollout.reshape(1, 1, grid_size, grid_size)
		saliency_map = F.interpolate(saliency_map, size=image_size, mode="bicubic", align_corners=False)
		saliency_map = saliency_map.squeeze()
		saliency_map = (saliency_map - saliency_map.min()) / (saliency_map.max() - saliency_map.min() + 1e-8)
		self.model.zero_grad()
		return saliency_map


MILESTONE_PCTS = [15, 30, 45, 60, 75, 90]


def get_checkpoint_path(dataset: str, arch: str, paradigm: str, tag: str = "best") -> str:
	return os.path.join(DIR_CHECKPOINTS, f"{dataset}_{arch}_{paradigm}_{tag}.pt")


def list_periodic_checkpoints(dataset: str, arch: str, paradigm: str) -> List[str]:
	pattern = os.path.join(DIR_CHECKPOINTS, f"{dataset}_{arch}_{paradigm}_epoch_*.pt")
	return sorted(glob.glob(pattern))


def _epoch_from_checkpoint(path: str) -> int:
	return int(os.path.splitext(path)[0].rsplit("_", 1)[-1])


def select_and_prune_milestones(
	dataset: str,
	arch: str,
	paradigm: str,
	batch_size: int,
	device: torch.device,
	val_fraction: float = CONFIG["val_fraction"],
	probe_epochs: int = CONFIG["probe_epochs"],
	probe_lr: float = CONFIG["probe_lr"],
	num_slices: int = CONFIG["sigreg_slices"],
	t_max: float = CONFIG["sigreg_tmax"],
	n_points: int = CONFIG["sigreg_points"],
	lamb: float = CONFIG["lejepa_lambda"],
) -> Dict:
	"""Load all periodic checkpoints, probe them, select relative milestones, and prune the rest."""
	from src.evaluation import linear_probe

	paths = list_periodic_checkpoints(dataset, arch, paradigm)
	if not paths:
		raise FileNotFoundError(
			f"No periodic checkpoints found for {dataset}/{arch}/{paradigm}. "
			"Train first, or point this project at the checkpoint directory containing the periodic files."
		)

	set_seed(CONFIG["seed"])
	probe_train, probe_val, probe_test = get_dataloaders(
		dataset,
		batch_size=batch_size,
		paradigm="std",
		val_fraction=val_fraction,
		include_test=True,
	)

	records = []
	for path in paths:
		epoch = _epoch_from_checkpoint(path)
		model = build_model(
			arch,
			dataset,
			paradigm,
			num_slices=num_slices,
			t_max=t_max,
			n_points=n_points,
			lamb=lamb,
		).to(device)
		checkpoint = torch.load(path, map_location=device)
		model.load_state_dict(checkpoint["model_state_dict"])

		print(f"[Probe] Epoch {epoch}: {path}")
		result = linear_probe(
			model,
			probe_train,
			probe_val,
			probe_test,
			num_classes=DATASETS[dataset]["num_classes"],
			device=device,
			epochs=probe_epochs,
			lr=probe_lr,
		)

		records.append({
			"epoch": epoch,
			"path": path,
			"val_acc": float(result["best_val_acc"]),
			"test_acc": float(result["test_acc"]),
			"probe_best_epoch": int(result["best_epoch"]),
			"head_state_dict": result["head_state_dict"],
		})

		del model
		if torch.cuda.is_available():
			torch.cuda.empty_cache()

	# The best representation checkpoint is chosen only from probe validation accuracy.
	# Test accuracy is stored for reporting and never used for checkpoint selection.
	best = max(records, key=lambda record: (record["val_acc"], -record["epoch"]))
	chance = 100.0 / DATASETS[dataset]["num_classes"]
	accuracy_final = best["val_acc"]

	milestone_map = {}
	for milestone in MILESTONE_PCTS:
		target = chance + (milestone / 100.0) * (accuracy_final - chance)
		chosen = min(
			records,
			key=lambda record: (abs(record["val_acc"] - target), record["epoch"]),
		)
		milestone_map[milestone] = {
			"target_val_acc": target,
			"epoch": chosen["epoch"],
			"val_acc": chosen["val_acc"],
			"test_acc": chosen["test_acc"],
		}

	# Several relative milestones may legitimately map to the same periodic checkpoint.
	labels_by_epoch = {}
	for milestone, info in milestone_map.items():
		labels_by_epoch.setdefault(info["epoch"], []).append(milestone)

	# Retain every milestone checkpoint and the checkpoint with the best probe validation accuracy.
	# The separate *_best.pt recovery checkpoint is not part of this pruning process.
	keep_epochs = set(labels_by_epoch) | {best["epoch"]}
	retained = []

	for record in records:
		if record["epoch"] not in keep_epochs:
			os.remove(record["path"])
			continue

		checkpoint = torch.load(record["path"], map_location="cpu")
		checkpoint["linear_probe"] = {
			"best_val_acc": record["val_acc"],
			"test_acc": record["test_acc"],
			"best_probe_epoch": record["probe_best_epoch"],
			"head_state_dict": record["head_state_dict"],
		}
		checkpoint["milestones"] = sorted(labels_by_epoch.get(record["epoch"], []))
		checkpoint["is_best_probe_checkpoint"] = record["epoch"] == best["epoch"]
		torch.save(checkpoint, record["path"])
		retained.append(record["path"])

	summary = {
		"dataset": dataset,
		"arch": arch,
		"paradigm": paradigm,
		"chance_accuracy": chance,
		"accuracy_final": accuracy_final,
		"best_epoch": best["epoch"],
		"best_val_acc": best["val_acc"],
		"best_test_acc": best["test_acc"],
		"model_config": {
			"num_slices": num_slices,
			"t_max": t_max,
			"n_points": n_points,
			"lamb": lamb,
		},
		"milestones": milestone_map,
		"retained_checkpoints": retained,
		"all_probe_results": [
			{key: value for key, value in record.items() if key != "head_state_dict"}
			for record in records
		],
	}

	summary_path = os.path.join(
		DIR_CHECKPOINTS,
		f"{dataset}_{arch}_{paradigm}_milestones.json",
	)
	with open(summary_path, "w") as file:
		json.dump(summary, file, indent=2)

	print(f"[Milestones] Summary saved to {summary_path}")
	return summary


def denormalize(images: torch.Tensor, dataset_name: str) -> torch.Tensor:
	mean, std = get_or_compute_stats(dataset_name)
	mean_tensor = torch.tensor(mean, dtype=images.dtype).view(1, -1, 1, 1)
	std_tensor = torch.tensor(std, dtype=images.dtype).view(1, -1, 1, 1)
	return (images.cpu() * std_tensor + mean_tensor).clamp(0, 1)


def run_pca_for_milestones(
	summary: Dict,
	batch_size: int,
	device: torch.device,
	num_samples: int,
	val_fraction: float = CONFIG["val_fraction"],
):
	"""Run the original SVD-based spatial PCA on a fixed test subset for every milestone checkpoint."""
	from src.evaluation import pca_outputs

	if num_samples < 1:
		return

	dataset = summary["dataset"]
	arch = summary["arch"]
	paradigm = summary["paradigm"]
	model_config = summary.get("model_config", {})

	_, _, test_loader = get_dataloaders(
		dataset,
		batch_size=batch_size,
		paradigm="std",
		val_fraction=val_fraction,
		include_test=True,
	)

	milestone_epochs = sorted({info["epoch"] for info in summary["milestones"].values()})

	for epoch in milestone_epochs:
		path = next(
			checkpoint_path
			for checkpoint_path in summary["retained_checkpoints"]
			if _epoch_from_checkpoint(checkpoint_path) == epoch
		)

		checkpoint = torch.load(path, map_location=device)
		model = build_model(
			arch,
			dataset,
			paradigm,
			num_slices=model_config.get("num_slices", CONFIG["sigreg_slices"]),
			t_max=model_config.get("t_max", CONFIG["sigreg_tmax"]),
			n_points=model_config.get("n_points", CONFIG["sigreg_points"]),
			lamb=model_config.get("lamb", CONFIG["lejepa_lambda"]),
		).to(device)
		model.load_state_dict(checkpoint["model_state_dict"])
		model.eval()

		milestone_labels = sorted(checkpoint.get("milestones", []))
		output_root = os.path.join(
			DIR_OUTPUT,
			"pca",
			dataset,
			f"{arch}_{paradigm}",
			f"epoch_{epoch:04d}_milestones_{'-'.join(map(str, milestone_labels))}",
		)

		saved = 0
		sample_index = 0
		num_layers = None

		with torch.no_grad():
			for images, labels in test_loader:
				remaining = num_samples - saved
				if remaining <= 0:
					break

				images = images[:remaining]
				labels = labels[:remaining]
				features = model.forward_features(images.to(device, non_blocking=True))
				originals = denormalize(images, dataset)
				num_layers = len(features)

				for batch_index in range(images.size(0)):
					for layer_index, feature_map in enumerate(features):
						result = pca_outputs(feature_map, image_index=batch_index)
						result_dir = os.path.join(
							output_root,
							f"layer_{layer_index:02d}",
							f"sample_{sample_index:05d}",
						)
						os.makedirs(result_dir, exist_ok=True)

						metadata = {
							"dataset": dataset,
							"sample_index": sample_index,
							"label": int(labels[batch_index]),
							"architecture": arch,
							"paradigm": paradigm,
							"epoch": epoch,
							"milestones": milestone_labels,
							"layer_index": layer_index,
							"pca_components": 3,
							"checkpoint_path": path,
						}

						payload = {
							"original": originals[batch_index],
							"components": result["components"],
							"mask": result["mask"],
							"rgb": result["rgb"],
							"metadata": metadata,
						}

						torch.save(payload, os.path.join(result_dir, "pca.pt"))
						save_image(originals[batch_index], os.path.join(result_dir, "original.png"))
						save_image(result["rgb"].permute(2, 0, 1), os.path.join(result_dir, "pca_rgb.png"))
						save_image(result["mask"].float().unsqueeze(0), os.path.join(result_dir, "pca_mask.png"))
						with open(os.path.join(result_dir, "metadata.json"), "w") as file:
							json.dump(metadata, file, indent=2)

					sample_index += 1
					saved += 1

		print(
			f"[PCA] Epoch {epoch}: saved {saved} test samples "
			f"across {num_layers} feature layers to '{output_root}'."
		)


def test_cuda():
	print(f"PyTorch Version: {torch.__version__}")
	print(f"CUDA Available:  {torch.cuda.is_available()}")
	if torch.cuda.is_available():
		print(f"Device Name:     {torch.cuda.get_device_name(0)}")


def test_config(dataset: str, arch: str, paradigm: str):
	set_seed(CONFIG["seed"])
	print(f"Active Device:       {DEVICE}")
	print(f"Target Dataset:      {dataset}")
	print(f"Architecture:        {arch}")
	print(f"Training Paradigm:   {paradigm}")
	print(f"Number of Classes:   {DATASETS[dataset]['num_classes']}")


def test_pipeline(dataset: str, arch: str, paradigm: str):
	train_loader, _ = get_dataloaders(dataset_name=dataset, batch_size=8, paradigm=paradigm)
	batch, _ = next(iter(train_loader))
	model = build_model(arch=arch, dataset=dataset, paradigm=paradigm).to(DEVICE)
	if paradigm == "lejepa":
		global_views = [x.to(DEVICE) for x in batch["global"]]
		local_views = [x.to(DEVICE) for x in batch["local"]]
		loss, inv, sig = model(global_views=global_views, local_views=local_views)
		print(f"LeJEPA loss: {loss.item():.4f} | Inv: {inv.item():.4f} | SIGReg: {sig.item():.4f}")
		images = global_views[0]
	else:
		images = batch.to(DEVICE)
		print(f"Logits output shape: {model(images).shape}")
	for i, feature in enumerate(model.forward_features(images), 1):
		print(f"Layer {i} feature shape: {tuple(feature.shape)}")
