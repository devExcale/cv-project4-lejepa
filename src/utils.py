import glob
import json
import os
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from matplotlib import pyplot as plt

from src.data import get_dataloaders, get_or_compute_stats
from src.globals import CONFIG, DATASETS, DEVICE, DIR_CHECKPOINTS, DIR_OUTPUT, set_seed
from src.network import AttentionEncoder, LinearProbeModel, build_model


def _guided_act_backward_hook(module, grad_in, grad_out):
	"""
	Clamp negative input gradients for guided backpropagation.
	Works for both nn.ReLU and nn.GELU.
	"""
	if isinstance(grad_in[0], torch.Tensor):
		return (torch.clamp(grad_in[0], min=0.0),)
	return None


class GuidedBackprop:
	"""
	Guided Backpropagation hook manager supporting both CNNs (ReLU) and ViTs (GELU).
	"""

	def __init__(self, model: nn.Module):
		self.model = model
		self.hooks = []
		for module in self.model.modules():
			if isinstance(module, nn.ReLU):
				module.inplace = False

	def _register_hooks(self):
		for module in self.model.modules():
			if isinstance(module, (nn.ReLU, nn.GELU)):
				self.hooks.append(
					module.register_full_backward_hook(_guided_act_backward_hook)
				)

	def generate_gradients(
			self,
			input_tensor: torch.Tensor,
			target_class: int | torch.Tensor | None = None,
	) -> np.ndarray:
		"""Return guided input gradients as [B, H, W, C]."""
		self.model.eval()
		self._register_hooks()
		try:
			input_tensor = input_tensor.clone().detach().requires_grad_(True)
			self.model.zero_grad()
			output = self.model(input_tensor)
			targets = _resolve_target_classes(output, target_class)
			score = output.gather(1, targets.unsqueeze(1)).sum()
			score.backward()
			if input_tensor.grad is None:
				raise RuntimeError("Input gradients were not captured")
			return input_tensor.grad.detach().cpu().permute(0, 2, 3, 1).numpy()
		finally:
			self.remove_hooks()

	def remove_hooks(self):
		for hook in self.hooks:
			hook.remove()
		self.hooks = []


def _resolve_target_classes(
	logits: torch.Tensor,
	target_class: int | torch.Tensor | None,
) -> torch.Tensor:
	batch_size = logits.size(0)
	if target_class is None:
		return logits.argmax(dim=1)
	if isinstance(target_class, int):
		return torch.full(
			(batch_size,),
			target_class,
			device=logits.device,
			dtype=torch.long,
		)
	targets = torch.as_tensor(target_class, device=logits.device, dtype=torch.long).flatten()
	if targets.numel() != batch_size:
		raise ValueError(
			f"Expected one target class per sample ({batch_size}), got {targets.numel()}"
		)
	return targets


class GradCAM:
	"""Grad-CAM hook manager and batched saliency map generator for CNN backbones."""

	def __init__(self, model: nn.Module, target_layer: nn.Module):
		self.model = model
		self.target_layer: nn.Module = target_layer
		self.activations: Optional[torch.Tensor] = None
		self.gradients: Optional[torch.Tensor] = None
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
		target_class: int | torch.Tensor | None = None,
	) -> torch.Tensor:
		"""Generate one normalized Grad-CAM heatmap per input, shaped [B, H, W]."""
		self.model.eval()
		self.model.zero_grad()
		logits = self.model(input_tensor)
		targets = _resolve_target_classes(logits, target_class)
		logits.gather(1, targets.unsqueeze(1)).sum().backward()

		if self.gradients is None or self.activations is None:
			raise RuntimeError("Grad-CAM activations or gradients were not captured")

		weights = self.gradients.mean(dim=(2, 3), keepdim=True)
		cam = torch.sum(weights * self.activations, dim=1, keepdim=True)
		cam = F.relu(cam)
		_, _, height, width = input_tensor.shape
		cam = F.interpolate(cam, size=(height, width), mode="bilinear", align_corners=False)
		cam = cam.squeeze(1)

		flat = cam.flatten(1)
		cam_min = flat.min(dim=1).values.view(-1, 1, 1)
		cam_max = flat.max(dim=1).values.view(-1, 1, 1)
		denominator = cam_max - cam_min
		cam = torch.where(
			denominator > 0,
			(cam - cam_min) / denominator.clamp_min(1e-8),
			torch.zeros_like(cam),
		)
		return cam.detach().cpu()

	def remove_hooks(self):
		self.forward_handle.remove()
		self.backward_handle.remove()


class GMAR:
	def __init__(self, model: nn.Module):
		self.model = model
		self.attn_modules = [
			module for module in self.model.modules()
			if isinstance(module, AttentionEncoder)
		]
		# if not found, try inside the backbone (for LinearProbeModel or similar wrappers)
		if not self.attn_modules and hasattr(self.model, "backbone"):
			print("[GMAR] No AttentionEncoder modules found in the model; checking the backbone...")
			self.attn_modules = [
				module for module in self.model.backbone.modules()
				if isinstance(module, AttentionEncoder)
			]
		print(f"[GMAR] Found {len(self.attn_modules)} AttentionEncoder modules in the model")
		if not self.attn_modules:
			raise ValueError("GMAR requires a model containing AttentionEncoder modules")

	def _get_attention_matrices_and_grads(
		self,
		inputs: torch.Tensor,
		target_category: int | torch.Tensor | None = None,
	):
		self.model.eval()
		self.model.zero_grad()

		with torch.enable_grad():
			try:
				logits = self.model(inputs, need_attn=True)
			except TypeError as exc:
				raise TypeError(
					"Il modello passato a GMAR non accetta l'argomento `need_attn`. "
					"Passa la ViT nuda o un LinearProbeModel che lo inoltri al backbone."
				) from exc

			targets = _resolve_target_classes(logits, target_category)
			logits.gather(1, targets.unsqueeze(1)).sum().backward()

		attentions = []
		gradients = []
		for module in self.attn_modules:
			if module.attention is None or module.attention.grad is None:
				raise RuntimeError("Attention weights or gradients were not captured")
			attentions.append(module.attention.detach())
			gradients.append(module.attention.grad.detach())
			module.attention = None

		return attentions, gradients, targets

	def compute_head_weights(self, gradients: list[torch.Tensor], norm_type: str = "l1"):
		layer_head_weights = []
		for grad in gradients:
			if norm_type == "l1":
				norms = grad.abs().sum(dim=(-2, -1))
			elif norm_type == "l2":
				norms = grad.square().sum(dim=(-2, -1)).sqrt()
			else:
				raise ValueError("norm_type must be 'l1' or 'l2'")
			layer_head_weights.append(
				norms / norms.sum(dim=-1, keepdim=True).clamp_min(1e-8)
			)
		return layer_head_weights

	def attention_rollout(self, attentions, head_weights, residual_ratio: float = 0.25):
		batch_size, _, num_tokens, _ = attentions[0].shape
		device = attentions[0].device
		rollout = torch.eye(num_tokens, device=device).unsqueeze(0).expand(batch_size, -1, -1)

		for attention, weights in zip(attentions, head_weights):
			weighted = (attention * weights.unsqueeze(-1).unsqueeze(-1)).sum(dim=1)
			identity = torch.eye(num_tokens, device=device).unsqueeze(0)
			combined = weighted + residual_ratio * identity
			combined = combined / combined.sum(dim=-1, keepdim=True).clamp_min(1e-8)
			rollout = combined @ rollout
		return rollout

	def generate_saliency_map(
		self,
		image_tensor: torch.Tensor,
		target_category: int | torch.Tensor | None = None,
		image_size: tuple = (32, 32),
	) -> torch.Tensor:
		"""Return one normalized GMAR saliency map per input, shaped [B, H, W]."""
		attentions, gradients, _ = self._get_attention_matrices_and_grads(
			image_tensor,
			target_category,
		)
		head_weights = self.compute_head_weights(gradients)
		rollout = self.attention_rollout(attentions, head_weights)
		cls_rollout = rollout[:, 0, 1:]
		num_patches = cls_rollout.size(-1)
		grid_size = int(num_patches ** 0.5)
		if grid_size * grid_size != num_patches:
			raise ValueError("Number of patch tokens must form a square grid")

		saliency_map = cls_rollout.reshape(-1, 1, grid_size, grid_size)
		saliency_map = F.interpolate(
			saliency_map,
			size=image_size,
			mode="bicubic",
			align_corners=False,
		).squeeze(1)
		flat = saliency_map.flatten(1)
		minimum = flat.min(dim=1).values.view(-1, 1, 1)
		maximum = flat.max(dim=1).values.view(-1, 1, 1)
		saliency_map = (saliency_map - minimum) / (maximum - minimum).clamp_min(1e-8)
		self.model.zero_grad()
		return saliency_map.detach()



class SAS:
	'''Semantic Alignment Score'''
	def __init__(self):
		pass
	def compute_sas(self, XAI_sal_map: torch.Tensor, PCA_sem_map: torch.Tensor) -> float:
		"""
		Compute the Semantic Alignment Score (SAS) between XAI saliency maps and PCA semantic maps.
		Args:
			XAI_sal_map (torch.Tensor): Saliency map from XAI method, shape [B, H, W].
			PCA_sem_map (torch.Tensor): Semantic map from PCA, shape [B, H, W].

		Returns:
			float: The computed SAS.
		"""
		# Flatten the tensors
		XAI_flat = XAI_sal_map.flatten(1)
		PCA_flat = PCA_sem_map.flatten(1)

		# Compute the correlation
		corr = torch.corrcoef(torch.stack([XAI_flat, PCA_flat]))[0, 1]

		return corr.item()
	def jaccard_index(self, XAI_flat: torch.Tensor, PCA_flat: torch.Tensor) -> float:
		'''Compute the Jaccard Index between XAI saliency maps and PCA semantic maps.'''
	

		# Compute the Jaccard Index
		intersection = torch.sum(XAI_flat * PCA_flat)
		union = torch.sum(XAI_flat) + torch.sum(PCA_flat) - intersection

		return (intersection / union).item()
	def MSE(self, XAI_flat: torch.Tensor, PCA_flat: torch.Tensor) -> float:
		'''Compute the Mean Squared Error (MSE) between XAI saliency maps and PCA semantic maps.'''
		# Compute the Mean Squared Error
		return torch.mean((XAI_flat - PCA_flat) ** 2).item()
	def MAE(self, XAI_flat: torch.Tensor, PCA_flat: torch.Tensor) -> float:
		'''Compute the Mean Absolute Error (MAE) between XAI saliency maps and PCA semantic maps.'''
		# Compute the Mean Absolute Error
		return torch.mean(torch.abs(XAI_flat - PCA_flat)).item()
	def spearman_correlation(self, XAI_flat: torch.Tensor, PCA_flat: torch.Tensor) -> float:
		'''Compute the Spearman's rank correlation coefficient between XAI saliency maps and PCA semantic maps.'''
		# Compute ranks
		XAI_rank = torch.argsort(torch.argsort(XAI_flat))
		PCA_rank = torch.argsort(torch.argsort(PCA_flat))
		# Compute Spearman's rank correlation coefficient
		n = XAI_flat.numel()
		d = XAI_rank - PCA_rank
		spearman_corr = 1 - (6 * torch.sum(d ** 2)) / (n * (n ** 2 - 1))

		return spearman_corr.item()
	def pearson_correlation(self, XAI_flat: torch.Tensor, PCA_flat: torch.Tensor) -> float:
		'''Compute the Pearson correlation coefficient between XAI saliency maps and PCA semantic maps.'''
		# Compute means
		XAI_mean = torch.mean(XAI_flat)
		PCA_mean = torch.mean(PCA_flat)
		# Compute covariance and standard deviations
		covariance = torch.mean((XAI_flat - XAI_mean) * (PCA_flat - PCA_mean))
		XAI_std = torch.std(XAI_flat)
		PCA_std = torch.std(PCA_flat)
		# Compute Pearson correlation coefficient
		pearson_corr = covariance / (XAI_std * PCA_std)

		return pearson_corr.item()
	def mutual_information(self, XAI_flat: torch.Tensor, PCA_flat: torch.Tensor, num_bins: int = 20) -> float:
		'''Compute the Mutual Information (MI) between XAI saliency maps and PCA semantic maps.'''
		# Compute joint histogram
		joint_hist = torch.histc(XAI_flat * num_bins + PCA_flat, bins=num_bins**2, min=0, max=num_bins**2 - 1)
		joint_prob = joint_hist / torch.sum(joint_hist)
		joint_prob = joint_prob[joint_prob > 0]  # Remove zero probabilities

		# Compute marginal probabilities
		XAI_hist = torch.histc(XAI_flat, bins=num_bins, min=0, max=num_bins - 1)
		PCA_hist = torch.histc(PCA_flat, bins=num_bins, min=0, max=num_bins - 1)
		XAI_prob = XAI_hist / torch.sum(XAI_hist)
		PCA_prob = PCA_hist / torch.sum(PCA_hist)

		XAI_prob = XAI_prob[XAI_prob > 0]
		PCA_prob = PCA_prob[PCA_prob > 0]

		# Compute Mutual Information
		mi = torch.sum(joint_prob * torch.log(joint_prob / (XAI_prob.unsqueeze(1) * PCA_prob.unsqueeze(0)).clamp_min(1e-8)))

		return mi.item()
	def sq_sum_ratio(self, XAI_flat: torch.Tensor, PCA_flat: torch.Tensor) -> float:
		'''My invented metric'''
		nXAI, nPCA = (XAI_flat - torch.mean(XAI_flat)) / torch.std(XAI_flat), (PCA_flat - torch.mean(PCA_flat)) / torch.std(PCA_flat)
		squared_diff = (nXAI - nPCA) ** 2
		squared_sum = (nXAI + nPCA) ** 2
		sq_ratio = torch.sum(squared_sum / (squared_diff + 1e-8)) # if equal --> inf, if opposite --> 0
		sq_sum = torch.sum(squared_sum - squared_diff) # if equal

		return 






def load_probe_summary(dataset: str, arch: str, paradigm: str) -> Dict:
	path = os.path.join(DIR_CHECKPOINTS, f"{dataset}_{arch}_{paradigm}", "probe_results.json")
	if not os.path.exists(path):
		raise FileNotFoundError(
			f"No probe summary found for {dataset}/{arch}/{paradigm} at '{path}'. "
			"Run mode 'probe' first."
		)
	with open(path, "r", encoding="utf-8") as file:
		return json.load(file)


def build_relative_accuracy_comparison(dataset: str, arch: str) -> Dict:
	"""Match all STD and LeJEPA checkpoints by nearest relative probe accuracy."""
	std_summary = load_probe_summary(dataset, arch, "std")
	lejepa_summary = load_probe_summary(dataset, arch, "lejepa")
	std_records = list(std_summary["probe_results"])
	lejepa_records = list(lejepa_summary["probe_results"])

	std_to_lejepa = []
	for reference in std_records:
		valid = [record for record in lejepa_records if record.get("relative_accuracy") is not None]
		matched = None
		if reference.get("relative_accuracy") is not None and valid:
			matched = min(
				valid,
				key=lambda record: (
					abs(float(record["relative_accuracy"]) - float(reference["relative_accuracy"])),
					abs(int(record["epoch"]) - int(reference["epoch"])),
					int(record["epoch"]),
				),
			)
		pair = {
			"reference_epoch": int(reference["epoch"]),
			"reference_val_acc": float(reference["val_acc"]),
			"reference_relative_accuracy": reference.get("relative_accuracy"),
			"reference_checkpoint_path": reference["checkpoint_path"],
			"reference_probe_path": reference["probe_path"],
			"matched_epoch": None,
			"matched_val_acc": None,
			"matched_relative_accuracy": None,
			"matched_checkpoint_path": None,
			"matched_probe_path": None,
			"relative_accuracy_delta": None,
		}
		if matched is not None:
			pair.update({
				"matched_epoch": int(matched["epoch"]),
				"matched_val_acc": float(matched["val_acc"]),
				"matched_relative_accuracy": matched["relative_accuracy"],
				"matched_checkpoint_path": matched["checkpoint_path"],
				"matched_probe_path": matched["probe_path"],
				"relative_accuracy_delta": abs(
					float(reference["relative_accuracy"]) - float(matched["relative_accuracy"])
				),
			})
		std_to_lejepa.append(pair)

	lejepa_to_std = []
	for reference in lejepa_records:
		valid = [record for record in std_records if record.get("relative_accuracy") is not None]
		matched = None
		if reference.get("relative_accuracy") is not None and valid:
			matched = min(
				valid,
				key=lambda record: (
					abs(float(record["relative_accuracy"]) - float(reference["relative_accuracy"])),
					abs(int(record["epoch"]) - int(reference["epoch"])),
					int(record["epoch"]),
				),
			)
		pair = {
			"reference_epoch": int(reference["epoch"]),
			"reference_val_acc": float(reference["val_acc"]),
			"reference_relative_accuracy": reference.get("relative_accuracy"),
			"reference_checkpoint_path": reference["checkpoint_path"],
			"reference_probe_path": reference["probe_path"],
			"matched_epoch": None,
			"matched_val_acc": None,
			"matched_relative_accuracy": None,
			"matched_checkpoint_path": None,
			"matched_probe_path": None,
			"relative_accuracy_delta": None,
		}
		if matched is not None:
			pair.update({
				"matched_epoch": int(matched["epoch"]),
				"matched_val_acc": float(matched["val_acc"]),
				"matched_relative_accuracy": matched["relative_accuracy"],
				"matched_checkpoint_path": matched["checkpoint_path"],
				"matched_probe_path": matched["probe_path"],
				"relative_accuracy_delta": abs(
					float(reference["relative_accuracy"]) - float(matched["relative_accuracy"])
				),
			})
		lejepa_to_std.append(pair)

	comparison = {
		"dataset": dataset,
		"arch": arch,
		"chance_accuracy": float(std_summary["chance_accuracy"]),
		"relative_accuracy_definition": (
			"100 * (val_acc - chance_accuracy) / (accuracy_final - chance_accuracy)"
		),
		"std_accuracy_final": float(std_summary["accuracy_final"]),
		"lejepa_accuracy_final": float(lejepa_summary["accuracy_final"]),
		"std_to_lejepa": std_to_lejepa,
		"lejepa_to_std": lejepa_to_std,
	}

	path = os.path.join(DIR_CHECKPOINTS, f"{dataset}_{arch}_relative_accuracy_comparison.json")
	with open(path, "w", encoding="utf-8") as file:
		json.dump(comparison, file, indent=2)
	print(f"[Relative Accuracy] Comparison saved to {path}")
	return comparison


def probe_all_checkpoints(
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
	resume: bool = False,
) -> Dict:
	"""Train/resume one persistent linear probe for every periodic backbone checkpoint."""
	from src.evaluation import linear_probe

	root = os.path.join(DIR_CHECKPOINTS, f"{dataset}_{arch}_{paradigm}")
	paths = sorted(glob.glob(os.path.join(root, "epoch_*", "checkpoint_*.pt")))
	if not paths:
		raise FileNotFoundError(
			f"No periodic checkpoints found for {dataset}/{arch}/{paradigm}. Train first."
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
	for checkpoint_path in paths:
		epoch_dir = os.path.basename(os.path.dirname(checkpoint_path))
		epoch = int(epoch_dir.removeprefix("epoch_"))
		probe_path = os.path.join(os.path.dirname(checkpoint_path), f"probe_{epoch:04d}.pt")

		model = build_model(
			arch,
			dataset,
			paradigm,
			num_slices=num_slices,
			t_max=t_max,
			n_points=n_points,
			lamb=lamb,
		).to(device)
		checkpoint = torch.load(checkpoint_path, map_location=device)
		model.load_state_dict(checkpoint["model_state_dict"])

		print(f"[Probe] Backbone epoch {epoch}: {checkpoint_path}")
		result = linear_probe(
			model,
			probe_train,
			probe_val,
			probe_test,
			num_classes=DATASETS[dataset]["num_classes"],
			device=device,
			epochs=probe_epochs,
			lr=probe_lr,
			probe_checkpoint_path=probe_path,
			resume=resume,
			metadata={
				"dataset": dataset,
				"arch": arch,
				"paradigm": paradigm,
				"backbone_epoch": epoch,
				"backbone_checkpoint_path": checkpoint_path,
			},
		)
		# linear_probe() persists its state every probe epoch and writes completed=True
		# before returning. Verify that durable file before advancing to the next backbone.
		if not os.path.exists(probe_path):
			raise RuntimeError(f"Probe finished but was not saved: '{probe_path}'")
		probe_checkpoint = torch.load(probe_path, map_location="cpu")
		if not probe_checkpoint.get("completed", False):
			raise RuntimeError(f"Probe returned without a completed checkpoint: '{probe_path}'")

		record = {
			"epoch": epoch,
			"checkpoint_path": checkpoint_path,
			"probe_path": probe_path,
			"val_acc": float(result["best_val_acc"]),
			"test_acc": float(result["test_acc"]),
			"test_loss": float(result["test_loss"]),
			"probe_best_epoch": int(result["best_epoch"]),
			"probe_stopped_epoch": int(result.get("probe_epoch", result["best_epoch"])),
			"probe_stop_reason": result.get("stop_reason", "completed"),
			"probe_converged": bool(result.get("converged", False)),
		}
		records.append(record)

		# Persist the probe/backbone association immediately. Relative accuracy is
		# filled in after the full trajectory is known, but completed probe work is
		# never held only in memory.
		checkpoint = torch.load(checkpoint_path, map_location="cpu")
		checkpoint["probe"] = {
			"path": probe_path,
			"completed": True,
			"best_val_acc": record["val_acc"],
			"test_acc": record["test_acc"],
			"best_probe_epoch": record["probe_best_epoch"],
			"relative_accuracy": None,
		}
		torch.save(checkpoint, checkpoint_path)
		print(f"[Probe] Saved completed probe immediately: {probe_path}")

		del model
		if torch.cuda.is_available():
			torch.cuda.empty_cache()

	best = max(records, key=lambda record: (record["val_acc"], -record["epoch"]))
	chance_accuracy = 100.0 / DATASETS[dataset]["num_classes"]
	accuracy_final = float(best["val_acc"])
	denominator = accuracy_final - chance_accuracy

	for record in records:
		record["relative_accuracy"] = (
			None
			if denominator <= 0.0
			else 100.0 * (float(record["val_acc"]) - chance_accuracy) / denominator
		)

		probe = torch.load(record["probe_path"], map_location="cpu")
		probe["chance_accuracy"] = chance_accuracy
		probe["accuracy_final"] = accuracy_final
		probe["relative_accuracy"] = record["relative_accuracy"]
		torch.save(probe, record["probe_path"])

		checkpoint = torch.load(record["checkpoint_path"], map_location="cpu")
		checkpoint["probe"] = {
			"path": record["probe_path"],
			"completed": True,
			"best_val_acc": record["val_acc"],
			"test_acc": record["test_acc"],
			"best_probe_epoch": record["probe_best_epoch"],
			"relative_accuracy": record["relative_accuracy"],
		}
		torch.save(checkpoint, record["checkpoint_path"])

	summary = {
		"dataset": dataset,
		"arch": arch,
		"paradigm": paradigm,
		"chance_accuracy": chance_accuracy,
		"accuracy_final": accuracy_final,
		"relative_accuracy_definition": (
			"100 * (val_acc - chance_accuracy) / (accuracy_final - chance_accuracy)"
		),
		"last_probed_epoch": max(record["epoch"] for record in records),
		"best_epoch": int(best["epoch"]),
		"best_val_acc": float(best["val_acc"]),
		"best_test_acc": float(best["test_acc"]),
		"best_checkpoint_path": best["checkpoint_path"],
		"best_probe_path": best["probe_path"],
		"model_config": {
			"num_slices": num_slices,
			"t_max": t_max,
			"n_points": n_points,
			"lamb": lamb,
		},
		"probe_config": {
			"epochs": probe_epochs,
			"lr": probe_lr,
			"convergence_cutoff": CONFIG["probe_convergence_cutoff"],
			"convergence_patience": CONFIG["probe_convergence_patience"],
		},
		"probe_results": records,
	}

	summary_path = os.path.join(root, "probe_results.json")
	with open(summary_path, "w", encoding="utf-8") as file:
		json.dump(summary, file, indent=2)
	print(f"[Probes] Summary saved to {summary_path}")

	other_paradigm = "lejepa" if paradigm == "std" else "std"
	other_summary_path = os.path.join(
		DIR_CHECKPOINTS,
		f"{dataset}_{arch}_{other_paradigm}",
		"probe_results.json",
	)
	if os.path.exists(other_summary_path):
		build_relative_accuracy_comparison(dataset, arch)

	return summary

def denormalize(
	images: torch.Tensor,
	dataset_name: str,
	val_fraction: float = CONFIG["val_fraction"],
) -> torch.Tensor:
	mean, std = get_or_compute_stats(dataset_name, val_fraction=val_fraction)
	mean_tensor = torch.tensor(mean, dtype=images.dtype).view(1, -1, 1, 1)
	std_tensor = torch.tensor(std, dtype=images.dtype).view(1, -1, 1, 1)
	return (images.cpu() * std_tensor + mean_tensor).clamp(0, 1)


def _get_spatial_feature_maps(model: nn.Module, images: torch.Tensor) -> tuple[torch.Tensor, ...]:
	"""Normalize CNN/ViT ``forward_features`` outputs to spatial [B, C, H, W] maps."""
	features = model.forward_features(images)

	# ViT returns (tuple(spatial_maps), tuple(class_tokens)).
	if (
		isinstance(features, tuple)
		and len(features) == 2
		and isinstance(features[0], (tuple, list))
	):
		features = features[0]

	if isinstance(features, torch.Tensor):
		features = (features,)

	if not isinstance(features, (tuple, list)) or not features:
		raise ValueError("forward_features() did not return any spatial feature maps")

	spatial_maps = tuple(features)
	for layer_index, feature_map in enumerate(spatial_maps):
		if not isinstance(feature_map, torch.Tensor) or feature_map.ndim != 4:
			raise ValueError(
				f"PCA requires spatial [B, C, H, W] feature maps; layer {layer_index} "
				f"returned {type(feature_map).__name__} with shape "
				f"{getattr(feature_map, 'shape', None)}"
			)
	return spatial_maps


def _run_pca_checkpoint(
	checkpoint_path: str,
	dataset: str,
	arch: str,
	paradigm: str,
	test_loader,
	device: torch.device,
	num_samples: int,
	val_fraction: float,
	model_config: Dict,
	output_name: str,
	probe_record: Dict | None = None,
	plot: bool = False,
):
	"""Run spatial PCA for one checkpoint and retain both correct and missed predictions."""
	from src.evaluation import pca_outputs

	checkpoint = torch.load(checkpoint_path, map_location=device)
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

	epoch = int(checkpoint.get("epoch", -1))
	probe_record = probe_record or {}
	probe_path = probe_record.get("probe_path")
	if not probe_path or not os.path.exists(probe_path):
		raise FileNotFoundError(
			f"PCA requires the completed linear probe for backbone epoch {epoch}, "
			f"but it was not found at '{probe_path}'. Run mode 'probe' first."
		)

	probe_checkpoint = torch.load(probe_path, map_location="cpu")
	if not probe_checkpoint.get("completed", False):
		raise RuntimeError(
			f"Probe for backbone epoch {epoch} is incomplete. "
			"Resume mode 'probe' before running PCA."
		)

	backbone = model.backbone if paradigm == "lejepa" else model
	head = nn.Linear(backbone.embed_dim, DATASETS[dataset]["num_classes"]).to(device)
	head.load_state_dict(probe_checkpoint["head_state_dict"])
	probe_model = LinearProbeModel(backbone, head).to(device)
	probe_model.eval()

	output_root = os.path.join(
		DIR_OUTPUT,
		"pca",
		dataset,
		f"{arch}_{paradigm}",
		output_name,
	)
	os.makedirs(output_root, exist_ok=True)

	saved = 0
	correct_saved = 0
	missed_saved = 0
	sample_index = 0
	num_layers = None
	plot_originals: list[torch.Tensor] = []
	plot_labels: list[int] = []
	plot_preds: list[int] = []
	plot_correct: list[bool] = []
	plot_rgb_by_sample: list[list[torch.Tensor]] = []

	with torch.no_grad():
		for images, labels in test_loader:
			remaining = num_samples - saved
			if remaining <= 0:
				break

			# As in the original PCA path, num_samples is the total number of test
			# samples considered. Each one is then routed by prediction correctness.
			images = images[:remaining]
			labels = labels[:remaining]
			inputs = images.to(device, non_blocking=True)
			targets = labels.to(device, non_blocking=True)

			preds = probe_model(inputs).argmax(dim=-1)
			features = _get_spatial_feature_maps(probe_model, inputs)
			originals = denormalize(images, dataset, val_fraction=val_fraction)
			num_layers = len(features)

			for batch_index in range(images.size(0)):
				sample_rgb: list[torch.Tensor] = []
				true_label = int(labels[batch_index])
				pred_label = int(preds[batch_index].item())
				is_correct = pred_label == true_label
				status_dir = "correct" if is_correct else "missed"

				for layer_index, feature_map in enumerate(features):
					result = pca_outputs(feature_map, image_index=batch_index)

					if plot:
						sample_rgb.append(result["rgb"])
						continue

					result_dir = os.path.join(
						output_root,
						status_dir,
						f"layer_{layer_index:02d}",
						f"sample_{sample_index:05d}",
					)
					os.makedirs(result_dir, exist_ok=True)

					metadata = {
						"dataset": dataset,
						"sample_index": sample_index,
						"label": true_label,
						"prediction": pred_label,
						"correct": is_correct,
						"architecture": arch,
						"paradigm": paradigm,
						"epoch": epoch,
						"layer_index": layer_index,
						"pca_components": 3,
						"checkpoint_path": checkpoint_path,
						"probe_path": probe_path,
						"probe_val_acc": probe_record.get("val_acc"),
						"probe_test_acc": probe_record.get("test_acc"),
						"relative_accuracy": probe_record.get("relative_accuracy"),
					}

					payload = {
						"original": originals[batch_index],
						"components": result["components"],
						"mask": result["mask"],
						"rgb": result["rgb"],
						"metadata": metadata,
					}
					torch.save(payload, os.path.join(result_dir, "pca.pt"))

				if plot:
					plot_originals.append(originals[batch_index])
					plot_labels.append(true_label)
					plot_preds.append(pred_label)
					plot_correct.append(is_correct)
					plot_rgb_by_sample.append(sample_rgb)

				if is_correct:
					correct_saved += 1
				else:
					missed_saved += 1
				sample_index += 1
				saved += 1

	if saved == 0 or num_layers is None:
		raise ValueError("The provided test DataLoader is empty.")

	if not plot:
		print(
			f"[PCA] Epoch {epoch}: saved {saved} test samples "
			f"({correct_saved} correct, {missed_saved} missed) across {num_layers} "
			f"feature layers to '{output_root}'."
		)
		return output_root

	# PCA does not reconstruct the input image. The first three PCA score maps are
	# normalized independently and displayed as pseudo-RGB channels (PC1/PC2/PC3).
	fig, axes = plt.subplots(
		saved,
		1 + num_layers,
		figsize=(2.8 * (1 + num_layers), 2.8 * saved),
		squeeze=False,
	)
	layer_word = "Block" if arch == "vit" else "Stage"

	for row in range(saved):
		original = plot_originals[row].permute(1, 2, 0).numpy()
		axes[row, 0].imshow(original)
		status = "[CORRECT]" if plot_correct[row] else f"[MISS: Pred {plot_preds[row]}]"
		axes[row, 0].set_title(
			f"Sample {row + 1}\nLabel {plot_labels[row]} {status}",
			fontsize=9,
		)
		axes[row, 0].axis("off")

		for layer_index, rgb in enumerate(plot_rgb_by_sample[row]):
			axes[row, layer_index + 1].imshow(rgb.numpy())
			axes[row, layer_index + 1].set_title(
				f"{layer_word} {layer_index + 1}\nPCA RGB (PC1/2/3)",
				fontsize=9,
			)
			axes[row, layer_index + 1].axis("off")

	plt.tight_layout()
	output_filepath = os.path.join(output_root, "pca_preview.png")
	fig.savefig(output_filepath, dpi=300, bbox_inches="tight")
	plt.close(fig)
	print(
		f"[PCA Complete] Saved {saved} samples ({correct_saved} correct, "
		f"{missed_saved} missed). Visualization saved to: {output_filepath}"
	)
	return output_filepath


def run_pca_for_all_checkpoints(
	summary: Dict,
	batch_size: int,
	device: torch.device,
	num_samples: int,
	val_fraction: float = CONFIG["val_fraction"],
	plot: bool = False,
):
	"""Run PCA on every backbone checkpoint represented in the probe summary."""
	if num_samples < 1:
		return []

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

	outputs = []
	for record in summary["probe_results"]:
		epoch = int(record["epoch"])
		relative = record.get("relative_accuracy")
		relative_tag = "na" if relative is None else f"{float(relative):06.2f}"
		outputs.append(_run_pca_checkpoint(
			record["checkpoint_path"],
			dataset,
			arch,
			paradigm,
			test_loader,
			device,
			num_samples,
			val_fraction,
			model_config,
			f"epoch_{epoch:04d}_relative_{relative_tag}",
			probe_record=record,
			plot=plot,
		))
	return outputs


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
