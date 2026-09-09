import glob
import json
import os
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.data import get_dataloaders, get_or_compute_stats
from src.globals import CONFIG, DATASETS, DEVICE, DIR_CHECKPOINTS, set_seed
from src.network import AttentionEncoder, build_model


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
	'''Semantic Alignment Score (SAS) tra Mappe XAI e Mappe Semantiche PCA.'''

	def __init__(self, num_bins: int = 16, threshold: float = 0.8):
		'''
		Args:
			num_bins (int): Numero di bin per la quantizzazione nell'Informazione Mutua.
			threshold (float): Soglia per la binarizzazione delle mappe di saliency.
		'''
		self.num_bins = num_bins
		self.threshold = threshold
		self.metrics_funct = {
			"jaccard": self.jaccard_index,
			"mse": self.MSE,
			"mae": self.MAE,
			"pearson": self.pearson_correlation,
			"spearman": self.spearman_correlation,
			"mi": self.mutual_information
		}
		self.metrics_list = list(self.metrics_funct.keys())

	def compute_sas(self, XAI_sal_map: torch.Tensor, PCA_sem_map: torch.Tensor) -> tuple[
		torch.Tensor, dict[str, torch.Tensor]]:
		"""
		Calcola il Semantic Alignment Score per un'immagine singola [C, H, W] o un batch [B, C, H, W].
		"""
		is_single_image = (XAI_sal_map.ndim == 3)

		if is_single_image:
			xai = XAI_sal_map.unsqueeze(0)
			pca = PCA_sem_map.unsqueeze(0)
		else:
			xai = XAI_sal_map
			pca = PCA_sem_map

		xai_flat = xai.flatten(start_dim=2)
		pca_flat = pca.flatten(start_dim=2)

		metrics_dict = {}
		# Fixed: access via self.metrics_list
		for metric in self.metrics_list:
			metrics_dict[metric] = self.metrics_funct[metric](xai_flat, pca_flat)

		metrics = torch.stack(list(metrics_dict.values()), dim=2)  # [B, C, M]

		if is_single_image:
			metrics = metrics.squeeze(0)

		return metrics, metrics_dict

	def jaccard_index(self, XAI_flat: torch.Tensor, PCA_flat: torch.Tensor) -> torch.Tensor:
		'''Jaccard Index (IoU) vettoriale lungo la dimensione spaziale. Restituisce [B, C].'''

		xai_b = (XAI_flat > self.threshold).float()
		pca_b = (PCA_flat > self.threshold).float()
		intersection = torch.sum(xai_b * pca_b, dim=-1)
		union = torch.sum(xai_b, dim=-1) + torch.sum(pca_b, dim=-1) - intersection
		return intersection / (union + 1e-8)

	def MSE(self, XAI_flat: torch.Tensor, PCA_flat: torch.Tensor) -> torch.Tensor:
		'''Mean Squared Error (MSE) vettoriale. Restituisce [B, C]. Non utile ma fa nulla'''
		return torch.mean((XAI_flat - PCA_flat) ** 2, dim=-1)

	def MAE(self, XAI_flat: torch.Tensor, PCA_flat: torch.Tensor) -> torch.Tensor:
		'''Mean Absolute Error (MAE) vettoriale. Restituisce [B, C]. Non utile ma fa nulla'''
		return torch.mean(torch.abs(XAI_flat - PCA_flat), dim=-1)

	def pearson_correlation(self, XAI_flat: torch.Tensor, PCA_flat: torch.Tensor) -> torch.Tensor:
		'''Correlazione di Pearson vettoriale lungo la dimensione spaziale. Restituisce [B, C].'''
		xai_mean = torch.mean(XAI_flat, dim=-1, keepdim=True)
		pca_mean = torch.mean(PCA_flat, dim=-1, keepdim=True)

		xai_dev = XAI_flat - xai_mean
		pca_dev = PCA_flat - pca_mean

		cov = torch.sum(xai_dev * pca_dev, dim=-1)
		var_xai = torch.sum(xai_dev ** 2, dim=-1)
		var_pca = torch.sum(pca_dev ** 2, dim=-1)

		return cov / (torch.sqrt(var_xai * var_pca) + 1e-8)

	def spearman_correlation(self, XAI_flat: torch.Tensor, PCA_flat: torch.Tensor) -> torch.Tensor:
		'''Correlazione di Spearman vettoriale (Pearson sui Ranghi). Restituisce [B, C].'''
		xai_rank = torch.argsort(torch.argsort(XAI_flat, dim=-1), dim=-1).float()  # we dont know what to do with ties
		pca_rank = torch.argsort(torch.argsort(PCA_flat, dim=-1), dim=-1).float()  # we dont know what to do with ties

		return self.pearson_correlation(xai_rank, pca_rank)

	def mutual_information(self, XAI_flat: torch.Tensor, PCA_flat: torch.Tensor, num_bins: int = 16) -> torch.Tensor:
		'''Informazione Mutua Vettoriale tramite Istogramma Congiunto One-Hot. Restituisce [B, C].'''
		B, C, N = XAI_flat.shape

		# 1. Discretizzazione nei bin [0, num_bins - 1]
		xai_bin = (XAI_flat * (num_bins - 1e-5)).clamp(0, num_bins - 1).long()  # range [0, 1] allegedly
		pca_bin = (PCA_flat * (num_bins - 1e-5)).clamp(0, num_bins - 1).long()  # range [0, 1] allegedly

		# 2. Indice congiunto
		joint_bin = xai_bin * num_bins + pca_bin

		# 3. Istogramma congiunto vettoriale con One-Hot Encoding
		one_hot = F.one_hot(joint_bin, num_classes=num_bins ** 2).float()
		joint_hist = one_hot.sum(dim=2)

		# 4. Probabilità congiunta e marginali
		joint_prob = (joint_hist / N).view(B, C, num_bins, num_bins)
		xai_marg = joint_prob.sum(dim=-1, keepdim=True)
		pca_marg = joint_prob.sum(dim=-2, keepdim=True)

		p_x_p_y = xai_marg * pca_marg

		# 5. Calcolo MI
		ratio = (joint_prob / (p_x_p_y + 1e-12)).clamp(min=1e-12)
		mi = torch.sum(joint_prob * torch.log(ratio), dim=(-2, -1))

		return mi

	def compute_metrics_agreement(self, sas_results: torch.Tensor) -> tuple[torch.Tensor, list[str]]:
		"""
		Calcola la matrice di correlazione di Pearson TRA le M metriche,
		calcolata separatamente PER OGNI LAYER (canale C) lungo il batch B.

		Args:
			sas_results (torch.Tensor): Tensore di forma [B, C, M]
										(Batch, Layer/Canali, Metriche).

		Returns:
			corr_matrix (torch.Tensor): Forma [C, M, M] (matrice MxM per ogni canale C).
			metric_names (list[str]): Nomi delle M metriche ordinate.
		"""
		if sas_results.ndim == 2:  # Caso in cui è stata usata un'unica immagine [C, M]
			raise ValueError(
				"Impossibile calcolare la correlazione su un'unica immagine (B=1). "
				"Per la correlazione serve un batch con B >= 2 immagini (forma [B, C, M])."
			)

		B, C, M = sas_results.shape
		if B < 2:
			raise ValueError(f"Servono almeno 2 immagini nel batch per la correlazione (ricevuto B={B}).")

		# 1. Centriamo i dati rispetto alla dimensione del batch (B) -> [B, C, M]
		mean = torch.mean(sas_results, dim=0, keepdim=True)  # [1, C, M]
		zero_mean = sas_results - mean  # [B, C, M]

		# 2. Riordiniamo le dimensioni per isolare i canali C -> [C, M, B]
		zero_mean_perm = zero_mean.permute(1, 2, 0)  # [C, M, B]

		# 3. Calcolo Covarianza per ogni canale: [C, M, B] x [C, B, M] -> [C, M, M]
		cov = torch.matmul(zero_mean_perm, zero_mean_perm.transpose(1, 2)) / (B - 1 + 1e-8)

		# 4. Deviazione Standard per ogni metrica lungo B -> [C, M]
		std = torch.std(sas_results, dim=0)

		# 5. Matrice dei prodotti delle deviazioni standard -> [C, M, M]
		# [C, M, 1] * [C, 1, M] mediante broadcasting -> [C, M, M]
		std_matrix = std.unsqueeze(2) * std.unsqueeze(1)

		# 6. Matrice di Correlazione [C, M, M]. Do not add epsilon directly
		# to the denominator: that biases correlations when a metric has a
		# small (but non-zero) variance and can make the diagonal differ from 1.
		denominator = std_matrix.clamp_min(torch.finfo(std_matrix.dtype).tiny)
		corr_matrix = cov / denominator
		corr_matrix = torch.where(std_matrix > 0, corr_matrix, torch.zeros_like(corr_matrix))

		# Clamp only floating-point roundoff outside [-1, 1].
		corr_matrix = torch.clamp(corr_matrix, -1.0, 1.0)

		return corr_matrix, self.metrics_list






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
