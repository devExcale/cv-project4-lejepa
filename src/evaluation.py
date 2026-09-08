import json
import math
import os
from copy import deepcopy
from typing import Tuple, cast

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from matplotlib import pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.data import get_or_compute_stats
from src.globals import CONFIG, DATASETS, DIR_DATA, DIR_OUTPUT
from src.utils import GradCAM, GuidedBackprop, GMAR


def spatial_pca(feature_map: torch.Tensor, k: int = 3, image_index: int = 0) -> torch.Tensor:
	if feature_map.ndim != 4:
		raise ValueError(f"Expected [B, C, H, W], got {tuple(feature_map.shape)}")
	if not 0 <= image_index < feature_map.size(0):
		raise IndexError(f"image_index {image_index} is outside batch size {feature_map.size(0)}")

	x = feature_map[image_index].detach()  # [C, H, W]
	c, h, w = x.shape
	x = x.permute(1, 2, 0).reshape(h * w, c).float()
	x = x - x.mean(dim=0, keepdim=True)

	_, _, vh = torch.linalg.svd(x, full_matrices=False)
	loadings = vh[:k].transpose(0, 1)

	anchors = loadings.abs().argmax(dim=0)
	signs = torch.sign(loadings[anchors, torch.arange(k, device=loadings.device)])
	signs = torch.where(signs == 0, torch.ones_like(signs), signs)
	loadings = loadings * signs

	projected = x @ loadings
	return projected.reshape(h, w, k).cpu()


def _normalize_map(x: torch.Tensor) -> torch.Tensor:
	minimum = x.min()
	maximum = x.max()
	return (x - minimum) / (maximum - minimum).clamp_min(1e-8)


def pca_outputs(
	feature_map: torch.Tensor,
	image_index: int = 0,
	output_size: tuple[int, int] | None = None,
) -> dict[str, torch.Tensor]:
	"""Run PCA once and derive comparable semantic outputs plus an RGB visualization.

	Returns:
		components: raw PCA scores at feature-map resolution, shaped [Hf, Wf, 3]
		pc1: normalized first principal component at feature-map resolution, [Hf, Wf]
		semantic_map: normalized PC1 optionally resized to ``output_size``, [H, W]
		mask: binary threshold of ``semantic_map`` at its mean, [H, W]
		rgb: normalized pseudo-RGB visualization (PC1/PC2/PC3), [Hf, Wf, 3]
	"""
	projected = spatial_pca(feature_map, k=3, image_index=image_index)
	pc1 = _normalize_map(projected[..., 0])

	semantic_map = pc1
	if output_size is not None:
		semantic_map = F.interpolate(
			pc1.unsqueeze(0).unsqueeze(0),
			size=output_size,
			mode="bilinear",
			align_corners=False,
		).squeeze(0).squeeze(0)
		semantic_map = _normalize_map(semantic_map)

	mask = semantic_map > semantic_map.mean()

	mins = projected.amin(dim=(0, 1), keepdim=True)
	maxs = projected.amax(dim=(0, 1), keepdim=True)
	rgb = (projected - mins) / (maxs - mins).clamp_min(1e-8)
	return {
		"components": projected,
		"pc1": pc1,
		"semantic_map": semantic_map,
		"mask": mask,
		"rgb": rgb,
	}


def pca_mask(
	feature_map: torch.Tensor,
	image_index: int = 0,
	output_size: tuple[int, int] | None = None,
) -> torch.Tensor:
	return pca_outputs(feature_map, image_index=image_index, output_size=output_size)["mask"]


def pca_rgb(feature_map: torch.Tensor, image_index: int = 0) -> torch.Tensor:
	return pca_outputs(feature_map, image_index=image_index)["rgb"]


def get_dataset_class_names(dataset_name: str) -> list[str]:
	dataset_name = dataset_name.lower()
	if dataset_name not in DATASETS:
		return []

	meta_data = DATASETS[dataset_name]
	label_key = meta_data["label_key"]

	try:
		from datasets import load_dataset_builder

		builder = load_dataset_builder(meta_data["hf_path"], cache_dir=DIR_DATA)
		features = builder.info.features
		if label_key in features and hasattr(features[label_key], "names"):
			return list(features[label_key].names)
	except Exception:
		pass

	for root, _, files in os.walk(DIR_DATA):
		if "dataset_info.json" not in files:
			continue
		path = os.path.join(root, "dataset_info.json")
		try:
			with open(path, "r", encoding="utf-8") as f:
				info = json.load(f)
			target_features = info.get("features", {})
			if label_key in target_features and "names" in target_features[label_key]:
				return list(target_features[label_key]["names"])
		except Exception:
			continue

	return []


def get_class_name(dataset_name: str, class_idx: int) -> str:
	class_names = get_dataset_class_names(dataset_name)
	if 0 <= class_idx < len(class_names):
		return class_names[class_idx]
	return f"Class {class_idx}"


def _save_saliency_pdf(
		output_filepath: str,
		originals: np.ndarray,
		labels: list[int],
		preds: list[int],
		dataset_name: str,
		heatmaps_by_layer: list[np.ndarray],
		layer_names: list[str],
		guided_grads: np.ndarray | None = None,
		method_name: str = "CAM",
		samples_per_page: int = 10,
		dpi: int = 120,
) -> None:
	"""
	Renders a multi-page PDF using PdfPages, generating small page figures
	and immediately closing them to prevent Python & PDF viewer memory blowups.
	"""

	total_samples = len(originals)
	if total_samples == 0:
		return

	num_layers = len(layer_names)
	has_guided = guided_grads is not None
	num_cols = 1 + (2 * num_layers if has_guided else num_layers)
	num_pages = math.ceil(total_samples / samples_per_page)

	os.makedirs(os.path.dirname(output_filepath), exist_ok=True)

	with PdfPages(output_filepath) as pdf:
		for page_idx in range(num_pages):
			start_i = page_idx * samples_per_page
			end_i = min(start_i + samples_per_page, total_samples)
			page_samples = end_i - start_i

			# Create a manageable figure for just this page
			fig, axes = plt.subplots(
				page_samples,
				num_cols,
				figsize=(2.4 * num_cols, 2.5 * page_samples),
				squeeze=False,
			)

			for row_idx, sample_i in enumerate(range(start_i, end_i)):
				true_label = int(labels[sample_i])
				pred_label = int(preds[sample_i])
				true_label_name = get_class_name(dataset_name, true_label)
				pred_label_name = get_class_name(dataset_name, pred_label)
				is_correct = (pred_label == true_label)

				status_tag = "[CORRECT]" if is_correct else f"[MISS: Pred {pred_label_name}]"
				status_color = "darkgreen" if is_correct else "crimson"

				# Col 0: Input image
				axes[row_idx, 0].imshow(originals[sample_i], rasterized=True)
				axes[row_idx, 0].set_title(
					f"#{sample_i + 1}: {true_label_name}\n{status_tag}",
					fontsize=8,
					fontweight="bold",
					color=status_color,
				)
				axes[row_idx, 0].axis("off")

				# Columns for each layer / block
				for l_idx, layer_name in enumerate(layer_names):
					cam = heatmaps_by_layer[l_idx][sample_i]

					if has_guided:
						col_heat = 1 + 2 * l_idx
						col_guided = 2 + 2 * l_idx

						# Rollout Heatmap
						axes[row_idx, col_heat].imshow(cam, cmap="jet", rasterized=True)
						axes[row_idx, col_heat].set_title(f"{layer_name}\nHeatmap", fontsize=8)
						axes[row_idx, col_heat].axis("off")

						# Guided overlay
						guided = guided_grads[sample_i] * cam[..., np.newaxis]
						guided -= guided.mean()
						guided /= (guided.std() + 1e-8)
						guided = np.clip(guided * 0.15 + 0.5, 0.0, 1.0)

						axes[row_idx, col_guided].imshow(guided, rasterized=True)
						axes[row_idx, col_guided].set_title(f"{layer_name}\nGuided {method_name}", fontsize=8)
						axes[row_idx, col_guided].axis("off")
					else:
						col_heat = 1 + l_idx
						axes[row_idx, col_heat].imshow(cam, cmap="jet", rasterized=True)
						axes[row_idx, col_heat].set_title(f"{layer_name}\n{method_name}", fontsize=8)
						axes[row_idx, col_heat].axis("off")

			# Save the individual page figure into the multi-page stream
			plt.tight_layout()
			pdf.savefig(fig, dpi=dpi, bbox_inches="tight")

			# Destroy the figure and free buffers
			plt.close(fig)

	return


def run_gradcam_pipeline(
		model: nn.Module,
		loader: DataLoader,
		dataset_name: str,
		arch: str,
		paradigm: str,
		device: torch.device,
		val_fraction: float = CONFIG["val_fraction"],
		output_name: str | None = None,
		plot: bool = True,
		resume: bool = True,
) -> str:
	"""
	Extract Grad-CAM maps across all 4 stages with correct/miss sample labels.
	"""

	if loader is None:
		raise ValueError("A DataLoader must be provided.")

	model.eval().to(device)
	backbone = getattr(model, "backbone", model)

	stages = ["layer1", "layer2", "layer3", "layer4"]
	for stage in stages:
		if not hasattr(backbone, stage):
			raise ValueError(f"Grad-CAM requires backbone stage '{stage}'")

	target_layers = [
		cast(nn.Module, list(getattr(backbone, stage).children())[-1])
		for stage in stages
	]

	model_id = f"{dataset_name}_{arch}_{paradigm}"
	if output_name:
		model_id = f"{model_id}_{output_name}"
	output_stem = f"gradcam_{model_id}"
	output_dir = os.path.join(DIR_OUTPUT, "gradcam", model_id)
	os.makedirs(output_dir, exist_ok=True)

	# --- Individual tensor heatmaps (no plot) ---
	if not plot:
		correct_dir = os.path.join(output_dir, "correct")
		missed_dir = os.path.join(output_dir, "missed")
		os.makedirs(correct_dir, exist_ok=True)
		os.makedirs(missed_dir, exist_ok=True)

		class_counters: dict[int, int] = {}
		total_saved = 0
		total_skipped = 0

		for images, labels in loader:
			batch_size = images.size(0)
			needed_indices = []
			sample_filenames = []

			# Check cache deterministically across both correct and missed folders
			for b in range(batch_size):
				true_label = int(labels[b])
				point_idx = class_counters.get(true_label, 0)
				filename = f"{output_stem}_c{true_label}_{point_idx}.pt"
				sample_filenames.append(filename)
				class_counters[true_label] = point_idx + 1

				cached_correct = os.path.join(correct_dir, filename)
				cached_missed = os.path.join(missed_dir, filename)

				if resume and (os.path.exists(cached_correct) or os.path.exists(cached_missed)):
					total_skipped += 1
				else:
					needed_indices.append(b)

			# If all samples in this batch already exist, skip computation
			if not needed_indices:
				continue

			# Slice batch to compute Grad-CAM for missing samples
			sub_inputs = images[needed_indices].to(device)
			sub_targets = labels[needed_indices].to(device)

			with torch.no_grad():
				preds = model(sub_inputs).argmax(dim=-1)

			batch_cams = []
			for target_layer in target_layers:
				grad_cam = GradCAM(model=model, target_layer=target_layer)
				try:
					cams = grad_cam.generate_cam(sub_inputs, target_class=sub_targets)
					batch_cams.append(cams)
				finally:
					grad_cam.remove_hooks()

			# Stack along final axis: [len(needed_indices), 32, 32, 4]
			stacked_cams = torch.stack(batch_cams, dim=-1)

			# Route each tensor into correct/ or missed/
			for idx, b in enumerate(needed_indices):
				is_correct = (preds[idx].item() == int(labels[b]))
				target_dir = correct_dir if is_correct else missed_dir
				torch.save(stacked_cams[idx], os.path.join(target_dir, sample_filenames[b]))
				total_saved += 1

		print(f"[Grad-CAM] Directory: {output_dir} | Saved: {total_saved} new | Skipped: {total_skipped} existing")
		return output_dir

	# --- Preview plot (gradcam heatmaps + guided backprop) ---
	output_filepath = os.path.join(output_dir, f"{output_stem}.pdf")
	if resume and os.path.exists(output_filepath):
		print(f"[Grad-CAM Resume] Visualizations already exist at: {output_filepath}. Skipping.")
		return output_filepath

	mean, std = get_or_compute_stats(dataset_name, val_fraction=val_fraction)
	mean_array = np.array(mean).reshape(1, 3, 1, 1)
	std_array = np.array(std).reshape(1, 3, 1, 1)

	guided_bp = GuidedBackprop(model=model)
	collected_originals, collected_labels, collected_preds = [], [], []
	collected_guided, collected_cams = [], [[] for _ in stages]

	for images, labels in loader:
		inputs = images.to(device)
		targets = labels.to(device)

		# 1. Capture model predictions for correctness badges
		with torch.no_grad():
			preds = model(inputs).argmax(dim=-1)
		collected_preds.extend(preds.cpu().tolist())

		# 2. Guided Backprop & Stage Grad-CAMs
		guided_grads = guided_bp.generate_gradients(inputs, target_class=targets)
		collected_guided.append(guided_grads)

		for stage_idx, target_layer in enumerate(target_layers):
			grad_cam = GradCAM(model=model, target_layer=target_layer)
			try:
				collected_cams[stage_idx].append(grad_cam.generate_cam(inputs, target_class=targets).numpy())
			finally:
				grad_cam.remove_hooks()

		orig = inputs.detach().cpu().numpy() * std_array + mean_array
		collected_originals.append(np.clip(orig.transpose(0, 2, 3, 1), 0.0, 1.0))
		collected_labels.extend(labels.tolist())

	originals = np.concatenate(collected_originals, axis=0)
	all_guided = np.concatenate(collected_guided, axis=0)
	cams_by_stage = [np.concatenate(stage_cams, axis=0) for stage_cams in collected_cams]

	# Render main PDF
	_save_saliency_pdf(
		output_filepath=output_filepath,
		originals=originals,
		labels=collected_labels,
		preds=collected_preds,
		dataset_name=dataset_name,
		heatmaps_by_layer=cams_by_stage,
		layer_names=stages,
		guided_grads=all_guided,
		method_name="CAM",
	)
	print(f"[Grad-CAM Complete] Visualizations saved to: {output_filepath}")

	return output_filepath


def run_gmar_pipeline(
		model: nn.Module,
		loader: DataLoader,
		dataset_name: str,
		arch: str,
		paradigm: str,
		device: torch.device,
		val_fraction: float = CONFIG["val_fraction"],
		output_name: str | None = None,
		plot: bool = True,
		resume: bool = True,
) -> str:
	"""
	Extract GMAR saliency heatmaps across all 6 blocks with correct/missed folder distribution.
	"""

	if loader is None:
		raise ValueError("A DataLoader must be provided.")

	model.eval().to(device)
	backbone = getattr(model, "backbone", model)

	if not hasattr(backbone, "encoder"):
		raise ValueError("GMAR requires a ViT backbone with an 'encoder' ModuleList")

	num_blocks = len(backbone.encoder)
	block_names = [f"Block {i + 1}" for i in range(num_blocks)]

	model_id = f"{dataset_name}_{arch}_{paradigm}"
	if output_name:
		model_id = f"{model_id}_{output_name}"
	output_stem = f"gmar_{model_id}"
	output_dir = os.path.join(DIR_OUTPUT, "gmar", model_id)
	os.makedirs(output_dir, exist_ok=True)

	# --- Individual tensor heatmaps (.pt files, no plot) ---
	if not plot:
		correct_dir = os.path.join(output_dir, "correct")
		missed_dir = os.path.join(output_dir, "missed")
		os.makedirs(correct_dir, exist_ok=True)
		os.makedirs(missed_dir, exist_ok=True)

		class_counters: dict[int, int] = {}
		total_saved = 0
		total_skipped = 0

		for images, labels in loader:
			batch_size = images.size(0)
			needed_indices = []
			sample_filenames = []

			# Check cache deterministically across both correct and missed folders
			for b in range(batch_size):
				true_label = int(labels[b])
				point_idx = class_counters.get(true_label, 0)
				filename = f"{output_stem}_c{true_label}_{point_idx}.pt"
				sample_filenames.append(filename)
				class_counters[true_label] = point_idx + 1

				cached_correct = os.path.join(correct_dir, filename)
				cached_missed = os.path.join(missed_dir, filename)

				if resume and (os.path.exists(cached_correct) or os.path.exists(cached_missed)):
					total_skipped += 1
				else:
					needed_indices.append(b)

			if not needed_indices:
				continue

			sub_inputs = images[needed_indices].to(device)
			sub_targets = labels[needed_indices].to(device)

			with torch.no_grad():
				preds = model(sub_inputs).argmax(dim=-1)

			# Extract [len(needed_indices), 32, 32, 6]
			stacked_gmar = compute_gmar_block_heatmaps(
				model=model,
				inputs=sub_inputs,
				targets=sub_targets,
				image_size=sub_inputs.shape[-2:],
			).detach().cpu()

			# Route each tensor into correct/ or missed/
			for idx, b in enumerate(needed_indices):
				is_correct = (preds[idx].item() == int(labels[b]))
				target_dir = correct_dir if is_correct else missed_dir
				torch.save(stacked_gmar[idx], os.path.join(target_dir, sample_filenames[b]))
				total_saved += 1

		print(
			f"[GMAR] Directory: {output_dir} | "
			f"Saved: {total_saved} new | Skipped: {total_skipped} existing"
		)
		return output_dir

	# --- PDF preview plot with Guided GMAR ---
	output_filepath = os.path.join(output_dir, f"{output_stem}.pdf")
	if resume and os.path.exists(output_filepath):
		print(f"[GMAR Resume] Visualizations already exist at: {output_filepath}. Skipping.")
		return output_filepath

	mean, std = get_or_compute_stats(dataset_name, val_fraction=val_fraction)
	mean_array = np.array(mean).reshape(1, 3, 1, 1)
	std_array = np.array(std).reshape(1, 3, 1, 1)

	guided_bp = GuidedBackprop(model=model)
	collected_originals, collected_labels, collected_preds = [], [], []
	collected_guided, collected_maps = [], []

	for images, labels in loader:
		inputs = images.to(device)
		targets = labels.to(device)

		# 1. Capture model predictions for correctness badges
		with torch.no_grad():
			preds = model(inputs).argmax(dim=-1)
		collected_preds.extend(preds.cpu().tolist())

		# 2. Guided Backprop & 6-block GMAR rollout
		guided_grads = guided_bp.generate_gradients(inputs, target_class=targets)
		collected_guided.append(guided_grads)

		batch_maps = compute_gmar_block_heatmaps(
			model=model,
			inputs=inputs,
			targets=targets,
			image_size=inputs.shape[-2:],
		).detach().cpu().numpy()
		collected_maps.append(batch_maps)

		# De-normalize inputs for plotting
		orig = inputs.detach().cpu().numpy() * std_array + mean_array
		collected_originals.append(np.clip(orig.transpose(0, 2, 3, 1), 0.0, 1.0))
		collected_labels.extend(labels.tolist())

	originals = np.concatenate(collected_originals, axis=0)
	all_guided = np.concatenate(collected_guided, axis=0)
	all_gmar_maps = np.concatenate(collected_maps, axis=0)  # [Total, 32, 32, 6]
	gmar_by_block = [all_gmar_maps[:, :, :, b] for b in range(num_blocks)]

	# Render main PDF
	_save_saliency_pdf(
		output_filepath=output_filepath,
		originals=originals,
		labels=collected_labels,
		preds=collected_preds,
		dataset_name=dataset_name,
		heatmaps_by_layer=gmar_by_block,
		layer_names=block_names,
		guided_grads=all_guided,
		method_name="GMAR",
	)
	print(f"[GMAR Complete] Visualizations saved to: {output_filepath}")

	return output_filepath


def compute_gmar_block_heatmaps(
		model: nn.Module,
		inputs: torch.Tensor,
		targets: torch.Tensor,
		image_size: tuple[int, int] = (32, 32),
) -> torch.Tensor:
	"""
	Computes gradient-weighted multi-head attention rollout (GMAR)
	progressively across each encoder block.
	"""

	model.eval()
	backbone = getattr(model, "backbone", model)
	if not hasattr(backbone, "encoder"):
		raise ValueError("GMAR requires a ViT backbone with an 'encoder' ModuleList.")

	num_blocks = len(backbone.encoder)
	batch_size = inputs.size(0)

	# Forward pass requesting attention storage and gradients
	model.zero_grad()
	logits = model(inputs, need_attn=True)

	# Backward pass for class-specific gradients
	loss = logits.gather(1, targets.unsqueeze(1)).sum()
	loss.backward(retain_graph=True)

	block_heatmaps = []
	rollout_matrix = None
	eye = torch.eye(backbone.grid_size ** 2 + 1, device=inputs.device).unsqueeze(0)  # [1, 65, 65]

	for block_idx in range(num_blocks):
		block = backbone.encoder[block_idx]
		attn = block.attention  # Shape: [B, num_heads, 65, 65]

		if attn is None or attn.grad is None:
			raise RuntimeError(
				f"Attention or its gradients are None at block {block_idx}. "
				"Ensure need_attn=True was passed and retain_grad() was called."
			)

		grad = attn.grad  # [B, num_heads, 65, 65]

		# Head importance: L1 norm of gradients across tokens
		head_importance = grad.abs().sum(dim=(-2, -1), keepdim=True)  # [B, num_heads, 1, 1]
		head_weights = head_importance / (head_importance.sum(dim=1, keepdim=True) + 1e-8)

		# Gradient-weighted head aggregation
		a_weighted = (head_weights * attn).sum(dim=1)  # [B, 65, 65]

		# Residual identity connection
		a_hat = a_weighted + 0.5 * eye
		a_hat = a_hat / a_hat.sum(dim=-1, keepdim=True)

		# Progressive rollout up to current block
		if rollout_matrix is None:
			rollout_matrix = a_hat
		else:
			rollout_matrix = torch.matmul(a_hat, rollout_matrix)

		# Extract [CLS] token attributions to image patches (excluding self-weight)
		cls_to_patches = rollout_matrix[:, 0, 1:]  # [B, 64]
		grid_dim = backbone.grid_size  # 8 for 32x32 image with 4x4 patches
		spatial = cls_to_patches.reshape(batch_size, 1, grid_dim, grid_dim)

		# Upsample to full image resolution
		upsampled = F.interpolate(
			spatial, size=image_size, mode="bilinear", align_corners=False
		).squeeze(1)  # [B, 32, 32]

		# Normalize per-sample to [0, 1]
		b_min = upsampled.amin(dim=(-2, -1), keepdim=True)
		b_max = upsampled.amax(dim=(-2, -1), keepdim=True)
		norm_map = (upsampled - b_min) / (b_max - b_min + 1e-8)
		block_heatmaps.append(norm_map)

	# Clean up hooks/cached gradients
	model.zero_grad()
	for block in backbone.encoder:
		block.attention = None

	# Stack along the final dimension: [B, 32, 32, 6]
	return torch.stack(block_heatmaps, dim=-1)


def evaluate_model(
		model: nn.Module,
		val_loader: DataLoader,
		device: torch.device,
		verbose: bool = True
) -> Tuple[float, float]:
	model.eval()
	model.to(device)
	criterion = nn.CrossEntropyLoss()

	running_loss = 0.0
	correct = 0
	total = 0

	loader = tqdm(val_loader, desc="[Evaluating]", leave=False) if verbose else val_loader

	with torch.no_grad():
		for images, labels in loader:
			images = images.to(device)
			labels = labels.to(device)

			outputs = model(images)
			loss = criterion(outputs, labels)

			running_loss += loss.item() * images.size(0)
			_, preds = outputs.max(1)
			correct += preds.eq(labels).sum().item()
			total += labels.size(0)

	val_loss = running_loss / total
	val_acc = 100.0 * correct / total

	if verbose:
		print()
		print(f"--- Evaluation Results ---")
		print(f"Validation Loss:     {val_loss:.4f}")
		print(f"Validation Accuracy: {val_acc:.2f}%")
		print()

	return val_loss, val_acc

'''
Linear probe evaluation for self-supervised models. Trains a linear classifier on top of the frozen backbone and evaluates on validation and test sets.
If accuracies for self-supervised and supervised models are similar, it indicates that the self-supervised model has learned useful representations.
'''

def evaluate_linear_head(backbone, head, loader, device):
	backbone.eval()
	head.eval()
	criterion = nn.CrossEntropyLoss()
	running_loss = 0.0
	correct = total = 0
	with torch.no_grad():
		for images, labels in loader:
			images = images.to(device, non_blocking=True)
			labels = labels.to(device, non_blocking=True)
			logits = head(backbone.forward_embedding(images))
			running_loss += criterion(logits, labels).item() * labels.size(0)
			correct += logits.argmax(dim=1).eq(labels).sum().item()
			total += labels.size(0)
	return running_loss / total, 100.0 * correct / total


def _rebuild_probe_convergence_state(val_losses, convergence_cutoff):
	"""Reconstruct early-stopping state from a probe's validation-loss history."""
	best_loss = float("inf")
	stalled_epochs = 0
	for loss_value in val_losses:
		loss_value = float(loss_value)
		if best_loss - loss_value > convergence_cutoff:
			best_loss = loss_value
			stalled_epochs = 0
		else:
			stalled_epochs += 1
	return best_loss, stalled_epochs


def _save_linear_probe_state(
	probe_checkpoint_path,
	metadata,
	current_epoch,
	target_probe_epochs,
	completed,
	converged,
	stop_reason,
	convergence_cutoff,
	convergence_patience,
	best_convergence_val_loss,
	epochs_without_substantial_improvement,
	head,
	best_head_state,
	optimizer,
	scheduler,
	history,
	num_classes,
	lr,
	weight_decay,
	best_epoch,
	best_val_acc,
	test_loss=None,
	test_acc=None,
):
	"""Persist the complete state required to inspect or resume a linear probe."""
	if not probe_checkpoint_path:
		return

	directory = os.path.dirname(probe_checkpoint_path)
	if directory:
		os.makedirs(directory, exist_ok=True)

	latest_train_loss = history["train_loss"][-1] if history["train_loss"] else None
	latest_train_acc = history["train_acc"][-1] if history["train_acc"] else None
	latest_val_loss = history["val_loss"][-1] if history["val_loss"] else None
	latest_val_acc = history["val_acc"][-1] if history["val_acc"] else None
	payload = {
		"probe_type": "linear",
		"probe_epoch": int(current_epoch),
		"target_probe_epochs": int(target_probe_epochs),
		"completed": bool(completed),
		"converged": bool(converged),
		"stop_reason": stop_reason,
		"convergence_cutoff": float(convergence_cutoff),
		"convergence_patience": int(convergence_patience),
		"best_convergence_val_loss": (
			None if best_convergence_val_loss == float("inf")
			else float(best_convergence_val_loss)
		),
		"epochs_without_substantial_improvement": int(
			epochs_without_substantial_improvement
		),
		"head_state_dict": {
			key: value.detach().cpu()
			for key, value in (best_head_state or head.state_dict()).items()
		},
		"last_head_state_dict": {
			key: value.detach().cpu() for key, value in head.state_dict().items()
		},
		"optimizer_state_dict": optimizer.state_dict(),
		"scheduler_state_dict": scheduler.state_dict(),
		"history": history,
		"num_classes": int(num_classes),
		"lr": float(lr),
		"weight_decay": float(weight_decay),
		"train_loss": None if latest_train_loss is None else float(latest_train_loss),
		"train_acc": None if latest_train_acc is None else float(latest_train_acc),
		"val_loss": None if latest_val_loss is None else float(latest_val_loss),
		"val_acc": None if latest_val_acc is None else float(latest_val_acc),
		"best_probe_epoch": int(best_epoch),
		"best_val_acc": float(best_val_acc),
		"test_loss": None if test_loss is None else float(test_loss),
		"test_acc": None if test_acc is None else float(test_acc),
	}
	payload.update(metadata)
	torch.save(payload, probe_checkpoint_path)


def linear_probe(
	backbone,
	train_loader,
	val_loader,
	test_loader,
	num_classes,
	device,
	epochs=CONFIG["probe_epochs"],
	lr=CONFIG["probe_lr"],
	weight_decay=0.0,
	probe_checkpoint_path=None,
	resume=False,
	metadata=None,
	convergence_cutoff=CONFIG["probe_convergence_cutoff"],
	convergence_patience=CONFIG["probe_convergence_patience"],
):
	"""Train, persist, and optionally resume a frozen-backbone linear probe.

	Probe training stops when validation loss has not improved by more than
	``convergence_cutoff`` for ``convergence_patience`` consecutive epochs.
	``epochs`` remains the hard maximum number of probe-training epochs.
	"""
	if epochs < 1:
		raise ValueError("epochs must be >= 1")
	if convergence_cutoff < 0:
		raise ValueError("convergence_cutoff must be >= 0")
	if convergence_patience < 1:
		raise ValueError("convergence_patience must be >= 1")

	metadata = dict(metadata or {})
	probe_checkpoint = None
	candidate = None
	if resume and probe_checkpoint_path and os.path.exists(probe_checkpoint_path):
		try:
			candidate = torch.load(probe_checkpoint_path, map_location="cpu")
		except Exception as exc:
			print(
				f"[Linear Probe Resume] Could not load '{probe_checkpoint_path}' ({exc}). "
				"Retraining that probe from scratch."
			)
			candidate = None

	if resume and probe_checkpoint_path and candidate is not None:
		compatible = (
			int(candidate.get("num_classes", -1)) == int(num_classes)
			and int(candidate.get("target_probe_epochs", -1)) == int(epochs)
			and float(candidate.get("lr", float("nan"))) == float(lr)
			and float(candidate.get("weight_decay", float("nan"))) == float(weight_decay)
		)
		# Probe checkpoints created before convergence stopping existed do not have
		# these fields. They remain resumable and reconstruct the stopping state
		# from their stored validation-loss history below.
		if "convergence_cutoff" in candidate:
			compatible = compatible and (
				float(candidate["convergence_cutoff"]) == float(convergence_cutoff)
			)
		if "convergence_patience" in candidate:
			compatible = compatible and (
				int(candidate["convergence_patience"]) == int(convergence_patience)
			)
		for key in ("dataset", "arch", "paradigm", "backbone_epoch"):
			if key in metadata and candidate.get(key) != metadata.get(key):
				compatible = False
				break

		if compatible:
			probe_checkpoint = candidate
			if probe_checkpoint.get("completed", False):
				print(
					f"[Linear Probe Resume] Reusing completed probe from '{probe_checkpoint_path}' "
					f"(best Val Acc {float(probe_checkpoint['best_val_acc']):.2f}%)."
				)
				return {
					"best_epoch": int(probe_checkpoint["best_probe_epoch"]),
					"best_val_acc": float(probe_checkpoint["best_val_acc"]),
					"test_loss": float(probe_checkpoint["test_loss"]),
					"test_acc": float(probe_checkpoint["test_acc"]),
					"history": probe_checkpoint.get("history", {}),
					"head_state_dict": probe_checkpoint["head_state_dict"],
					"probe_checkpoint_path": probe_checkpoint_path,
					"probe_epoch": int(probe_checkpoint.get("probe_epoch", epochs)),
					"stop_reason": probe_checkpoint.get("stop_reason", "completed"),
					"converged": bool(probe_checkpoint.get("converged", False)),
				}
		else:
			print(
				f"[Linear Probe Resume] Existing probe at '{probe_checkpoint_path}' is incompatible "
				"with the requested configuration; retraining it from scratch."
			)
			probe_checkpoint = None

	backbone = backbone.to(device)
	backbone.eval()
	original_requires_grad = [parameter.requires_grad for parameter in backbone.parameters()]
	for parameter in backbone.parameters():
		parameter.requires_grad_(False)

	head = nn.Linear(backbone.embed_dim, num_classes).to(device)
	criterion = nn.CrossEntropyLoss()
	optimizer = optim.SGD(head.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay)
	scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

	best_val_acc = float("-inf")
	best_head_state = None
	best_epoch = 0
	history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
	start_epoch = 1
	best_convergence_val_loss = float("inf")
	epochs_without_substantial_improvement = 0

	if probe_checkpoint is not None:
		last_state = probe_checkpoint.get("last_head_state_dict")
		best_state = probe_checkpoint.get("head_state_dict")
		if last_state is not None:
			head.load_state_dict(last_state)
		if probe_checkpoint.get("optimizer_state_dict") is not None:
			optimizer.load_state_dict(probe_checkpoint["optimizer_state_dict"])
		if probe_checkpoint.get("scheduler_state_dict") is not None:
			scheduler.load_state_dict(probe_checkpoint["scheduler_state_dict"])
		history = probe_checkpoint.get("history", history)
		best_val_acc = float(probe_checkpoint.get("best_val_acc", best_val_acc))
		best_epoch = int(probe_checkpoint.get("best_probe_epoch", 0))
		best_head_state = deepcopy(best_state) if best_state is not None else None
		start_epoch = int(probe_checkpoint.get("probe_epoch", 0)) + 1

		if probe_checkpoint.get("best_convergence_val_loss") is not None:
			best_convergence_val_loss = float(probe_checkpoint["best_convergence_val_loss"])
			epochs_without_substantial_improvement = int(
				probe_checkpoint.get("epochs_without_substantial_improvement", 0)
			)
		else:
			best_convergence_val_loss, epochs_without_substantial_improvement = (
				_rebuild_probe_convergence_state(
					history.get("val_loss", []),
					convergence_cutoff,
				)
			)

		print(
			f"[Linear Probe Resume] Resuming '{probe_checkpoint_path}' at probe epoch "
			f"{start_epoch}/{epochs} | convergence patience "
			f"{epochs_without_substantial_improvement}/{convergence_patience}."
		)

	try:
		stopped_epoch = min(max(start_epoch - 1, 0), epochs)
		converged = epochs_without_substantial_improvement >= convergence_patience

		for epoch in range(start_epoch, epochs + 1):
			if converged:
				break

			head.train()
			running_loss = 0.0
			correct = total = 0
			for images, labels in tqdm(train_loader, desc=f"Linear probe {epoch:03d}/{epochs:03d}", leave=False):
				images = images.to(device, non_blocking=True)
				labels = labels.to(device, non_blocking=True)
				with torch.no_grad():
					embeddings = backbone.forward_embedding(images)
				optimizer.zero_grad()
				logits = head(embeddings)
				loss = criterion(logits, labels)
				loss.backward()
				optimizer.step()
				running_loss += loss.item() * labels.size(0)
				correct += logits.argmax(dim=1).eq(labels).sum().item()
				total += labels.size(0)

			train_loss = running_loss / total
			train_acc = 100.0 * correct / total
			val_loss, val_acc = evaluate_linear_head(backbone, head, val_loader, device)
			scheduler.step()
			history["train_loss"].append(train_loss)
			history["train_acc"].append(train_acc)
			history["val_loss"].append(val_loss)
			history["val_acc"].append(val_acc)

			print(
				f"Linear probe {epoch:03d}/{epochs:03d} | "
				f"Train Loss {train_loss:.4f} | Train Acc {train_acc:.2f}% | "
				f"Val Loss {val_loss:.4f} | Val Acc {val_acc:.2f}%"
			)

			if val_acc > best_val_acc:
				best_val_acc = val_acc
				best_epoch = epoch
				best_head_state = deepcopy(head.state_dict())

			loss_improvement = best_convergence_val_loss - val_loss
			if loss_improvement > convergence_cutoff:
				best_convergence_val_loss = val_loss
				epochs_without_substantial_improvement = 0
			else:
				epochs_without_substantial_improvement += 1

			stopped_epoch = epoch
			converged = epochs_without_substantial_improvement >= convergence_patience
			_save_linear_probe_state(
				probe_checkpoint_path=probe_checkpoint_path,
				metadata=metadata,
				current_epoch=epoch,
				target_probe_epochs=epochs,
				completed=False,
				converged=converged,
				stop_reason=None,
				convergence_cutoff=convergence_cutoff,
				convergence_patience=convergence_patience,
				best_convergence_val_loss=best_convergence_val_loss,
				epochs_without_substantial_improvement=epochs_without_substantial_improvement,
				head=head,
				best_head_state=best_head_state,
				optimizer=optimizer,
				scheduler=scheduler,
				history=history,
				num_classes=num_classes,
				lr=lr,
				weight_decay=weight_decay,
				best_epoch=best_epoch,
				best_val_acc=best_val_acc,
			)

			if converged:
				print(
					f"[Linear Probe Converged] Validation loss did not improve by more than "
					f"{convergence_cutoff:g} for {convergence_patience} consecutive epochs. "
					f"Stopping at probe epoch {epoch:03d}/{epochs:03d}."
				)
				break

		if best_head_state is None:
			raise RuntimeError("Linear probe has no best head state to evaluate")

		head.load_state_dict(best_head_state)
		test_loss, test_acc = evaluate_linear_head(backbone, head, test_loader, device)
		stop_reason = "converged" if converged else "max_epochs"
		_save_linear_probe_state(
			probe_checkpoint_path=probe_checkpoint_path,
			metadata=metadata,
			current_epoch=stopped_epoch,
			target_probe_epochs=epochs,
			completed=True,
			converged=converged,
			stop_reason=stop_reason,
			convergence_cutoff=convergence_cutoff,
			convergence_patience=convergence_patience,
			best_convergence_val_loss=best_convergence_val_loss,
			epochs_without_substantial_improvement=epochs_without_substantial_improvement,
			head=head,
			best_head_state=best_head_state,
			optimizer=optimizer,
			scheduler=scheduler,
			history=history,
			num_classes=num_classes,
			lr=lr,
			weight_decay=weight_decay,
			best_epoch=best_epoch,
			best_val_acc=best_val_acc,
			test_loss=test_loss,
			test_acc=test_acc,
		)
		print(
			f"[Linear Probe Complete] Best epoch {best_epoch:03d} | "
			f"Best Val Acc {best_val_acc:.2f}% | Test Acc {test_acc:.2f}% | "
			f"Stopped at {stopped_epoch:03d}/{epochs:03d} ({stop_reason})"
		)
		return {
			"best_epoch": best_epoch,
			"best_val_acc": best_val_acc,
			"test_loss": test_loss,
			"test_acc": test_acc,
			"history": history,
			"head_state_dict": {k: v.cpu() for k, v in best_head_state.items()},
			"probe_checkpoint_path": probe_checkpoint_path,
			"probe_epoch": stopped_epoch,
			"stop_reason": stop_reason,
			"converged": converged,
		}
	finally:
		for parameter, requires_grad in zip(backbone.parameters(), original_requires_grad):
			parameter.requires_grad_(requires_grad)

