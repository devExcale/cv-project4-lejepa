import json
import os
from typing import Tuple, cast

import numpy as np
import torch
import torch.nn as nn
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.data import get_or_compute_stats
from src.globals import DATASETS, DIR_DATA, DIR_OUTPUT
from src.utils import GradCAM, GuidedBackprop


def get_dataset_class_names(dataset_name: str) -> list[str]:
	dataset_name = dataset_name.lower()
	if dataset_name not in DATASETS:
		return []

	spec = DATASETS[dataset_name]
	label_key = spec["label_key"]

	try:
		from datasets import load_dataset_builder

		builder = load_dataset_builder(spec["hf_path"], cache_dir=DIR_DATA)
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


def run_gradcam_pipeline(
		model: nn.Module,
		val_loader: DataLoader,
		dataset_name: str,
		arch: str,
		paradigm: str,
		device: torch.device,
		num_samples: int = 8
) -> str:
	"""
	Extract Grad-CAM and Guided Grad-CAM maps for validation samples.
	"""
	model.eval()
	model.to(device)

	target_layer = cast(nn.Module, list(getattr(model, "layer4").children())[-1])
	grad_cam = GradCAM(model=model, target_layer=target_layer)
	guided_bp = GuidedBackprop(model=model)

	mean, std = get_or_compute_stats(dataset_name)
	mean = np.array(mean).reshape(3, 1, 1)
	std = np.array(std).reshape(3, 1, 1)

	images_batch, labels_batch = next(iter(val_loader))
	num_samples = min(num_samples, len(images_batch))

	fig, axes = plt.subplots(num_samples, 3, figsize=(9, 3 * num_samples))
	if num_samples == 1:
		axes = np.expand_dims(axes, 0)

	for i in range(num_samples):
		img_tensor = images_batch[i:i + 1].to(device)
		true_label = labels_batch[i].item()
		true_label_name = get_class_name(dataset_name, true_label)

		# 1. Compute standard Grad-CAM
		cam_map = grad_cam.generate_cam(img_tensor, target_class=true_label).numpy()

		# 2. Compute Guided Backprop gradients
		guided_grads = guided_bp.generate_gradients(img_tensor, target_class=true_label)

		# 3. Denormalize input image
		orig_img = img_tensor.squeeze(0).cpu().numpy()
		orig_img = orig_img * std + mean
		orig_img = np.clip(orig_img.transpose(1, 2, 0), 0.0, 1.0)

		# 4. Fuse into Guided Grad-CAM
		guided_cam = guided_grads * cam_map[..., np.newaxis]
		guided_cam -= guided_cam.mean()
		guided_cam /= (guided_cam.std() + 1e-8)
		guided_cam = guided_cam * 0.15 + 0.5
		guided_cam = np.clip(guided_cam, 0.0, 1.0)

		# Plot 1: Input Image
		axes[i, 0].imshow(orig_img)
		axes[i, 0].set_title(f"Sample {i + 1} ({true_label_name})", fontsize=10)
		axes[i, 0].axis("off")

		# Plot 2: Grad-CAM Heatmap
		axes[i, 1].imshow(cam_map, cmap="jet")
		axes[i, 1].set_title("Grad-CAM Heatmap", fontsize=10)
		axes[i, 1].axis("off")

		# Plot 3: Guided Grad-CAM
		axes[i, 2].imshow(guided_cam)
		axes[i, 2].set_title("Guided Grad-CAM", fontsize=10)
		axes[i, 2].axis("off")

	plt.tight_layout()
	output_filepath = os.path.join(DIR_OUTPUT, f"gradcam_{dataset_name}_{arch}_{paradigm}.png")
	plt.savefig(output_filepath, dpi=300, bbox_inches="tight")
	plt.close()

	grad_cam.remove_hooks()
	print(f"[Grad-CAM Complete] Visualizations saved to: {output_filepath}")
	return output_filepath
