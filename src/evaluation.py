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


def pca_outputs(feature_map: torch.Tensor, image_index: int = 0) -> dict[str, torch.Tensor]:
	"""Run PCA once and derive the PC1 mask and RGB map from that decomposition."""
	projected = spatial_pca(feature_map, k=3, image_index=image_index)
	pc1 = projected[..., 0]
	mask = pc1 > pc1.mean()

	mins = projected.amin(dim=(0, 1), keepdim=True)
	maxs = projected.amax(dim=(0, 1), keepdim=True)
	rgb = (projected - mins) / (maxs - mins).clamp_min(1e-8)
	return {"components": projected, "mask": mask, "rgb": rgb}


def pca_mask(feature_map: torch.Tensor, image_index: int = 0) -> torch.Tensor:
	return pca_outputs(feature_map, image_index=image_index)["mask"]


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


def _collect_samples(loader: DataLoader, num_samples: int) -> tuple[torch.Tensor, torch.Tensor]:
	if num_samples < 1:
		raise ValueError("num_samples must be >= 1")

	image_parts = []
	label_parts = []
	collected = 0
	for images, labels in loader:
		remaining = num_samples - collected
		if remaining <= 0:
			break
		take = min(remaining, images.size(0))
		image_parts.append(images[:take])
		label_parts.append(labels[:take])
		collected += take

	if not image_parts:
		raise ValueError("The provided loader is empty")
	return torch.cat(image_parts, dim=0), torch.cat(label_parts, dim=0)


def run_gradcam_pipeline(
		model: nn.Module,
		val_loader: DataLoader,
		dataset_name: str,
		arch: str,
		paradigm: str,
		device: torch.device,
		num_samples: int = 8,
		val_fraction: float = CONFIG["val_fraction"],
) -> str:
	"""Extract batched Grad-CAM and Guided Grad-CAM maps for test samples."""
	model.eval().to(device)
	backbone = getattr(model, "backbone", model)
	if not hasattr(backbone, "layer4"):
		raise ValueError("Grad-CAM requires a CNN backbone with a layer4 stage")

	target_layer = cast(nn.Module, list(backbone.layer4.children())[-1])
	grad_cam = GradCAM(model=model, target_layer=target_layer)
	guided_bp = GuidedBackprop(model=model)

	mean, std = get_or_compute_stats(dataset_name, val_fraction=val_fraction)
	mean_array = np.array(mean).reshape(1, 3, 1, 1)
	std_array = np.array(std).reshape(1, 3, 1, 1)

	images_batch, labels_batch = _collect_samples(val_loader, num_samples)
	inputs = images_batch.to(device)
	targets = labels_batch.to(device)

	try:
		cam_maps = grad_cam.generate_cam(inputs, target_class=targets).numpy()
		guided_grads = guided_bp.generate_gradients(inputs, target_class=targets)
	finally:
		grad_cam.remove_hooks()

	originals = inputs.detach().cpu().numpy() * std_array + mean_array
	originals = np.clip(originals.transpose(0, 2, 3, 1), 0.0, 1.0)

	num_samples = inputs.size(0)
	fig, axes = plt.subplots(num_samples, 3, figsize=(9, 3 * num_samples))
	if num_samples == 1:
		axes = np.expand_dims(axes, 0)

	for i in range(num_samples):
		true_label = int(labels_batch[i])
		true_label_name = get_class_name(dataset_name, true_label)
		guided_cam = guided_grads[i] * cam_maps[i][..., np.newaxis]
		guided_cam -= guided_cam.mean()
		guided_cam /= guided_cam.std() + 1e-8
		guided_cam = np.clip(guided_cam * 0.15 + 0.5, 0.0, 1.0)

		axes[i, 0].imshow(originals[i])
		axes[i, 0].set_title(f"Sample {i + 1} ({true_label_name})", fontsize=10)
		axes[i, 0].axis("off")

		axes[i, 1].imshow(cam_maps[i], cmap="jet")
		axes[i, 1].set_title("Grad-CAM Heatmap", fontsize=10)
		axes[i, 1].axis("off")

		axes[i, 2].imshow(guided_cam)
		axes[i, 2].set_title("Guided Grad-CAM", fontsize=10)
		axes[i, 2].axis("off")

	plt.tight_layout()
	output_filepath = os.path.join(DIR_OUTPUT, f"gradcam_{dataset_name}_{arch}_{paradigm}.png")
	fig.savefig(output_filepath, dpi=300, bbox_inches="tight")
	plt.close(fig)

	print(f"[Grad-CAM Complete] Visualizations saved to: {output_filepath}")
	return output_filepath


def run_GMAR_pipeline(
		model: nn.Module,
		val_loader: DataLoader,
		dataset_name: str,
		arch: str,
		paradigm: str,
		device: torch.device,
		num_samples: int = 8,
		val_fraction: float = CONFIG["val_fraction"],
) -> str:
	"""Extract batched GMAR maps for test samples."""
	model.eval().to(device)
	gmar = GMAR(model=model)

	mean, std = get_or_compute_stats(dataset_name, val_fraction=val_fraction)
	mean_array = np.array(mean).reshape(1, 3, 1, 1)
	std_array = np.array(std).reshape(1, 3, 1, 1)

	images_batch, labels_batch = _collect_samples(val_loader, num_samples)
	inputs = images_batch.to(device)
	targets = labels_batch.to(device)
	gmar_maps = gmar.generate_saliency_map(
		inputs,
		target_category=targets,
		image_size=inputs.shape[-2:],
	).cpu().numpy()

	originals = inputs.detach().cpu().numpy() * std_array + mean_array
	originals = np.clip(originals.transpose(0, 2, 3, 1), 0.0, 1.0)

	num_samples = inputs.size(0)
	fig, axes = plt.subplots(num_samples, 2, figsize=(6, 3 * num_samples))
	if num_samples == 1:
		axes = np.expand_dims(axes, 0)

	for i in range(num_samples):
		true_label = int(labels_batch[i])
		true_label_name = get_class_name(dataset_name, true_label)

		axes[i, 0].imshow(originals[i])
		axes[i, 0].set_title(f"Sample {i + 1} ({true_label_name})", fontsize=10)
		axes[i, 0].axis("off")

		axes[i, 1].imshow(gmar_maps[i], cmap="jet")
		axes[i, 1].set_title("GMAR Saliency", fontsize=10)
		axes[i, 1].axis("off")

	plt.tight_layout()
	output_filepath = os.path.join(DIR_OUTPUT, f"gmar_{dataset_name}_{arch}_{paradigm}.png")
	fig.savefig(output_filepath, dpi=300, bbox_inches="tight")
	plt.close(fig)

	print(f"[GMAR Complete] Visualizations saved to: {output_filepath}")
	return output_filepath

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


def linear_probe(
    backbone,
    train_loader,
    val_loader,
    test_loader,
    num_classes,
    device,
    epochs=50,
    lr=0.1,
    weight_decay=0.0,
):
    """Train an identical frozen-backbone linear probe for std and LeJEPA."""
    from copy import deepcopy
    import torch.optim as optim

    backbone = backbone.to(device)
    backbone.eval()
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

    for epoch in range(1, epochs + 1):
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
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch
            best_head_state = deepcopy(head.state_dict())

    head.load_state_dict(best_head_state)
    test_loss, test_acc = evaluate_linear_head(backbone, head, test_loader, device)
    for parameter in backbone.parameters():
        parameter.requires_grad_(True)
    return {
        "best_epoch": best_epoch,
        "best_val_acc": best_val_acc,
        "test_loss": test_loss,
        "test_acc": test_acc,
        "history": history,
        "head_state_dict": {k: v.cpu() for k, v in best_head_state.items()},
    }
