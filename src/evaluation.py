from copy import deepcopy
from typing import Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

def spatial_pca(feature_map: torch.Tensor, k: int = 3, image_index: int = 0) -> torch.Tensor:
	if feature_map.ndim != 4:
		raise ValueError(f"Expected [B, C, H, W], got {tuple(feature_map.shape)}")
	if not 0 <= image_index < feature_map.size(0):
		raise IndexError(f"image_index {image_index} is outside batch size {feature_map.size(0)}")
	if k < 1:
		raise ValueError("k must be at least 1")

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
	"""Compatibility helper for callers that only need a PC1 mask."""
	return pca_outputs(feature_map, image_index=image_index)["mask"]


def pca_rgb(feature_map: torch.Tensor, image_index: int = 0) -> torch.Tensor:
	"""Compatibility helper for callers that only need a three-component image."""
	return pca_outputs(feature_map, image_index=image_index)["rgb"]



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
		print(f"Validation Loss: {val_loss:.4f}")
		print(f"Validation Accuracy: {val_acc:.2f}%")
	return val_loss, val_acc

'''
Linear probe evaluation for self-supervised models. Trains a linear classifier on top of the frozen backbone and evaluates on validation and test sets.
If accuracies for self-supervised and supervised models are similar, it indicates that the self-supervised model has learned useful representations.
'''

def linear_probe(
		backbone: nn.Module,
		train_loader: DataLoader,
		val_loader: DataLoader,
		test_loader: DataLoader,
		num_classes: int,
		device: torch.device,
		epochs: int = 50,
		lr: float = 0.1,
		weight_decay: float = 0.0,
) -> dict:
	backbone = backbone.to(device)
	backbone.eval()
	for parameter in backbone.parameters():
		parameter.requires_grad_(False)

	head = nn.Linear(backbone.embed_dim, num_classes).to(device)
	criterion = nn.CrossEntropyLoss()
	optimizer = optim.SGD(head.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay)
	scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

	history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
	best_val_acc = float("-inf")
	best_head_state = None
	best_epoch = 0

	for epoch in range(1, epochs + 1):
		backbone.eval()
		head.train()
		running_loss = 0.0
		correct = total = 0

		for images, labels in tqdm(
				train_loader,
				desc=f"Linear probe {epoch:03d}/{epochs:03d}",
				leave=False,
		):
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
			f"Probe epoch {epoch:03d}/{epochs:03d} | "
			f"Train Acc: {train_acc:.2f}% | Val Acc: {val_acc:.2f}%"
		)

		if val_acc > best_val_acc:
			best_val_acc = val_acc
			best_epoch = epoch
			best_head_state = deepcopy(head.state_dict())

	head.load_state_dict(best_head_state)
	test_loss, test_acc = evaluate_linear_head(backbone, head, test_loader, device)

	return {
		"best_epoch": best_epoch,
		"best_val_acc": best_val_acc,
		"test_loss": test_loss,
		"test_acc": test_acc,
		"history": history,
		"head_state_dict": {key: value.cpu() for key, value in best_head_state.items()},
	}

def evaluate_linear_head(
		backbone: nn.Module,
		head: nn.Module,
		loader: DataLoader,
		device: torch.device,
) -> Tuple[float, float]:
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
