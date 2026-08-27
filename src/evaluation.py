from typing import Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm


def spatial_pca(feature_map: torch.Tensor, k: int = 3, image_index: int = 0) -> torch.Tensor:
	"""PCA over one spatial layer map. H and W are read directly from the actual layer tensor."""
	if feature_map.ndim != 4:
		raise ValueError(f"Expected [B, C, H, W], got {tuple(feature_map.shape)}")

	x = feature_map[image_index].detach()  # [C, H, W]
	c, h, w = x.shape
	x = x.permute(1, 2, 0).reshape(h * w, c).float()
	x = x - x.mean(dim=0, keepdim=True)
	_, _, v = torch.pca_lowrank(x, q=min(k, x.shape[0], x.shape[1]), center=False)
	projected = x @ v[:, :k]
	return projected.reshape(h, w, k).cpu()


def pca_mask(feature_map: torch.Tensor, image_index: int = 0) -> torch.Tensor:
	x = spatial_pca(feature_map, k=1, image_index=image_index)[..., 0]
	return x > x.mean()


def pca_rgb(feature_map: torch.Tensor, image_index: int = 0) -> torch.Tensor:
	x = spatial_pca(feature_map, k=3, image_index=image_index)
	return (x - x.amin()) / (x.amax() - x.amin() + 1e-8)


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


def evaluate_lejepa(model: nn.Module, val_loader: DataLoader, device: torch.device) -> Tuple[float, float]:
	model.eval().to(device)
	means, stds = [], []
	with torch.no_grad():
		for images, _ in val_loader:
			z = model(images=images.to(device))
			means.append(z.mean().item())
			stds.append(z.std().item())
	print(f"Embedding mean: {sum(means) / len(means):.4f}")
	print(f"Embedding std:  {sum(stds) / len(stds):.4f}")
	return sum(means) / len(means), sum(stds) / len(stds)
