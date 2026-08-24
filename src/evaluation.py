from typing import Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm


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
