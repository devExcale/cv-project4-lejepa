import time

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.evaluation import evaluate_model
from src.globals import CONFIG
from src.utils import get_checkpoint_path


def train_supervised(
		model: nn.Module,
		train_loader: DataLoader,
		val_loader: DataLoader,
		dataset_name: str,
		arch: str,
		paradigm: str,
		epochs: int = CONFIG["epochs"],
		lr: float = CONFIG["lr"],
		weight_decay: float = CONFIG["weight_decay"],
		momentum: float = CONFIG["momentum"],
		device: torch.device = CONFIG["device"]
) -> dict:
	"""
	Execute standard supervised training loop with best-accuracy checkpointing.
	:param model: PyTorch model to train
	:param train_loader: DataLoader for training data
	:param val_loader: DataLoader for validation data
	:param dataset_name: Name of the dataset (e.g., 'cifar10', 'cifar100')
	:param arch: Model architecture (e.g., 'cnn', 'vit')
	:param paradigm: Training paradigm (e.g., 'std', 'lejepa')
	:param epochs: Number of training epochs
	:param lr: Learning rate
	:param weight_decay: Weight decay for optimizer
	:param momentum: Momentum for optimizer
	:param device: Device to run training on (e.g., 'cuda', 'cpu')
	:return: Dictionary containing training history (loss and accuracy)
	"""

	# Train model on the specified device
	model = model.to(device)

	# Use SGD with Cross Entropy Loss
	criterion = nn.CrossEntropyLoss()
	optimizer = optim.SGD(
		model.parameters(),
		lr=lr,
		momentum=momentum,
		weight_decay=weight_decay
	)

	scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

	best_acc = 0.0
	best_ckpt_path = get_checkpoint_path(dataset=dataset_name, arch=arch, paradigm=paradigm, tag="best")
	history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}

	print(
		f"\n[Starting Training] Dataset: {dataset_name.upper()}"
		f" | Arch: {arch.upper()}"
		f" | Epochs: {epochs}"
		f" | Device: {device}"
	)
	start_time = time.time()

	for epoch in range(1, epochs + 1):
		model.train()
		running_loss = 0.0
		correct = 0
		total = 0

		pbar = tqdm(train_loader, desc=f"Epoch {epoch:03d}/{epochs:03d} [Train]", leave=False)
		for images, labels in pbar:
			images = images.to(device)
			labels = labels.to(device)

			optimizer.zero_grad()
			outputs = model(images)
			loss = criterion(outputs, labels)
			loss.backward()
			optimizer.step()

			running_loss += loss.item() * images.size(0)
			_, preds = outputs.max(1)
			correct += preds.eq(labels).sum().item()
			total += labels.size(0)

			pbar.set_postfix({"loss": f"{loss.item():.4f}", "acc": f"{100.0 * correct / total:.2f}%"})

		scheduler.step()

		train_loss = running_loss / total
		train_acc = 100.0 * correct / total

		val_loss, val_acc = evaluate_model(model, val_loader, device=device, verbose=False)

		history["train_loss"].append(train_loss)
		history["train_acc"].append(train_acc)
		history["val_loss"].append(val_loss)
		history["val_acc"].append(val_acc)

		print(
			f"Epoch {epoch:03d}/{epochs:03d} | "
			f"Train Loss: {train_loss:.4f} - Train Acc: {train_acc:.2f}% | "
			f"Val Loss: {val_loss:.4f} - Val Acc: {val_acc:.2f}% | "
			f"LR: {scheduler.get_last_lr()[0]:.5f}"
		)

		# Save checkpoint if validation accuracy improves
		if val_acc > best_acc:
			best_acc = val_acc
			torch.save({
				"epoch": epoch,
				"model_state_dict": model.state_dict(),
				"optimizer_state_dict": optimizer.state_dict(),
				"val_acc": best_acc,
				"history": history,
				"dataset": dataset_name,
				"arch": arch,
				"paradigm": paradigm
			}, best_ckpt_path)
			print(f"--> Saved new best checkpoint to `{best_ckpt_path}` (Val Acc: {best_acc:.2f}%)")

	total_time = time.time() - start_time
	print(f"\n[Training Complete] Elapsed Time: {total_time / 60:.2f} mins | Best Val Acc: {best_acc:.2f}%\n")

	return history
