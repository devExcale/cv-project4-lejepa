import math
import os
import time

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.evaluation import evaluate_model
from src.globals import CONFIG
from src.utils import get_checkpoint_path


def get_lr_for_epoch(epoch: int, epochs: int, base_lr: float, warmup_epochs: int = 5) -> float:
	"""Simple warmup + cosine decay schedule for ViT."""
	if warmup_epochs > 0 and epoch <= warmup_epochs:
		return base_lr * (epoch / max(1, warmup_epochs))
	progress = (epoch - warmup_epochs) / max(1, epochs - warmup_epochs)
	progress = min(max(progress, 0.0), 1.0)
	return base_lr * 0.5 * (1.0 + math.cos(math.pi * progress))


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
		device: torch.device = CONFIG["device"],
		resume: bool = False,
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
	:param resume: If True, load the best checkpoint and resume training
	:return: Dictionary containing training history (loss and accuracy)
	"""

	# Train model on the specified device
	model = model.to(device)

	# ViT-specific training settings
	if arch == "vit":
		label_smoothing = 0.1  # Reduce overconfidence
		vit_weight_decay = min(weight_decay * 5.0, 0.1)  # Stronger regularization
		warmup_epochs = max(5, min(10, epochs // 10))  # Adaptive warmup
	else:
		label_smoothing = 0.0
		vit_weight_decay = weight_decay
		warmup_epochs = 0

	# Optimizer setup
	if arch == "vit":
		optimizer = optim.AdamW(
			model.parameters(),
			lr=lr,
			weight_decay=vit_weight_decay
		)
	else:
		optimizer = optim.SGD(
			model.parameters(),
			lr=lr,
			momentum=momentum,
			weight_decay=weight_decay
		)

	criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
	scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

	best_acc = 0.0
	start_epoch = 1
	best_ckpt_path = get_checkpoint_path(dataset=dataset_name, arch=arch, paradigm=paradigm, tag="best")
	history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}

	if resume:
		if os.path.exists(best_ckpt_path):

			print(f"Resume] Found checkpoint at `{best_ckpt_path}`. Loading states...")
			checkpoint = torch.load(best_ckpt_path, map_location=device)

			# Restore model and optimizer weights
			model.load_state_dict(checkpoint["model_state_dict"])
			optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

			# Restore training tracking variables
			saved_epoch = checkpoint.get("epoch", 0)
			start_epoch = saved_epoch + 1
			best_acc = checkpoint.get("val_acc", 0.0)
			history = checkpoint.get("history", history)

			# Align learning rate scheduler for non-ViT architectures
			if arch != "vit":
				if "scheduler_state_dict" in checkpoint and checkpoint["scheduler_state_dict"] is not None:
					scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
				else:
					# Fast-forward scheduler steps to match the resumed epoch
					for _ in range(saved_epoch):
						scheduler.step()

			print(
				f"Successfully loaded checkpoint. "
				f"Resuming from Epoch {start_epoch}/{epochs} (Best Val Acc so far: {best_acc:.2f}%)"
			)

			if start_epoch > epochs:
				print(f"[Resume] Model has already completed all {epochs} epochs.")
				return history
		else:
			print(f"No checkpoint found at `{best_ckpt_path}`, training from scratch.")

	print(
		f"\n[Starting Training] Dataset: {dataset_name.upper()}"
		f" | Arch: {arch.upper()}"
		f" | Epochs: {start_epoch} -> {epochs}"
		f" | Device: {device}"
	)
	start_time = time.time()

	for epoch in range(start_epoch, epochs + 1):

		# Update learning rate for ViT
		if arch == "vit":
			lr_for_epoch = get_lr_for_epoch(epoch, epochs, lr, warmup_epochs=warmup_epochs)
			for group in optimizer.param_groups:
				group["lr"] = lr_for_epoch

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
			if arch == "vit":
				nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
			optimizer.step()

			running_loss += loss.item() * images.size(0)
			_, preds = outputs.max(1)
			correct += preds.eq(labels).sum().item()
			total += labels.size(0)

			pbar.set_postfix({"loss": f"{loss.item():.4f}", "acc": f"{100.0 * correct / total:.2f}%"})

		if arch != "vit":
			scheduler.step()

		train_loss = running_loss / total
		train_acc = 100.0 * correct / total

		val_loss, val_acc = evaluate_model(model, val_loader, device=device, verbose=False)

		history["train_loss"].append(train_loss)
		history["train_acc"].append(train_acc)
		history["val_loss"].append(val_loss)
		history["val_acc"].append(val_acc)

		if arch == "vit":
			current_lr = optimizer.param_groups[0]["lr"]
		else:
			current_lr = scheduler.get_last_lr()[0]

		print(
			f"Epoch {epoch:03d}/{epochs:03d} | "
			f"Train Loss: {train_loss:.4f} - Train Acc: {train_acc:.2f}% | "
			f"Val Loss: {val_loss:.4f} - Val Acc: {val_acc:.2f}% | "
			f"LR: {current_lr:.5f}"
		)

		# Save checkpoint if validation accuracy improves
		if val_acc > best_acc:
			best_acc = val_acc
			torch.save({
				"epoch": epoch,
				"model_state_dict": model.state_dict(),
				"optimizer_state_dict": optimizer.state_dict(),
				"scheduler_state_dict": scheduler.state_dict() if arch != "vit" else None,
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


def train_lejepa(
		model,
		train_loader,
		dataset_name,
		arch,
		epochs=CONFIG["epochs"],
		lr=1e-3,
		weight_decay=CONFIG["weight_decay"],
		device=CONFIG["device"],
		resume: bool = False,
) -> dict:

	model = model.to(device)
	optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

	best_loss = float("inf")
	start_epoch = 1
	best_ckpt_path = get_checkpoint_path(dataset_name, arch, "lejepa", "best")
	history = {"loss": [], "invariance": [], "sigreg": []}

	# ----------------------------------------------------
	# Resume checkpoint logic
	# ----------------------------------------------------
	if resume:
		if os.path.exists(best_ckpt_path):
			print(f"[Resume] Found checkpoint at `{best_ckpt_path}`. Loading states...")
			checkpoint = torch.load(best_ckpt_path, map_location=device)

			# Restore weights and optimizer state
			model.load_state_dict(checkpoint["model_state_dict"])
			optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

			# Restore training tracking variables
			saved_epoch = checkpoint.get("epoch", 0)
			start_epoch = saved_epoch + 1
			best_loss = checkpoint.get("loss", float("inf"))
			history = checkpoint.get("history", history)

			print(
				f"[Resume] Successfully loaded checkpoint. "
				f"Resuming from Epoch {start_epoch}/{epochs} (Best Loss so far: {best_loss:.4f})"
			)

			if start_epoch > epochs:
				print(f"[Resume] Model has already completed all {epochs} epochs. Exiting training.")
				return history
		else:
			print(f"[Resume] Warning: No checkpoint found at `{best_ckpt_path}`. Training from scratch.")

	for epoch in range(start_epoch, epochs + 1):
		model.train()
		total_loss = total_inv = total_sig = 0.0

		pbar = tqdm(train_loader, desc=f"Epoch {epoch:03d}/{epochs:03d} [LeJEPA]", leave=False)

		for views, _ in pbar:
			global_views = [x.to(device, non_blocking=True) for x in views["global"]]
			local_views = [x.to(device, non_blocking=True) for x in views["local"]]

			optimizer.zero_grad()
			# Call LeJEPA.forward(...)
			loss, inv, sig = model(global_views=global_views, local_views=local_views)

			loss.backward()
			optimizer.step()

			total_loss += loss.item()
			total_inv += inv.item()
			total_sig += sig.item()

		n = len(train_loader)
		values = total_loss / n, total_inv / n, total_sig / n

		for key, value in zip(history, values):
			history[key].append(value)

		print(f"Epoch {epoch:03d}/{epochs:03d} | Loss {values[0]:.4f} | Inv {values[1]:.4f} | SIGReg {values[2]:.4f}")

		# Save checkpoint if loss improves
		if values[0] < best_loss:
			best_loss = values[0]
			torch.save({
				"epoch": epoch,
				"model_state_dict": model.state_dict(),
				"optimizer_state_dict": optimizer.state_dict(),
				"loss": best_loss,
				"history": history,
				"dataset": dataset_name,
				"arch": arch,
				"paradigm": "lejepa"
			}, best_ckpt_path)
			print(f"--> Saved new best checkpoint to `{best_ckpt_path}` (Loss: {best_loss:.4f})")

	return history
