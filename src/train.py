import math
import os
import time

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.evaluation import evaluate_model
from src.globals import CONFIG, DIR_CHECKPOINTS



def get_lr_for_epoch(epoch: int, epochs: int, base_lr: float, warmup_epochs: int = 5) -> float:
	"""Simple warmup + cosine decay schedule for ViT."""
	if warmup_epochs > 0 and epoch <= warmup_epochs:
		return base_lr * (epoch / max(1, warmup_epochs))
	progress = (epoch - warmup_epochs) / max(1, epochs - warmup_epochs)
	progress = min(max(progress, 0.0), 1.0)
	return base_lr * 0.5 * (1.0 + math.cos(math.pi * progress))


def _periodic_checkpoint_path(dataset_name, arch, paradigm, epoch):
	return os.path.join(DIR_CHECKPOINTS, f"{dataset_name}_{arch}_{paradigm}_epoch_{epoch:04d}.pt")


def _best_checkpoint_path(dataset_name, arch, paradigm):
	return os.path.join(DIR_CHECKPOINTS, f"{dataset_name}_{arch}_{paradigm}_best.pt")


def _save_periodic_checkpoint(epoch, model, optimizer, history, dataset_name, arch, paradigm, scheduler=None, extra=None):
	path = _periodic_checkpoint_path(dataset_name, arch, paradigm, epoch)
	payload = {
		"epoch": epoch,
		"model_state_dict": model.state_dict(),
		"optimizer_state_dict": optimizer.state_dict(),
		"scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
		"history": history,
		"dataset": dataset_name,
		"arch": arch,
		"paradigm": paradigm,
	}
	if extra:
		payload.update(extra)
	torch.save(payload, path)
	print(f"--> Saved periodic checkpoint: {path}")
	return path


def train_supervised(
	model,
	train_loader,
	val_loader,
	dataset_name,
	arch,
	paradigm="std",
	epochs=CONFIG["epochs"],
	lr=CONFIG["lr"],
	weight_decay=CONFIG["weight_decay"],
	momentum=CONFIG["momentum"],
	device=CONFIG["device"],
	resume=False,
	checkpoint_interval=CONFIG["checkpoint_interval"],
	):
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
	if checkpoint_interval < 1:
		raise ValueError("checkpoint_interval must be >= 1")

	model = model.to(device)
	is_vit = arch == "vit"

	if is_vit:
		optimizer = optim.AdamW(
			model.parameters(),
			lr=lr,
			weight_decay=min(weight_decay * 5.0, 0.1),
		)
		scheduler = None
		label_smoothing = 0.1
		warmup_epochs = max(5, min(10, epochs // 10))
	else:
		optimizer = optim.SGD(
			model.parameters(),
			lr=lr,
			momentum=momentum,
			weight_decay=weight_decay,
		)
		scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
		label_smoothing = 0.0
		warmup_epochs = 0

	criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
	history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
	best_val_acc = float("-inf")
	best_path = _best_checkpoint_path(dataset_name, arch, paradigm)
	start_epoch = 1

	# The recovery checkpoint is intentionally independent of the periodic
	# checkpoints used later for linear-probe milestone selection.
	if resume:
		if os.path.exists(best_path):
			checkpoint = torch.load(best_path, map_location=device)
			model.load_state_dict(checkpoint["model_state_dict"])
			optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
			if scheduler is not None and checkpoint.get("scheduler_state_dict") is not None:
				scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

			history = checkpoint.get("history", history)
			best_val_acc = float(checkpoint.get("val_acc", float("-inf")))
			start_epoch = int(checkpoint.get("epoch", 0)) + 1
			print(
				f"[Resume] Loaded best supervised checkpoint from '{best_path}'. "
				f"Resuming at epoch {start_epoch} with best Val Acc {best_val_acc:.2f}%."
			)
		else:
			print(f"[Resume] No best checkpoint found at '{best_path}'. Training from scratch.")

	if start_epoch > epochs:
		print(f"[Resume] Best checkpoint already reached epoch {start_epoch - 1}; target is {epochs} epochs.")
		return history

	start_time = time.time()

	for epoch in range(start_epoch, epochs + 1):
		if is_vit:
			lr_for_epoch = get_lr_for_epoch(epoch, epochs, lr, warmup_epochs)
			for group in optimizer.param_groups:
				group["lr"] = lr_for_epoch

		model.train()
		running_loss = 0.0
		correct = total = 0
		pbar = tqdm(train_loader, desc=f"Epoch {epoch:03d}/{epochs:03d} [Train]", leave=False)

		for images, labels in pbar:
			images = images.to(device, non_blocking=True)
			labels = labels.to(device, non_blocking=True)

			optimizer.zero_grad()
			outputs = model(images)
			loss = criterion(outputs, labels)
			loss.backward()
			if is_vit:
				nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
			optimizer.step()

			running_loss += loss.item() * images.size(0)
			correct += outputs.argmax(dim=1).eq(labels).sum().item()
			total += labels.size(0)

		if scheduler is not None:
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
			f"Train Loss {train_loss:.4f} | Train Acc {train_acc:.2f}% | "
			f"Val Loss {val_loss:.4f} | Val Acc {val_acc:.2f}%"
		)

		# Recovery checkpoint: best validation accuracy seen during supervised training.
		if val_acc > best_val_acc:
			best_val_acc = val_acc
			torch.save({
				"epoch": epoch,
				"model_state_dict": model.state_dict(),
				"optimizer_state_dict": optimizer.state_dict(),
				"scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
				"val_acc": best_val_acc,
				"val_loss": val_loss,
				"history": history,
				"dataset": dataset_name,
				"arch": arch,
				"paradigm": paradigm,
			}, best_path)
			print(f"--> Saved new best recovery checkpoint: {best_path} (Val Acc {best_val_acc:.2f}%)")

		# Periodic checkpoints are the candidates used by the later linear-probe analysis.
		if epoch % checkpoint_interval == 0 or epoch == epochs:
			_save_periodic_checkpoint(
				epoch,
				model,
				optimizer,
				history,
				dataset_name,
				arch,
				paradigm,
				scheduler=scheduler,
				extra={"val_acc": val_acc, "val_loss": val_loss},
			)

	print(
		f"[Training Complete] Elapsed {(time.time() - start_time) / 60:.2f} mins | "
		f"Best Val Acc {best_val_acc:.2f}%"
	)
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
	resume=False,
	checkpoint_interval=CONFIG["checkpoint_interval"],
	):
	if checkpoint_interval < 1:
		raise ValueError("checkpoint_interval must be >= 1")

	model = model.to(device)
	optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
	history = {"loss": [], "invariance": [], "sigreg": []}
	best_loss = float("inf")
	best_path = _best_checkpoint_path(dataset_name, arch, "lejepa")
	start_epoch = 1

	if resume:
		if os.path.exists(best_path):
			checkpoint = torch.load(best_path, map_location=device)
			model.load_state_dict(checkpoint["model_state_dict"])
			optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

			history = checkpoint.get("history", history)
			best_loss = float(checkpoint.get("loss", float("inf")))
			start_epoch = int(checkpoint.get("epoch", 0)) + 1
			print(
				f"[Resume] Loaded best LeJEPA checkpoint from '{best_path}'. "
				f"Resuming at epoch {start_epoch} with best loss {best_loss:.4f}."
			)
		else:
			print(f"[Resume] No best checkpoint found at '{best_path}'. Training from scratch.")

	if start_epoch > epochs:
		print(f"[Resume] Best checkpoint already reached epoch {start_epoch - 1}; target is {epochs} epochs.")
		return history

	start_time = time.time()

	for epoch in range(start_epoch, epochs + 1):
		model.train()
		total_loss = total_inv = total_sig = 0.0
		pbar = tqdm(train_loader, desc=f"Epoch {epoch:03d}/{epochs:03d} [LeJEPA]", leave=False)

		for views, _ in pbar:
			global_views = [x.to(device, non_blocking=True) for x in views["global"]]
			local_views = [x.to(device, non_blocking=True) for x in views["local"]]

			optimizer.zero_grad()
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

		print(
			f"Epoch {epoch:03d}/{epochs:03d} | "
			f"Loss {values[0]:.4f} | Inv {values[1]:.4f} | SIGReg {values[2]:.4f}"
		)

		# Recovery checkpoint: lowest LeJEPA training loss seen so far.
		if values[0] < best_loss:
			best_loss = values[0]
			torch.save({
				"epoch": epoch,
				"model_state_dict": model.state_dict(),
				"optimizer_state_dict": optimizer.state_dict(),
				"loss": best_loss,
				"invariance": values[1],
				"sigreg": values[2],
				"history": history,
				"dataset": dataset_name,
				"arch": arch,
				"paradigm": "lejepa",
			}, best_path)
			print(f"--> Saved new best recovery checkpoint: {best_path} (Loss {best_loss:.4f})")

		if epoch % checkpoint_interval == 0 or epoch == epochs:
			_save_periodic_checkpoint(
				epoch,
				model,
				optimizer,
				history,
				dataset_name,
				arch,
				"lejepa",
				extra={"loss": values[0], "invariance": values[1], "sigreg": values[2]},
			)

	print(
		f"[Training Complete] Elapsed {(time.time() - start_time) / 60:.2f} mins | "
		f"Best LeJEPA Loss {best_loss:.4f}"
	)
	return history
