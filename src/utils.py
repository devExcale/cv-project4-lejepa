import glob
import os
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.data import get_dataloaders
from src.globals import DEVICE, CONFIG, DATASETS, set_seed, DIR_CHECKPOINTS
from src.network import build_model


class GuidedBackprop:
	"""Guided Backpropagation hook manager for capturing fine pixel gradients."""

	def __init__(self, model: nn.Module):
		self.model = model
		self.hooks = []
		# In-place ReLUs break backward hooks; disable them across the entire model
		for module in self.model.modules():
			if isinstance(module, nn.ReLU):
				module.inplace = False

	def _register_hooks(self):
		def relu_backward_hook(module, grad_in, grad_out):
			if isinstance(grad_in[0], torch.Tensor):
				return (torch.clamp(grad_in[0], min=0.0),)

		for module in self.model.modules():
			if isinstance(module, nn.ReLU):
				self.hooks.append(module.register_full_backward_hook(relu_backward_hook))

	def generate_gradients(self, input_tensor: torch.Tensor, target_class: int) -> np.ndarray:
		self.model.eval()
		self._register_hooks()  # Attach hooks only for Guided Backprop pass

		input_tensor = input_tensor.clone().detach().requires_grad_(True)
		output = self.model(input_tensor)
		self.model.zero_grad()

		score = output[0, target_class]
		score.backward()

		gradients = input_tensor.grad.detach().cpu().squeeze(0).permute(1, 2, 0).numpy()

		self.remove_hooks()  # Detach hooks immediately so other passes run normally
		return gradients

	def remove_hooks(self):
		for hook in self.hooks:
			hook.remove()
		self.hooks = []


class GradCAM:
	"""
	Grad-CAM hook manager and saliency map generator for CNN backbones.
	"""

	def __init__(self, model: nn.Module, target_layer: nn.Module):
		self.model = model
		self.target_layer: nn.Module = target_layer
		self.activations: Optional[torch.Tensor] = None
		self.gradients: Optional[torch.Tensor] = None

		# Register forward and backward hooks
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
			target_class: Optional[int] = None
	) -> torch.Tensor:
		"""
		Generate normalized Grad-CAM heatmaps for a single input tensor of shape [1, C, H, W].
		Returns a 2D float tensor of shape [H, W] normalized in [0, 1].
		"""
		self.model.eval()
		self.model.zero_grad()

		# Forward pass
		logits = self.model(input_tensor)  # [1, Num_Classes]

		if target_class is None:
			target_class = logits.argmax(dim=1).item()

		# Target score (pre-softmax)
		score = logits[0, target_class]
		score.backward(retain_graph=True)

		# Global average pooling of gradients: alpha_k = (1/Z) * sum(grad)
		# self.gradients: [1, C, H', W'], self.activations: [1, C, H', W']
		weights = torch.mean(self.gradients, dim=(2, 3), keepdim=True)  # [1, C, 1, 1]

		# Weighted combination of feature activation maps
		cam = torch.sum(weights * self.activations, dim=1, keepdim=True)  # [1, 1, H', W']
		cam = F.relu(cam)  # Apply ReLU to focus on features that have a positive influence

		# Upsample to original input resolution (32x32)
		_, _, h, w = input_tensor.shape
		cam = F.interpolate(cam, size=(h, w), mode="bilinear", align_corners=False)
		cam = cam.squeeze().cpu()

		# Normalize between 0 and 1
		cam_min, cam_max = cam.min(), cam.max()
		if cam_max > cam_min:
			cam = (cam - cam_min) / (cam_max - cam_min)
		else:
			cam = torch.zeros_like(cam)

		return cam

	def remove_hooks(self):
		"""Clean up PyTorch hooks."""
		self.forward_handle.remove()
		self.backward_handle.remove()


class MilestoneCheckpointer:
	"""
	Manages:
	1. The overall best model checkpoint (for resuming and optimal evaluation).
	2. Accuracy milestone checkpoints grouped by milestone folders:
		e.g., checkpoints/30/{dataset}_{arch}_{paradigm}_29.5.pt
		Keeps strictly the closest model to each milestone centroid.
	"""

	DEFAULT_MILESTONES: List[int] = [15, 30, 45, 60, 75, 90]

	def __init__(
			self,
			dataset: str,
			arch: str,
			paradigm: str,
			milestones: Optional[List[int]] = None,
			base_dir: str = DIR_CHECKPOINTS,
	):
		self.dataset = dataset
		self.arch = arch
		self.paradigm = paradigm
		self.base_dir = base_dir
		self.milestones = milestones or self.DEFAULT_MILESTONES
		self.best_acc: float = 0.0

		# Map milestone -> {"acc": float, "diff": float, "path": str}
		self.milestone_records: Dict[int, Dict[str, Any]] = {}
		self._sync_existing_milestone_files()

	@property
	def base_name(self) -> str:
		"""
		Constructs a base filename for checkpoints based on dataset, architecture, and paradigm.
		:return: A string in the format "{dataset}_{arch}_{paradigm}".
		"""
		return f"{self.dataset}_{self.arch}_{self.paradigm}"

	@property
	def best_path(self) -> str:
		"""
		Constructs the path to the best checkpoint file.
		:return: A string representing the path to the best checkpoint.
		"""
		return os.path.join(self.base_dir, f"{self.base_name}_best.pt")

	def get_milestone_dir(self, milestone: int) -> str:
		"""
		Constructs the directory path for a specific milestone.
		:param milestone: The milestone value (e.g., 30 for 30% accuracy).
		:return: A string representing the path to the milestone directory.
		"""
		return os.path.join(self.base_dir, str(milestone))

	def get_milestone_path(self, milestone: int, acc: float) -> str:
		"""
		Constructs the full path for a checkpoint corresponding to a specific milestone and accuracy.
		:param milestone: The milestone value (e.g., 30 for 30% accuracy).
		:param acc: The accuracy of the checkpoint.
		:return: A string representing the path to the checkpoint.
		"""
		return os.path.join(self.get_milestone_dir(milestone), f"{self.base_name}_{acc:.1f}.pt")

	def _sync_existing_milestone_files(self) -> None:
		"""
		Inspects disk for existing checkpoints to preserve closest distance records across restarts.
		"""

		# Scan each milestone
		for m in self.milestones:

			# Get milestone directory
			m_dir = self.get_milestone_dir(m)
			if not os.path.isdir(m_dir):
				continue

			# Look for checkpoints in milestone directory
			pattern = os.path.join(m_dir, f"{self.base_name}_*.pt")
			for filepath in glob.glob(pattern):
				try:

					# Extract accuracy from filename (e.g. cifar10_cnn_std_29.5.pt -> 29.5)
					acc_str = os.path.splitext(filepath)[0].rsplit("_", 1)[-1]
					acc = float(acc_str)
					diff = abs(acc - m)

					# Register checkpoint closest to milestone
					if m not in self.milestone_records or diff < self.milestone_records[m]["diff"]:
						self.milestone_records[m] = {"acc": acc, "diff": diff, "path": filepath}

				except ValueError:
					continue
		return

	def load_best(
			self,
			model: nn.Module,
			optimizer: torch.optim.Optimizer,
			scheduler: Optional[Any] = None,
			default_history: Optional[Dict[str, list]] = None,
			device: torch.device = torch.device("cpu"),
	) -> Tuple[int, float, Dict[str, list]]:
		"""
		Restores model, optimizer, scheduler, and history from the best checkpoint.
		:param model: PyTorch model to load state into
		:param optimizer: PyTorch optimizer to load state into
		:param scheduler: Optional learning rate scheduler to load state into
		:param default_history: Optional default history dictionary to use if no checkpoint is found
		:param device: Device to map the checkpoint to (default: CPU)
		:return: Tuple containing (start_epoch, best_val_acc, history)
		"""

		# Check if the best checkpoint exists
		if not os.path.exists(self.best_path):
			print(f"[Resume] No checkpoint found at `{self.best_path}`. Training from scratch.")
			return 1, 0.0, default_history or {}

		# Load the best checkpoint
		print(f"[Resume] Found checkpoint at `{self.best_path}`. Loading states...")
		ckpt = torch.load(self.best_path, map_location=device)

		model.load_state_dict(ckpt["model_state_dict"])
		optimizer.load_state_dict(ckpt["optimizer_state_dict"])

		saved_epoch = ckpt.get("epoch", 0)
		self.best_acc = ckpt.get("val_acc", 0.0)
		history = ckpt.get("history", default_history or {})
		start_epoch = saved_epoch + 1

		# Restore scheduler state if available, otherwise step it to the saved epoch
		if scheduler is not None:
			if ckpt.get("scheduler_state_dict"):
				scheduler.load_state_dict(ckpt["scheduler_state_dict"])
			else:
				for _ in range(saved_epoch):
					scheduler.step()

		print(
			f"[Resume] Successfully loaded checkpoint. "
			f"Resuming from Epoch {start_epoch} (Best Val Acc so far: {self.best_acc:.2f}%)"
		)

		return start_epoch, self.best_acc, history

	def step(
			self,
			epoch: int,
			val_acc: float,
			model: nn.Module,
			optimizer: torch.optim.Optimizer,
			history: Dict[str, list],
			scheduler: Optional[Any] = None,
	) -> None:
		"""
		Evaluates val_acc against both the overall best and the closest milestone centroid.
		:param epoch: Current epoch
		:param val_acc: Current validation accuracy
		:param model: PyTorch model
		:param optimizer: PyTorch optimizer
		:param history: Training history
		:param scheduler: Optional learning rate scheduler
		"""

		checkpoint_data = {
			"epoch": epoch,
			"model_state_dict": model.state_dict(),
			"optimizer_state_dict": optimizer.state_dict(),
			"scheduler_state_dict": scheduler.state_dict() if scheduler else None,
			"val_acc": val_acc,
			"history": history,
			"dataset": self.dataset,
			"arch": self.arch,
			"paradigm": self.paradigm,
		}

		# Update overall best checkpoint
		if val_acc > self.best_acc:
			self.best_acc = val_acc
			torch.save(checkpoint_data, self.best_path)
			print(f"--> Saved new best checkpoint to `{self.best_path}` (Val Acc: {self.best_acc:.2f}%)")

		# Update closest milestone checkpoint
		target_milestone = min(self.milestones, key=lambda c: abs(c - val_acc))
		current_diff = abs(val_acc - target_milestone)

		is_new_closest = (
				target_milestone not in self.milestone_records
				or current_diff < self.milestone_records[target_milestone]["diff"]
		)

		if is_new_closest:
			# Remove previous checkpoint for this milestone if it had a different accuracy name
			old_record = self.milestone_records.get(target_milestone)
			if old_record and os.path.exists(old_record["path"]):
				try:
					os.remove(old_record["path"])
				except OSError:
					pass

			target_dir = self.get_milestone_dir(target_milestone)
			os.makedirs(target_dir, exist_ok=True)

			new_path = self.get_milestone_path(target_milestone, val_acc)
			torch.save(checkpoint_data, new_path)

			self.milestone_records[target_milestone] = {
				"acc": val_acc,
				"diff": current_diff,
				"path": new_path,
			}
			print(
				f"--> [Milestone {int(target_milestone)}%] Saved closest checkpoint to `{new_path}` "
				f"(Val Acc: {val_acc:.2f}%, Diff: {current_diff:.2f}%)"
			)

		return


def get_checkpoint_path(dataset: str, arch: str, paradigm: str, tag: str = "best") -> str:
	return os.path.join(DIR_CHECKPOINTS, f"{dataset}_{arch}_{paradigm}_{tag}.pt")


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
	print(f"Checkpoint Target:   {get_checkpoint_path(dataset, arch, paradigm)}")


def test_pipeline(dataset: str, arch: str, paradigm: str):
	train_loader, _ = get_dataloaders(dataset_name=dataset, batch_size=8, paradigm=paradigm)
	batch, labels = next(iter(train_loader))
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

	features = model.forward_features(images)
	for i, feature in enumerate(features, 1):
		print(f"Layer {i} feature shape: {tuple(feature.shape)}")
