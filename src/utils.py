import os
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.data import get_dataloaders
from src.globals import DEVICE, CONFIG, DIR_CHECKPOINTS, DATASETS, set_seed
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


def get_checkpoint_path(dataset: str, arch: str, paradigm: str, tag: str = "best") -> str:
	filename = f"{dataset}_{arch}_{paradigm}_{tag}.pt"
	return os.path.join(DIR_CHECKPOINTS, filename)


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
	print("Configuration diagnostic verified successfully.")


def test_pipeline(dataset: str, arch: str, paradigm: str):
	print(f"Testing DataLoaders for {dataset}...")
	train_loader, val_loader = get_dataloaders(dataset_name=dataset, batch_size=8)
	images, labels = next(iter(train_loader))
	print(f"Batch shape: {images.shape}, Labels shape: {labels.shape}")

	print(f"Instantiating model ({arch}, {paradigm}) for {dataset}...")
	model = build_model(arch=arch, dataset=dataset, paradigm=paradigm).to(DEVICE)
	images = images.to(DEVICE)

	if hasattr(model, "forward_features"):
		s1, s2, s3, s4 = model.forward_features(images)
		print(f"Stage 1 feature shape: {s1.shape}")
		print(f"Stage 2 feature shape: {s2.shape}")
		print(f"Stage 3 feature shape: {s3.shape}")
		print(f"Stage 4 feature shape: {s4.shape}")

	logits = model(images)
	print(f"Logits output shape:   {logits.shape}")
	print("Pipeline diagnostic completed successfully.")
