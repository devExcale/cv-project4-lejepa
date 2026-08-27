from typing import Tuple

import torch
import torch.nn as nn
from torchvision.models.resnet import ResNet, BasicBlock, resnet18

from src.globals import DATASETS


class CIFARResNet18(ResNet):

	def __init__(self, num_classes: int = 100):
		super(CIFARResNet18, self).__init__(
			block=BasicBlock,
			layers=[2, 2, 2, 2],
			num_classes=num_classes
		)
		self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
		self.bn1 = nn.BatchNorm2d(64)
		self.maxpool = nn.Identity()

	def forward_features(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
		x = self.relu(self.bn1(self.conv1(x)))
		x = self.maxpool(x)

		stage1 = self.layer1(x)  # [B, 64, 32, 32]
		stage2 = self.layer2(stage1)  # [B, 128, 16, 16]
		stage3 = self.layer3(stage2)  # [B, 256, 8, 8]
		stage4 = self.layer4(stage3)  # [B, 512, 4, 4]

		return stage1, stage2, stage3, stage4

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		*_, stage4 = self.forward_features(x)
		out = self.avgpool(stage4)
		out = torch.flatten(out, 1)
		out = self.fc(out)
		return out


def build_cifar_resnet18(num_classes: int) -> nn.Module:
	"""
	Build a CIFAR-10/100 ResNet-18 model.
	:param num_classes: Number of output classes
	:return: Instantiated PyTorch model
	"""

	# Instantiate a standard ResNet-18 model
	model = resnet18(num_classes=num_classes)

	# Edit the first convolutional layer for 32x32 inputs (CIFAR)
	model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
	model.maxpool = nn.Identity()

	return model


def build_model(arch: str, dataset: str, paradigm: str) -> nn.Module:
	"""
	Build a model based on the specified architecture, dataset, and training paradigm.
	:param arch: Model architecture
	:param dataset: Dataset name
	:param paradigm: Training paradigm
	:return: Instantiated PyTorch model
	"""

	if dataset not in DATASETS:
		raise ValueError(f"Unknown dataset '{dataset}'. Registered datasets: {list(DATASETS.keys())}")

	num_classes = DATASETS[dataset]["num_classes"]
	input_size = DATASETS[dataset].get("input_size")

	if arch == "cnn" and paradigm == "std":
		if input_size == 32:
			return build_cifar_resnet18(num_classes)
		if input_size == 224:
			return resnet18(weights=None, num_classes=num_classes)
		raise ValueError(f"Unsupported input size '{input_size}' for architecture '{arch}'.")

	if arch == "vit":
		raise NotImplementedError("ViT")

	if paradigm == "lejepa":
		raise NotImplementedError("LeJEPA")

	raise ValueError(f"Unknown combination of architecture '{arch}' and paradigm '{paradigm}'.")
