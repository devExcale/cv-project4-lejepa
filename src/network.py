from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.resnet import ResNet, BasicBlock

from src.globals import DATASETS, CONFIG


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
		self.embed_dim = 512

	def forward_features(self, x: torch.Tensor) -> Tuple[torch.Tensor, ...]:
		x = self.relu(self.bn1(self.conv1(x)))
		x = self.maxpool(x)

		stage1 = self.layer1(x)          # [B, 64, H, W]
		stage2 = self.layer2(stage1)     # [B, 128, H/2, W/2]
		stage3 = self.layer3(stage2)     # [B, 256, H/4, W/4]
		stage4 = self.layer4(stage3)     # [B, 512, H/8, W/8]

		return stage1, stage2, stage3, stage4

	def forward_embedding(self, x: torch.Tensor) -> torch.Tensor:
		*_, stage4 = self.forward_features(x)
		return torch.flatten(self.avgpool(stage4), 1)

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		return self.fc(self.forward_embedding(x))


class EppsPulley(nn.Module):
	def __init__(
			self,
			t_max: float = 3.0,
			n_points: int = 17
	):
		super().__init__()
		if n_points < 3 or n_points % 2 == 0:
			raise ValueError("n_points must be an odd integer >= 3")

		t = torch.linspace(0, t_max, n_points)
		dt = t_max / (n_points - 1)
		phi = torch.exp(-0.5 * t.square())
		weights = torch.full((n_points,), 2 * dt)
		weights[[0, -1]] = dt

		self.register_buffer("t", t)
		self.register_buffer("phi", phi)
		self.register_buffer("weights", weights * phi)

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		x_t = x.unsqueeze(-1) * self.t
		cos_mean = x_t.cos().mean(0)
		sin_mean = x_t.sin().mean(0)
		error = (cos_mean - self.phi).square() + sin_mean.square()
		return (error @ self.weights) * x.size(0)


class SIGReg(nn.Module):
	def __init__(
			self,
			num_slices: int = 1024,
			t_max: float = 3.0,
			n_points: int = 17
	):
		super().__init__()
		self.num_slices = num_slices
		self.ep = EppsPulley(t_max=t_max, n_points=n_points)
		self.register_buffer("global_step", torch.zeros((), dtype=torch.long))

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		with torch.no_grad():
			g = torch.Generator(device=x.device).manual_seed(self.global_step.item())
			directions = torch.randn(
				x.size(-1), self.num_slices, device=x.device, dtype=x.dtype, generator=g
			)
			directions = directions / directions.norm(p=2, dim=0).clamp_min(1e-12)
			self.global_step.add_(1)

		projected = x @ directions
		return self.ep(projected).mean()


class LeJEPA(nn.Module):
	def __init__(
			self,
			backbone: nn.Module,
			num_slices: int = 1024,
			t_max: float = 3.0,
			n_points: int = 17,
			lamb: float = 0.02,
	):
		super().__init__()
		self.backbone = backbone
		self.embed_dim = backbone.embed_dim
		self.projector = nn.Sequential(
			nn.Linear(self.embed_dim, 512),
			nn.BatchNorm1d(512),
			nn.ReLU(),
			nn.Linear(512, 2048),
			nn.BatchNorm1d(2048),
			nn.ReLU(),
			nn.Linear(2048, 2048),
			nn.BatchNorm1d(2048),
			nn.ReLU(),
			nn.Linear(2048, 512),
		)
		self.sigreg = SIGReg(num_slices=num_slices, t_max=t_max, n_points=n_points)
		self.lamb = lamb

	def forward_features(self, x: torch.Tensor) -> Tuple[torch.Tensor, ...]:
		return self.backbone.forward_features(x)

	def forward_embedding(self, x: torch.Tensor) -> torch.Tensor:
		return self.backbone.forward_embedding(x)

	def forward(
			self,
			global_views=None,
			local_views=None,
			images: torch.Tensor = None,
	):
		if not self.training:
			if images is None:
				raise ValueError("images must be provided in eval mode")
			return self.forward_embedding(images)

		if global_views is None or local_views is None:
			raise ValueError("global_views and local_views are required for LeJEPA training")

		all_views = global_views + local_views
		features = torch.cat([self.forward_embedding(view) for view in all_views], dim=0)
		projected = self.projector(features)

		batch_size = global_views[0].size(0)
		projected = projected.view(len(all_views), batch_size, -1)
		center = projected[:len(global_views)].mean(0)
		
		inv_loss = (projected - center.unsqueeze(0)).square().mean()
		sigreg_loss = self.sigreg(projected.reshape(-1, projected.size(-1)))
		loss = inv_loss + self.lamb * sigreg_loss
		return loss, inv_loss, sigreg_loss


def build_model(
		arch: str,
		dataset: str,
		paradigm: str,
		num_slices: int = CONFIG.get("sigreg_slices", 1024),
		t_max: float = CONFIG.get("sigreg_tmax", 3.0),
		n_points: int = CONFIG.get("sigreg_points", 17),
		lamb: float = CONFIG.get("lejepa_lambda", 0.02),
) -> nn.Module:
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
	
	if arch == "cnn":
		backbone = CIFARResNet18(num_classes=num_classes)
	elif arch == "vit":
		pass
		#backbone = ViT(num_classes=num_classes)
	else:
		raise ValueError(f"Unknown architecture '{arch}'")

	if paradigm == "std":
		return backbone
	if paradigm == "lejepa":
		return LeJEPA(backbone, num_slices=num_slices, t_max=t_max, n_points=n_points, lamb=lamb)

	raise ValueError(f"Unknown training paradigm '{paradigm}'")
