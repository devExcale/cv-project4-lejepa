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


class VisionTransformer(nn.Module):
    """CIFAR-scale ViT using the newer encoder/feed-forward design."""

    def __init__(
        self,
        num_classes: int = 100,
        img_size: int = 32,
        patch_size: int = 4,
        in_chans: int = 3,
        embed_dim: int = 512,
        depth: int = 6,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        dropout: float = 0.15,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.img_size = img_size
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.grid_size = img_size // patch_size

        self.patch_embed = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        num_patches = self.grid_size ** 2
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.pos_drop = nn.Dropout(p=dropout)
        self.encoder = nn.ModuleList([
            AttentionEncoder(embed_dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, dropout=dropout)
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)

    def _position_embedding(self, height: int, width: int) -> torch.Tensor:
        if height == self.grid_size and width == self.grid_size:
            return self.pos_embed
        cls_position = self.pos_embed[:, :1]
        patch_positions = self.pos_embed[:, 1:].reshape(
            1, self.grid_size, self.grid_size, self.embed_dim
        ).permute(0, 3, 1, 2)
        patch_positions = F.interpolate(
            patch_positions, size=(height, width), mode="bicubic", align_corners=False
        )
        patch_positions = patch_positions.flatten(2).transpose(1, 2)
        return torch.cat((cls_position, patch_positions), dim=1)

    def _prepare_tokens(self, x: torch.Tensor):
        b = x.size(0)
        x = self.patch_embed(x)
        _, c, hp, wp = x.shape
        tokens = x.flatten(2).transpose(1, 2)
        cls = self.cls_token.expand(b, -1, -1)
        tokens = torch.cat((cls, tokens), dim=1)
        tokens = tokens + self._position_embedding(hp, wp)
        tokens = self.pos_drop(tokens)
        return tokens, c, hp, wp

    def forward_features(self, x: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        """Return one spatial feature map after every transformer block."""
        b = x.size(0)
        tokens, c, hp, wp = self._prepare_tokens(x)
        features = []
        for block in self.encoder:
            tokens = block(tokens)
            spatial = tokens[:, 1:, :].transpose(1, 2).reshape(b, c, hp, wp)
            features.append(spatial)
        return tuple(features)

    def forward_embedding(self, x: torch.Tensor) -> torch.Tensor:
        tokens, _, _, _ = self._prepare_tokens(x)
        for block in self.encoder:
            tokens = block(tokens)
        return self.norm(tokens[:, 0, :])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.forward_embedding(x))


    def forward_features(self, x: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        """
        Return a tuple of spatial feature maps. For the ViT we return a single
        feature map obtained by reshaping token embeddings back to a grid:
        [B, embed_dim, H_patch, W_patch]
        """
        B = x.size(0)
        x = self.patch_embed(x)                       # [B, C, Hp, Wp]
        _, C, Hp, Wp = x.shape
        tokens = x.flatten(2).transpose(1, 2)        # [B, N, C]

        cls = self.cls_token.expand(B, -1, -1)      # [B, 1, C]
        tokens = torch.cat((cls, tokens), dim=1)     # [B, N+1, C]
        tokens = tokens + self.pos_embed
        tokens = self.pos_drop(tokens)

        for blk in self.encoder:
            tokens = blk(tokens)

        # remove cls token and reshape tokens to spatial map
        tok = tokens[:, 1:, :].transpose(1, 2).reshape(B, C, Hp, Wp)
        return (tok,)

    def forward_embedding(self, x: torch.Tensor) -> torch.Tensor:
        """Return pooled embedding (cls token)"""
        B = x.size(0)
        x = self.patch_embed(x)
        _, C, Hp, Wp = x.shape
        tokens = x.flatten(2).transpose(1, 2)
        cls = self.cls_token.expand(B, -1, -1)
        tokens = torch.cat((cls, tokens), dim=1)
        tokens = tokens + self.pos_embed
        tokens = self.pos_drop(tokens)

        for blk in self.encoder:
            tokens = blk(tokens)

        cls_tok = tokens[:, 0, :]
        cls_tok = self.norm(cls_tok)
        return cls_tok

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        emb = self.forward_embedding(x)
        return self.head(emb)


class AttentionEncoder(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int = 8, mlp_ratio: float = 4.0, dropout: float = 0.0):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio
        self.dropout = dropout

        # Use batch_first=True so inputs are [B, N, E]
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.mlp = FeedForward(
            dim=embed_dim,
            hidden_dim=int(embed_dim * mlp_ratio),
            output_dim=embed_dim,
            dropout=dropout
        )
        self.norm1 = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Pre-norm: Norm → Attention → Residual
        x_norm = self.norm1(x)
        attn_output, _ = self.attn(x_norm, x_norm, x_norm)
        x = x + attn_output

        normed_mlp_output = self.mlp(x)
        x = x + normed_mlp_output
        return x


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, output_dim, dropout = 0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


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
		backbone = VisionTransformer(num_classes=num_classes, img_size=32, patch_size=4, embed_dim=512, depth=6, num_heads=8)
	else:
		raise ValueError(f"Unknown architecture '{arch}'")

	if paradigm == "std":
		return backbone
	if paradigm == "lejepa":
		return LeJEPA(backbone, num_slices=num_slices, t_max=t_max, n_points=n_points, lamb=lamb)

	raise ValueError(f"Unknown training paradigm '{paradigm}'")
