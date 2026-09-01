from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.resnet import ResNet, BasicBlock

from src.globals import DATASETS, CONFIG


class VisionTransformer(nn.Module):
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
            dropout: float = 0.1,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.img_size = img_size
        self.patch_size = patch_size
        self.embed_dim = embed_dim

        # Patch embedding using a conv with stride=patch_size
        self.patch_embed = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        num_patches = (img_size // patch_size) ** 2

        # Classification token and positional embedding
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.pos_drop = nn.Dropout(p=dropout)

        # Encoder blocks
        self.encoder = nn.ModuleList([
            AttentionEncoder(embed_dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, dropout=dropout)
            for _ in range(depth)
        ])

        self.norm = nn.LayerNorm(embed_dim)

        # Classification head
        self.head = nn.Sequential(
            nn.Linear(embed_dim, num_classes),
            nn.GELU(),
            nn.Linear(num_classes, num_classes),
        )

        # expose embed_dim for compatibility with other code
        self.embed_dim = embed_dim

        # initialize
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)

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
    def __init__(self, embed_dim: int, num_heads: int = 8, mlp_ratio: float = 4.0, dropout: float = 0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio
        self.dropout = dropout

        # Use batch_first=True so inputs are [B, N, E]
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, int(embed_dim * mlp_ratio)),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(int(embed_dim * mlp_ratio), embed_dim),
            nn.Dropout(dropout)
        )
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Pre-norm: Norm → Attention → Residual
        x_norm = self.norm1(x)
        attn_output, _ = self.attn(x_norm, x_norm, x_norm)
        x = x + attn_output

        # Pre-norm: Norm → MLP → Residual
        x_norm = self.norm2(x)
        mlp_output = self.mlp(x_norm)
        x = x + mlp_output

        return x

'''
patch + position embedding:
 patch flattening --> linear projection --> position embedding --> cls token
encoder:
 norm --> multihead attention --> residual connection --> norm --> MLP --> residual connection --> norm  XL
classification head:
 MLP --> softmax
'''