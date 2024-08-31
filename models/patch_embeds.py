import logging
from typing import Callable, List, Optional, Tuple, Union

import torch
from torch import nn as nn
import torch.nn.functional as F

try:
    from torch import _assert
except ImportError:
    def _assert(condition: bool, message: str):
        assert condition, message

class PatchEmbed(nn.Module):
    """ 2D Image to Patch Embedding
    """
    def __init__(
            self,
            img_size: Optional[Tuple] = (100, 100),
            patch_size: Tuple = (10, 10),
            in_chans: int = 3,
            embed_dim: int = 768,
            norm_layer: Optional[Callable] = None,
            bias: bool = True,
    ):
        super().__init__()
        self.patch_size = patch_size
        if img_size is not None:
            self.img_size = img_size
            self.grid_size = tuple([s // p for s, p in zip(self.img_size, self.patch_size)])
            self.num_patches = self.grid_size[0] * self.grid_size[1]
            self.num_patches_h = self.grid_size[0]
            self.num_patches_w = self.grid_size[1]
        else:
            self.img_size = None
            self.grid_size = None
            self.num_patches = None

        # flatten spatial dim and transpose to channels last, kept for bwd compat
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size, bias=bias)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        B, C, H, W = x.shape
        _assert(H == self.img_size[0], f"Input height ({H}) doesn't match model ({self.img_size[0]}).")
        _assert(W == self.img_size[1], f"Input width ({W}) doesn't match model ({self.img_size[1]}).")
        _assert(
            H % self.patch_size[0] == 0,
            f"Input height ({H}) should be divisible by patch size ({self.patch_size[0]})."
        )
        _assert(
            W % self.patch_size[1] == 0,
            f"Input width ({W}) should be divisible by patch size ({self.patch_size[1]})."
        )
        x = self.proj(x)
        x = x.flatten(2).transpose(1, 2)  # NCHW -> NLC
        x = self.norm(x)
        return x

