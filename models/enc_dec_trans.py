import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .trans_modules import Block
from .pos_embeds import PositionalEmbedding


class MaskedStateTransitionDecoder(nn.Module):
    """ MaskedStateTransitionDecoder for dynamic model learning and reconstruction
    """
    def __init__(self, embed_dim, action_dim, depth=8, num_heads=4,
                 mlp_ratio=4., norm_layer=nn.LayerNorm):
        super().__init__()

        # --------------------------------------------------------------------------
        # S + A -> S, S' State Decoder specifics: Casual Transformer Encoder
        self.pos_embed = PositionalEmbedding(embed_dim)  # fixed sin-cos embedding
        self.act_embed = nn.Linear(action_dim, embed_dim)
        self.blocks = nn.ModuleList([
            Block(embed_dim, num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer)
            for i in range(depth)])
        self.norm = norm_layer(embed_dim)
        # --------------------------------------------------------------------------

        self.initialize_weights()

    def initialize_weights(self):
        # initialize nn.Linear and nn.LayerNorm
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x, a):
        """
        x (state embeddings): [B, T, D]
        a (actions): [B, T, A]: A is raw action dim
        """
        # embed actions
        a = self.act_embed(a)
        B, T, D = a.size()

        # add pos embed
        x = x + self.pos_embed(T)
        a = a + self.pos_embed(T)

        # merge state and actions interleavely
        x = torch.stack([x, a], dim=2)
        x = torch.flatten(x, start_dim=1, end_dim=2)        

        # apply Transformer blocks
        for blk in self.blocks:
            x = blk(x, is_casual=True)
        x = self.norm(x)

        return x

