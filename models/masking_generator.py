import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import einops
from einops import rearrange, repeat
import torch_cluster
from torch_geometric.utils import sort_edge_index, degree


class STGraphMaskGenerator:
    def __init__(self, clip_shape, patch_shape, mask_ratio):
        """
        clip_shape: (H, W, T)
        patch_shape: (H_p, W_p, T_p)
        """
        assert mask_ratio <= 1.0
        self.mask_ratio = mask_ratio

        height, width, num_frames = clip_shape
        self.hp_size, self.wp_size, self.tp_size = patch_shape
        self.num_h_patches, self.num_w_patches, self.num_t_patches = height // self.hp_size, width // self.wp_size, num_frames // self.tp_size
        self.num_nodes = self.num_h_patches * self.num_w_patches * self.num_t_patches

        self.st_graph_edge_index = self._generate_cube_st_graph(self.num_t_patches, self.num_h_patches, self.num_w_patches)
        self.st_graph_edge_index = sort_edge_index(self.st_graph_edge_index, num_nodes=self.num_nodes)


    def __call__(self, per_walk_length=1500, p=1.0, q=1.0, sh_factor=1, sw_factor=1):
        """
        return mask: [T, 1, num_h_patches, num_w_patches]; 0 indicates masked areas
        """
        node_to_mask = self._generate_all_mask_nodes(per_walk_length=per_walk_length, p=p, q=q)
        st_graph_mask = self._generate_mask(node_to_mask, self.num_t_patches, self.num_h_patches, self.num_w_patches,
            sh_factor=sh_factor, sw_factor=sw_factor, t_factor=self.tp_size)
        return st_graph_mask


    def _generate_cube_st_graph(self, T, H, W):
        """
        generate edge list of cube-like spatial-temporal graph
        input: T, H, W
        output: edge_index [2, num_edges]
        """
        offsets = torch.tensor([(dt, dh, dw) for dt in range(-1, 2) for dh in range(-1, 2) for dw in range(-1, 2)])          # [27, 3]

        # Generate index lists
        T_idxs = rearrange(torch.arange(T), '... -> ... () () ()') # [T, 1, 1, 1]
        H_idxs = rearrange(torch.arange(H), '... -> () ... () ()') # [1, H, 1, 1]
        W_idxs = rearrange(torch.arange(W), '... -> () () ... ()') # [1, 1, W, 1]

        # Compute ht, wt, and tt
        tt = T_idxs + offsets[:, 0]
        ht = H_idxs + offsets[:, 1]
        wt = W_idxs + offsets[:, 2]

        # Filter out invalid indices
        valid_indices = ((tt >= 0) & (tt < T) & (ht >= 0) & (ht < H) & (wt >= 0) & (wt < W))    # [T, H, W, 27] / [T, H, W, 7]
        
        C = valid_indices.shape[-1]
        from_nodes = repeat((T_idxs * H * W + H_idxs * W + W_idxs), '... 1 -> ... c', c=C)
        to_nodes = tt * H * W + ht * W + wt

        neighbor_idxs = torch.nonzero(valid_indices)
        from_nodes = from_nodes[valid_indices]
        to_nodes = to_nodes[valid_indices]

        return torch.stack((from_nodes, to_nodes))


    def _generate_rw_nodes(self, start_idx, walk_length=10, p=1.0, q=1.0):
        """
        perform 1 random walk of length <walk_length> on graph
        input: edge_index: [2, num_edges]
        return: node index list
        """
        row, col = self.st_graph_edge_index
        start = torch.empty(1).fill_(start_idx).long()
        deg = degree(row, num_nodes=self.num_nodes)
        rowptr = row.new_zeros(self.num_nodes + 1)
        torch.cumsum(deg, 0, out=rowptr[1:])
        n_id, _ = torch.ops.torch_cluster.random_walk(rowptr, col, start, walk_length, p, q)
        return n_id


    def _generate_all_mask_nodes(self, per_walk_length=1500, p=1.0, q=1.0):
        """
        generate all nodes to be masked on cube graph
        input: edge_index: [2, num_edges], num_nodes, mask_ratio, walk_length
        return: node index tensor
        """
        num_nodes_to_mask = int(self.num_nodes * self.mask_ratio)
        selected_nodes = []
        self.counter = 0

        start_idx = int(self.num_nodes // 2)

        if num_nodes_to_mask > 0:
            while len(selected_nodes) < num_nodes_to_mask:
                rw_sequence = self._generate_rw_nodes(start_idx, walk_length=per_walk_length, p=p, q=q)[0]
                selected_nodes += rw_sequence.tolist()
                selected_nodes = list(dict.fromkeys(selected_nodes))
                start_idx = rw_sequence[-1]
                self.counter += 1  
            selected_nodes = torch.tensor(selected_nodes)[:num_nodes_to_mask]
        else:
            selected_nodes = None
        return selected_nodes


    def _generate_mask(self, node_to_mask, T, H, W, sh_factor=1, sw_factor=1, t_factor=4):
        """
        generate full cube mask
        input: node_to_mask: [num_mask_nodes], THW, ratios
        return: complete cube mask: [T * expand_t_ratio, 1, H * expand_h_ratio, W * expand_w_ratio]
        """
        num_nodes = T * H * W
        mask = torch.ones(num_nodes)
        if node_to_mask is not None:
            mask[node_to_mask] = 0
        mask = rearrange(mask, '(t h w) -> 1 1 t h w', t=T, h=H, w=W)
        out_size = (T * t_factor, H * sh_factor, W * sw_factor)
        mask = F.interpolate(mask, out_size)
        mask = rearrange(mask, '1 1 t h w -> t 1 h w')
        return mask
