import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def tie_weights(src, trg):
    assert type(src) == type(trg)
    trg.weight = src.weight
    trg.bias = src.bias


def in2out_conv2d(h, w, kernel_size, padding=0, dilation=1, stride=1):
    nh = h + 2 * padding - dilation * (kernel_size - 1) - 1
    nw = w + 2 * padding - dilation * (kernel_size - 1) - 1
    return math.floor(nh / stride + 1), math.floor(nw / stride + 1)


class ConvObservationEncoder(nn.Module):
    """Basic Convolutional encoder of pixels observations."""
    def __init__(
        self,
        obs_shape,
        feature_dim,
        num_layers=4,
        num_filters=64,
    ):
        super().__init__()

        assert len(obs_shape) == 3
        self.obs_shape = obs_shape
        self.feature_dim = feature_dim
        self.num_layers = num_layers

        self.convs = nn.ModuleList([nn.Conv2d(obs_shape[0], num_filters, 3, stride=2)])
        out_h, out_w = in2out_conv2d(obs_shape[-2], obs_shape[-1], 3, stride=2)
        
        for i in range(num_layers - 1):
            self.convs.append(nn.Conv2d(num_filters, num_filters, 3, stride=1))
            out_h, out_w = in2out_conv2d(out_h, out_w, 3, stride=1)
        
        self.fc = nn.Linear(num_filters * out_h * out_w, self.feature_dim)
        self.ln = nn.LayerNorm(self.feature_dim)


    def forward_conv(self, obs, flatten=True):
        conv = torch.relu(self.convs[0](obs))
        for i in range(1, self.num_layers):
            conv = torch.relu(self.convs[i](conv))
        h = conv.view(conv.size(0), -1) if flatten else conv        
        return h

    def forward(self, obs, mask=None, detach=False):
        """
        input: [B, C, H, W]
        output: [B, Z]
        """
        h = self.forward_conv(obs)
        if detach:
            h = h.detach()
        h_fc = self.fc(h)
        h_norm = self.ln(h_fc)
        out = h_norm
        return out

    def copy_conv_weights_from(self, source):
        """Tie convolutional layers"""
        # only tie conv layers
        for i in range(self.num_layers):
            tie_weights(src=source.convs[i], trg=self.convs[i])
