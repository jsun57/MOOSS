import torch
import torch.nn as nn
from functools import partial
from .enc_dec_convs import ConvObservationEncoder
from .enc_dec_trans import MaskedStateTransitionDecoder

def get_encoder(obs_shape, action_shape, args):
    if args.encoder_type == 'conv_0':
        encoder = ConvObservationEncoder(
            obs_shape=(3*args.frame_stack, args.image_size[0], args.image_size[1]),
            feature_dim=args.encoder_feature_dim,
            num_layers=args.num_layers,
            num_filters=args.num_filters,
        )
    else:
        raise NotImplementedError
    return encoder

def get_decoder(obs_shape, action_shape, args):
    if args.decoder_type == 'trans_0':
        decoder = MaskedStateTransitionDecoder(
            embed_dim=args.encoder_feature_dim, action_dim=action_shape[-1], 
            depth=args.num_dec_layers, num_heads=args.num_dec_heads,
            mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6)
        )
    else:
        raise NotImplementedError
    return decoder