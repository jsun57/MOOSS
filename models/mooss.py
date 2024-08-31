import copy
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
from einops import rearrange, repeat

from kornia.augmentation import (CenterCrop, RandomCrop)

import utils
from utils import PositionalEmbedding, InverseSquareRootSchedule, AnneallingSchedule, search_last_step
from .load_models import get_encoder, get_decoder
from .masking_generator import STGraphMaskGenerator
from .temp_conts_loss import temp_contrastive_efficient


def exists(x):
    return x is not None


def gaussian_logprob(noise, log_std):
    residual = (-0.5 * noise.pow(2) - log_std).sum(-1, keepdim=True)
    return residual - 0.5 * np.log(2 * np.pi) * noise.size(-1)


def squash(mu, pi, log_pi):
    mu = torch.tanh(mu)
    if pi is not None:
        pi = torch.tanh(pi)
    if log_pi is not None:
        log_pi -= torch.log(F.relu(1 - pi.pow(2)) + 1e-6).sum(-1, keepdim=True)
    return mu, pi, log_pi


def weight_init(m):
    if isinstance(m, nn.Linear):
        nn.init.orthogonal_(m.weight.data)
        m.bias.data.fill_(0.0)


def weight_init_conv(m):
    """Custom weight init for Conv2D and Linear layers."""
    if isinstance(m, nn.Linear):
        nn.init.orthogonal_(m.weight.data)
        m.bias.data.fill_(0.0)
    elif isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        # delta-orthogonal init from https://arxiv.org/pdf/1806.05393.pdf
        assert m.weight.size(2) == m.weight.size(3)
        m.weight.data.fill_(0.0)
        m.bias.data.fill_(0.0)
        mid = m.weight.size(2) // 2
        gain = nn.init.calculate_gain('relu')
        nn.init.orthogonal_(m.weight.data[:, :, mid, mid], gain)


class Actor(nn.Module):
    """MLP actor network."""
    def __init__(
        self, 
        obs_shape, 
        action_shape, 
        hidden_dim, 
        encoder_feature_dim,
        num_layers,
        patch_size,
        in_channels,
        log_std_min, 
        log_std_max, 
        configs,
    ):
        super().__init__()

        self.encoder = get_encoder(obs_shape, action_shape, configs)
        
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max

        self.trunk = nn.Sequential(
            nn.Linear(encoder_feature_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, 2 * action_shape[0]))

        self.outputs = dict()
        self.apply(weight_init_conv)

    def forward(self, obs, compute_pi=True, compute_log_pi=True, detach_encoder=False):
        # detach_encoder allows to stop gradient propogation to encoder
        obs = self.encoder(obs, detach=detach_encoder)
        mu, log_std = self.trunk(obs).chunk(2, dim=-1)

        # constrain log_std inside [log_std_min, log_std_max]
        log_std = torch.tanh(log_std)
        log_std = self.log_std_min + 0.5 * (self.log_std_max - self.log_std_min) * (log_std + 1)

        self.outputs['mu'], self.outputs['std'] = mu, log_std.exp()

        if compute_pi:
            std = log_std.exp()
            noise = torch.randn_like(mu)
            pi = mu + noise * std
        else:
            pi = None
            entropy = None

        if compute_log_pi:
            log_pi = gaussian_logprob(noise, log_std)
        else:
            log_pi = None

        mu, pi, log_pi = squash(mu, pi, log_pi)
        return mu, pi, log_pi, log_std


class QFunction(nn.Module):
    """MLP for q-function."""
    def __init__(self, obs_dim, action_dim, hidden_dim):
        super().__init__()

        self.trunk = nn.Sequential(nn.Linear(obs_dim + action_dim, hidden_dim), nn.ReLU(),
                                   nn.Linear(hidden_dim, hidden_dim),
                                   nn.ReLU(), nn.Linear(hidden_dim, 1))

    def forward(self, obs, action):
        assert obs.size(0) == action.size(0)
        obs_action = torch.cat([obs, action], dim=1)
        return self.trunk(obs_action)


class Critic(nn.Module):
    """Critic network, employes two q-functions."""
    def __init__(
        self, 
        obs_shape, 
        action_shape, 
        hidden_dim, 
        encoder_feature_dim, 
        num_layers,
        patch_size,
        in_channels,
        configs,
    ):
        super().__init__()

        self.encoder = get_encoder(obs_shape, action_shape, configs)

        self.Q1 = QFunction(encoder_feature_dim, action_shape[0],
                            hidden_dim)
        self.Q2 = QFunction(encoder_feature_dim, action_shape[0],
                            hidden_dim)

        self.outputs = dict()
        self.apply(weight_init_conv)

    def forward(self, obs, action, detach_encoder=False):
        # detach_encoder allows to stop gradient propogation to encoder
        obs = self.encoder(obs, detach=detach_encoder)

        q1 = self.Q1(obs, action)
        q2 = self.Q2(obs, action)

        self.outputs['q1'] = q1
        self.outputs['q2'] = q2

        return q1, q2


class Intensity(nn.Module):
    def __init__(self, scale):
        super().__init__()
        self.scale = scale

    def forward(self, x):
        r = torch.randn((x.size(0),) + (1,) * (x.dim() - 1), device=x.device)
        noise = 1.0 + (self.scale * r.clamp(-2.0, 2.0))
        return x * noise


class SpatialTemporalMaskModule(nn.Module):
    def __init__(
        self,
        clip_shape,
        patch_shape,
        obs_shape,
        action_shape,
        critic, 
        augmentation, 
        aug_prob, 
        encoder_feature_dim, 
        hidden_dim, 
        num_dec_layers, 
        num_dec_heads, 
        mask_ratio, 
        configs,
    ):
        super().__init__()
        self.aug_prob = aug_prob

        self.masker = STGraphMaskGenerator(clip_shape=clip_shape, patch_shape=patch_shape, mask_ratio=mask_ratio)

        self.encoder = critic.encoder
        self.target_encoder = copy.deepcopy(critic.encoder)
        
        # MLP head on Predictive Decoder
        self.global_final_classifier = nn.Sequential(
            nn.Linear(encoder_feature_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, encoder_feature_dim))
        
        # MaskedStateTransitionDecoder
        self.decoder = get_decoder(obs_shape, action_shape, configs)
        
        # bilinear product on temporal contrastive
        self.W = nn.Parameter(torch.rand(encoder_feature_dim, encoder_feature_dim))

        ''' Data augmentation '''
        self.transforms = []
        self.eval_transforms = []
        self.uses_augmentation = True
        self.aug_names = augmentation
        for aug in augmentation:
            if aug == "crop":
                transformation = RandomCrop((configs.image_size[0], configs.image_size[1]))
                eval_transformation = CenterCrop((configs.image_size[0], configs.image_size[1]))                
            elif aug == "intensity":
                transformation = Intensity(scale=0.05)
                eval_transformation = nn.Identity()
            elif aug == "none":
                transformation = eval_transformation = nn.Identity()
            else:
                raise NotImplementedError()
            self.transforms.append(transformation)
            self.eval_transforms.append(eval_transformation)

    def apply_transforms(self, transforms, eval_transforms, image):
        if eval_transforms is None:
            for transform in transforms:
                image = transform(image)
        else:
            for transform, eval_transform in zip(transforms, eval_transforms):
                image = utils.maybe_transform(image, transform, eval_transform, p=self.aug_prob)
        return image

    @torch.no_grad()
    def transform(self, images, augment=False):
        images = images.float()
        if augment:
            processed_images = self.apply_transforms(self.transforms, self.eval_transforms, images)
        else:
            processed_images = self.apply_transforms(self.eval_transforms, None, images)
        return processed_images

    def contras_smooth_loss(self, x_pred, x_target, block_size=6, min_temp=0.07, temp_skip=0.075):
        """
        x_pred: [B, T, Z]
        x_target: [B, T, Z]
        """
        x_pred = self.global_final_classifier(x_pred)  # pred

        # [B, T, Z]
        pred_embeddings = F.normalize(x_pred, p=2, dim=-1, eps=1e-3)
        targ_embeddings = F.normalize(x_target, p=2, dim=-1, eps=1e-3)
        B, T = pred_embeddings.size(0), pred_embeddings.size(1)

        pred_embeddings = pred_embeddings.view(-1, *pred_embeddings.shape[2:])  # [BT, Z]
        targ_embeddings = targ_embeddings.view(-1, *targ_embeddings.shape[2:])  # [BT, Z]
        sim = pred_embeddings @ self.W @ targ_embeddings.T
        sm_loss = temp_contrastive_efficient(sim, T, min_temp, max_levels=block_size, skip=temp_skip)

        return sm_loss


class MOOSSAgent(object):
    def __init__(
        self,
        # general parameters
        obs_shape,              # this is the model input shape
        action_shape,
        # actor-critic parameters
        ac_hidden_dim=1024,
        encoder_feature_dim=64,
        num_layers=4,
        patch_size=(10, 10),
        in_channels=12,         # RGBD=4, frame_stack=3
        actor_log_std_min=-10,
        actor_log_std_max=2,
        # STM: masker and transition decoder parameters
        clip_frames=16,
        block_size=4,
        augmentation=[],
        aug_prob=1.0,
        num_dec_layers=2, 
        num_dec_heads=4,
        mask_ratio=0.5,
        proj_hidden_dim=100,
        # RL hyper-parameters
        discount=0.99,
        init_temperature=0.01,
        alpha_lr=1e-3,
        alpha_beta=0.9,
        actor_lr=1e-3,
        actor_beta=0.9,
        actor_update_freq=2,
        critic_lr=1e-3,
        critic_beta=0.9,
        critic_tau=0.005,
        critic_target_update_freq=2,
        encoder_tau=0.005,
        auxiliary_task_lr=1e-3,
        momentum_tau=1.0,
        log_interval=100,
        detach_encoder=False,
        device=None,
        configs=None,
    ):
        self.image_size = configs.image_size
        self.device = device

        self.encoder_type = configs.encoder_type
        self.decoder_type = configs.decoder_type
        self.configs = configs
        
        self.discount = discount
        self.critic_tau = critic_tau
        self.encoder_tau = encoder_tau
        self.momentum_tau = momentum_tau
        self.actor_update_freq = actor_update_freq
        self.critic_target_update_freq = critic_target_update_freq
        self.log_interval = log_interval

        self.detach_encoder = detach_encoder
        self.encoder_feature_dim = encoder_feature_dim

        self.actor = Actor(obs_shape, action_shape, ac_hidden_dim, 
            encoder_feature_dim, num_layers, patch_size, in_channels, 
            actor_log_std_min, actor_log_std_max, configs).to(device)

        self.critic = Critic(obs_shape, action_shape, ac_hidden_dim, encoder_feature_dim, 
            num_layers, patch_size, in_channels, configs).to(device)

        self.critic_target = Critic(obs_shape, action_shape, ac_hidden_dim, encoder_feature_dim, 
            num_layers, patch_size, in_channels, configs).to(device)

        # tie critic and critic_target
        self.critic_target.load_state_dict(self.critic.state_dict())

        # tie encoders between actor and critic
        self.actor.encoder.copy_conv_weights_from(self.critic.encoder)

        self.log_alpha = torch.tensor(np.log(init_temperature)).to(device)
        self.log_alpha.requires_grad = True
        # set target entropy to -|A|
        self.target_entropy = -np.prod(action_shape)

        # optimizers
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr, betas=(actor_beta, 0.999))
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=critic_lr, betas=(critic_beta, 0.999))
        self.log_alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=alpha_lr, betas=(alpha_beta, 0.999))

        ''' STM configuration '''
        clip_shape = (obs_shape[-2], obs_shape[-1], clip_frames)
        patch_shape = (patch_size[-2], patch_size[-1], block_size)
        self.STM = SpatialTemporalMaskModule(clip_shape, patch_shape, obs_shape, action_shape,
            self.critic, augmentation, aug_prob, encoder_feature_dim, proj_hidden_dim, 
            num_dec_layers, num_dec_heads, mask_ratio, configs).to(device)
        self.stm_optimizer = torch.optim.Adam(self.STM.parameters(), lr=0.5*auxiliary_task_lr)

        # learning related
        warmup = True
        adam_warmup_step = 6e3
        if warmup:
            lrscheduler = InverseSquareRootSchedule(adam_warmup_step)
            lrscheduler_lambda = lambda x: lrscheduler.step(x)
            self.stm_lrscheduler = torch.optim.lr_scheduler.LambdaLR(self.stm_optimizer, lrscheduler_lambda)
        else:
            self.stm_lrscheduler = None

        self.train()
        self.critic_target.train()
        self.STM.train()

    def train(self, training=True):
        self.training = training
        self.actor.train(training)
        self.critic.train(training)

    @property
    def alpha(self):
        return self.log_alpha.exp()

    def select_action(self, obs):
        with torch.no_grad():
            obs = torch.FloatTensor(obs).to(self.device)
            obs = obs.unsqueeze(0)
            mu, _, _, _ = self.actor(obs, compute_pi=False, compute_log_pi=False)
            return mu.cpu().data.numpy().flatten()

    def sample_action(self, obs):
        if obs.shape[-1] != self.image_size[-1] or obs.shape[-2] != self.image_size[-2]:
            obs = utils.center_crop_image(obs, self.image_size[-1])
        with torch.no_grad():
            obs = torch.FloatTensor(obs).to(self.device)
            obs = obs.unsqueeze(0)
            mu, pi, _, _ = self.actor(obs, compute_log_pi=False)
            return pi.cpu().data.numpy().flatten()

    def update_critic(self, obs, action, reward, next_obs, not_done, L, step):
        with torch.no_grad():
            _, policy_action, log_pi, _ = self.actor(next_obs)
            target_Q1, target_Q2 = self.critic_target(next_obs, policy_action)
            target_V = torch.min(target_Q1,
                                 target_Q2) - self.alpha.detach() * log_pi
            target_Q = reward + (not_done * self.discount * target_V)

        # get current Q estimates
        current_Q1, current_Q2 = self.critic(
            obs, action, detach_encoder=self.detach_encoder)
        critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(
            current_Q2, target_Q)
        if step % self.log_interval == 0:
            L.log('train_critic/loss', critic_loss, step)

        # Optimize the critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

    def update_actor_and_alpha(self, obs, L, step):
        # detach encoder, so we don't update it with the actor loss
        _, pi, log_pi, log_std = self.actor(obs, detach_encoder=True)

        actor_Q1, actor_Q2 = self.critic(obs, pi, detach_encoder=True)

        actor_Q = torch.min(actor_Q1, actor_Q2)
        actor_loss = (self.alpha.detach() * log_pi - actor_Q).mean()

        if step % self.log_interval == 0:
            L.log('train_actor/loss', actor_loss, step)
            L.log('train_actor/target_entropy', self.target_entropy, step)
        
        entropy = 0.5 * log_std.shape[1] * \
            (1.0 + np.log(2 * np.pi)) + log_std.sum(dim=-1)
        if step % self.log_interval == 0:
            L.log('train_actor/entropy', entropy.mean(), step)

        # optimize the actor
        self.actor_optimizer.zero_grad()
        actor_loss.backward(retain_graph=True)
        self.actor_optimizer.step()

        self.log_alpha_optimizer.zero_grad()
        alpha_loss = (self.alpha * (-log_pi - self.target_entropy).detach()).mean()
        if step % self.log_interval == 0:
            L.log('train_alpha/loss', alpha_loss, step)
            L.log('train_alpha/value', self.alpha, step)
        alpha_loss.backward()
        self.log_alpha_optimizer.step()

    def update_stm(self, stm_kwargs, L, step):
        observation = stm_kwargs["observation"] # [T, B, 9, 1, pre_H, pre_W]
        action = stm_kwargs["action"]           # [T, B, dim_A]
        reward = stm_kwargs["reward"]           # [T, B, 1]

        T, B, C = observation.size()[:3]
        Z = self.encoder_feature_dim

        ### step 1: augmentation ###
        x = rearrange(observation.squeeze(-3), 't b c h w -> b t c h w')
        x = rearrange(x, 'b t c h w -> (b t) c h w')
        x = self.STM.transform(x, augment=True)
        x = rearrange(x, '(b t) c h w -> b t c h w', b=B)

        ### step 2: masking ##
        mask = self.STM.masker(sh_factor=self.configs.patch_size[0], sw_factor=self.configs.patch_size[1]).squeeze(1)  # (T, H, W)
        mask = repeat(mask, 't h w -> (b t) 1 h w', b=B).to(x.device)

        ### step 3: encode ###
        x = rearrange(x, 'b t c h w -> (b t) c h w')
        x = x * mask.float()
        x = self.STM.encoder(x) # [(BT), Z]
        x = x.view(B, T, Z)                 # x is the encoded states

        ### step 4: decode ###
        x = self.STM.decoder(x, action.permute(1, 0, 2))    # [B, 2*T, Z]
        
        ### step 5: get states ###
        recon_curr_states = x[:,::2,:].view(B*T, Z)  # [BT, Z]

        ### step 6: process targets (unmasked) ###
        target_x = rearrange(observation.squeeze(-3), 't b c h w -> b t c h w')
        target_x = rearrange(target_x, 'b t c h w -> (b t) c h w')
        target_x = self.STM.transform(target_x, augment=True)
        target_x = rearrange(target_x, '(b t) c h w -> b t c h w', b=B)
        target_x = rearrange(target_x, 'b t c h w -> (b t) c h w')
        with torch.no_grad():
            target_x = self.STM.target_encoder(target_x)    # [(BT), Z]

        ### step 7: prepare losses ###
        x_pred = rearrange(recon_curr_states, '(b t) z -> b t z', b=B)
        x_target = rearrange(target_x, '(b t) z -> b t z', b=B)

        conts_loss = self.STM.contras_smooth_loss(x_pred, x_target, block_size=self.configs.conts_bs, \
        min_temp=self.configs.conts_mint, temp_skip=self.configs.conts_skip)

        loss = self.configs.conts_weight * conts_loss

        self.stm_optimizer.zero_grad()
        loss.backward()
        model_grad_norm = torch.nn.utils.clip_grad_norm_(self.STM.parameters(), 10)
        self.stm_optimizer.step()

        if step % self.log_interval == 0:
            L.log('train/conts_loss', conts_loss, step)
            L.log('train/stm_loss', loss, step)

        if self.stm_lrscheduler is not None:
            self.stm_lrscheduler.step()
            L.log('train/stm_lr', self.stm_optimizer.param_groups[0]['lr'], step)

    def update(self, replay_buffer, L, step):
        elements = replay_buffer.sample_spr()
        obs, action, reward, next_obs, not_done, stm_kwargs = elements          # [b, c, h, w]; [b, 1]; [b, 1]

        if step % self.log_interval == 0:
            L.log('train/batch_reward', reward.mean(), step)
        
        self.update_critic(obs, action, reward, next_obs, not_done, L, step)
        self.update_stm(stm_kwargs, L, step)

        if step % self.actor_update_freq == 0:
            self.update_actor_and_alpha(obs, L, step)

        if step % self.critic_target_update_freq == 0:
            utils.soft_update_params(self.critic.Q1, self.critic_target.Q1,
                                     self.critic_tau)
            utils.soft_update_params(self.critic.Q2, self.critic_target.Q2,
                                     self.critic_tau)
            utils.soft_update_params(self.critic.encoder,
                                     self.critic_target.encoder,
                                     self.encoder_tau)
            utils.soft_update_params(self.STM.encoder,
                                     self.STM.target_encoder,
                                     self.momentum_tau)

    def save(self, model_dir, step, **kwargs):
        data = {
            'step': step,
            'actor': self.actor.state_dict(),
            'critic': self.critic.state_dict(),
            'critic_target': self.critic_target.state_dict(),
            'stm': self.STM.state_dict(),
            'log_alpha': self.log_alpha.item(),
            'actor_optimizer': self.actor_optimizer.state_dict(),
            'critic_optimizer': self.critic_optimizer.state_dict(),
            'stm_optimizer': self.stm_optimizer.state_dict(),
            'log_alpha_optimizer': self.log_alpha_optimizer.state_dict(),
            'stm_lrscheduler': self.stm_lrscheduler.state_dict() if exists(self.stm_lrscheduler) else None,
        }
        for key, value in kwargs.items():
            data[key] = value
        torch.save(data, '%s/ckpt_%s.pt' % (model_dir, step))

    def load(self, model_dir):
        checkpoint_path = '%s/ckpt_%s.pt' % (model_dir, search_last_step(model_dir))
        data = torch.load(checkpoint_path)

        self.actor.load_state_dict(data['actor'])
        self.critic.load_state_dict(data['critic'])
        self.critic_target.load_state_dict(data['critic_target'])
        self.STM.load_state_dict(data['stm'])

        self.log_alpha.data.copy_(torch.tensor(data['log_alpha']))

        self.actor_optimizer.load_state_dict(data['actor_optimizer'])
        self.critic_optimizer.load_state_dict(data['critic_optimizer'])
        self.stm_optimizer.load_state_dict(data['stm_optimizer'])
        self.log_alpha_optimizer.load_state_dict(data['log_alpha_optimizer'])

        if exists(data['stm_lrscheduler']):
            self.stm_lrscheduler.load_state_dict(data['stm_lrscheduler'])  
        else: self.scheduler = None

        additional_data = {k: v for k, v in data.items() if k not in self.__dict__}

        return data['step'], additional_data
