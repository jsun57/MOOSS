import torch
import torch.nn as nn
import numpy as np
import time
import math
import gym
import os
from collections import deque
from torch.utils.data import Dataset
from .image_utils import random_crop, center_crop, sync_crop


class InverseSquareRootSchedule(object):
    def __init__(self, warmup_step=4e4):
        if warmup_step is None:
            self.warmup_step = warmup_step
        else:
            warmup_step = int(warmup_step)
            assert warmup_step > 0 and isinstance(warmup_step, int)
            self.warmup_step = warmup_step
            init = 5e-4
            end = 1
            self.init_lr = init
            self.lr_step = (end - init) / warmup_step
            self.decay = warmup_step ** 0.5

    def step(self, step):
        if self.warmup_step is None:
            return  1
        else:
            if step < self.warmup_step:
                return self.init_lr + self.lr_step * step
            else:
                return self.decay * (step ** -0.5)


class AnneallingSchedule(object):
    def __init__(self, warmup_step=4e4):
        if warmup_step is None:
            self.warmup_step = warmup_step
        else:
            warmup_step = int(warmup_step)
            assert warmup_step > 0 and isinstance(warmup_step, int)
            self.warmup_step = warmup_step
            self.decay = warmup_step ** 0.5

    def step(self, step):
        if self.warmup_step is None:
            return  1
        else:
            if step < self.warmup_step:
                return 1
            else:
                return self.decay * (step ** -0.5)


class PositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_len=128):
        super().__init__()

        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model).float()
        pe.require_grad = False

        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)).exp()

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, length):
        return self.pe[:, :length]


class ReplayBuffer(Dataset):
    """Buffer to store environment transitions."""
    def __init__(
        self,
        obs_shape,
        action_shape,
        capacity,
        batch_size,
        image_size,
        transform=None,
        auxiliary_task_batch_size=64,
        jumps=5,
        device=None,
    ):
        self.capacity = capacity
        self.batch_size = batch_size
        self.image_size = image_size
        self.device = device
        self.transform = transform
        self.auxiliary_task_batch_size = auxiliary_task_batch_size
        self.jumps = jumps
        # the proprioceptive obs is stored as float32, pixels obs as uint8
        obs_dtype = np.float32 if len(obs_shape) == 1 else np.uint8

        self.obses = np.empty((capacity, *obs_shape), dtype=obs_dtype)
        self.next_obses = np.empty((capacity, *obs_shape), dtype=obs_dtype)
        self.actions = np.empty((capacity, *action_shape), dtype=np.float32)
        self.rewards = np.empty((capacity, 1), dtype=np.float32)
        self.not_dones = np.empty((capacity, 1), dtype=np.float32)  # for infinite bootstrap
        self.real_dones = np.empty((capacity, 1), dtype=np.float32) # for auxiliary task

        self.idx = 0
        self.last_save = 0
        self.full = False

    def add(self, obs, action, reward, next_obs, done):

        np.copyto(self.obses[self.idx], obs)
        np.copyto(self.actions[self.idx], action)
        np.copyto(self.rewards[self.idx], reward)
        np.copyto(self.next_obses[self.idx], next_obs)
        np.copyto(self.not_dones[self.idx], not done)   # "not done" is always True
        np.copyto(self.real_dones[self.idx], isinstance(done, int))   # "not done" is always True

        self.idx = (self.idx + 1) % self.capacity
        self.full = self.full or self.idx == 0

    def sample_spr(self):
        # sample batch for auxiliary task 
        idxs = np.random.randint(
            0,  self.capacity - self.jumps - 1 if \
            self.full else self.idx - self.jumps - 1, 
            size=self.auxiliary_task_batch_size * 2   #  size=self.auxiliary_task_batch_size)
        )
        idxs = idxs.reshape(-1, 1)
        step = np.arange(self.jumps + 1).reshape(1, -1) # this is a range
        idxs = idxs + step

        real_dones = torch.as_tensor(self.real_dones[idxs], device=self.device)   # (B, jumps+1, 1)
        # we add this to avoid sampling the episode boundaries
        valid_idxs = torch.where((real_dones.mean(1)==0).squeeze(-1))[0].cpu().numpy()
        idxs = idxs[valid_idxs] # (B, jumps+1)
        idxs = idxs[:self.auxiliary_task_batch_size] if idxs.shape[0] >= self.auxiliary_task_batch_size else idxs

        obses = torch.as_tensor(self.obses[idxs], device=self.device).float()   # (B, jumps+1, 3*3=9, 100, 100)
        next_obses = torch.as_tensor(self.next_obses[idxs], device=self.device).float()
        actions = torch.as_tensor(self.actions[idxs], device=self.device)
        rewards = torch.as_tensor(self.rewards[idxs], device=self.device)
        not_dones = torch.as_tensor(self.not_dones[idxs], device=self.device)   # (B, jumps+1, 1)

        spr_samples = {
            'observation': obses.transpose(0, 1).unsqueeze(3),
            'action': actions.transpose(0, 1),
            'reward': rewards.transpose(0, 1),
        }
        return (*self.sample_aug(original_augment=True), spr_samples)

    def sample_aug(self, original_augment=False):
        idxs = np.random.randint(0, self.capacity if self.full else self.idx, size=self.batch_size)

        obses = self.obses[idxs]
        next_obses = self.next_obses[idxs]

        if not original_augment:
            obses = torch.as_tensor(obses, device=self.device).float()
            next_obses = torch.as_tensor(next_obses, device=self.device).float()
            if hasattr(self, 'STM'):
                obses = self.STM.transform(obses, augment=True)
                next_obses = self.STM.transform(next_obses, augment=True)
            else:
                raise NotImplementedError
        else:
            # 1. Normal
            if obses.shape[-1] != self.image_size[-1] or obses.shape[-2] != self.image_size[-2]:
                obses = random_crop(obses, self.image_size[-1])
                next_obses = random_crop(next_obses, self.image_size[-1])

            obses = torch.as_tensor(obses, device=self.device).float()
            next_obses = torch.as_tensor(next_obses, device=self.device).float()

        actions = torch.as_tensor(self.actions[idxs], device=self.device)
        rewards = torch.as_tensor(self.rewards[idxs], device=self.device)
        not_dones = torch.as_tensor(self.not_dones[idxs], device=self.device)
        return obses, actions, rewards, next_obses, not_dones

    def add_agent(self, agent):
        if hasattr(agent, 'STM'):
            self.STM = agent.STM
        else:
            raise NotImplementedError

    def save(self, save_dir):
        if self.idx == self.last_save:
            return
        path = os.path.join(save_dir, '%d_%d.pt' % (self.last_save, self.idx))
        payload = [
            self.obses[self.last_save:self.idx],
            self.next_obses[self.last_save:self.idx],
            self.actions[self.last_save:self.idx],
            self.rewards[self.last_save:self.idx],
            self.not_dones[self.last_save:self.idx]
        ]
        self.last_save = self.idx
        torch.save(payload, path)

    def load(self, save_dir):
        chunks = os.listdir(save_dir)
        chucks = sorted(chunks, key=lambda x: int(x.split('_')[0]))
        for chunk in chucks:
            start, end = [int(x) for x in chunk.split('.')[0].split('_')]
            path = os.path.join(save_dir, chunk)
            payload = torch.load(path)
            assert self.idx == start
            self.obses[start:end] = payload[0]
            self.next_obses[start:end] = payload[1]
            self.actions[start:end] = payload[2]
            self.rewards[start:end] = payload[3]
            self.not_dones[start:end] = payload[4]
            self.idx = end
        self.last_save = self.idx

    def __getitem__(self, idx):
        idx = np.random.randint(0,
                                self.capacity if self.full else self.idx,
                                size=1)
        idx = idx[0]
        obs = self.obses[idx]
        action = self.actions[idx]
        reward = self.rewards[idx]
        next_obs = self.next_obses[idx]
        not_done = self.not_dones[idx]

        if self.transform:
            obs = self.transform(obs)
            next_obs = self.transform(next_obs)

        return obs, action, reward, next_obs, not_done

    def __len__(self):
        return self.capacity


class FrameStack(gym.Wrapper):
    def __init__(self, env, k):
        gym.Wrapper.__init__(self, env)
        self._k = k
        self._frames = deque([], maxlen=k)
        shp = env.observation_space.shape
        self.observation_space = gym.spaces.Box(
            low=0,
            high=1,
            shape=((shp[0] * k, ) + shp[1:]),
            dtype=env.observation_space.dtype)
        self._max_episode_steps = env._max_episode_steps

    def reset(self):
        obs = self.env.reset()
        for _ in range(self._k):
            self._frames.append(obs)
        return self._get_obs()

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        self._frames.append(obs)
        return self._get_obs(), reward, done, info

    def _get_obs(self):
        assert len(self._frames) == self._k
        return np.concatenate(list(self._frames), axis=0)


class ScaleGrad(torch.autograd.Function):
    """Model component to scale gradients back from layer, without affecting
    the forward pass.  Used e.g. in dueling heads DQN models."""
    @staticmethod
    def forward(ctx, tensor, scale):
        """Stores the ``scale`` input to ``ctx`` for application in
        ``backward()``; simply returns the input ``tensor``."""
        ctx.scale = scale
        return tensor

    @staticmethod
    def backward(ctx, grad_output):
        """Return the ``grad_output`` multiplied by ``ctx.scale``.  Also returns
        a ``None`` as placeholder corresponding to (non-existent) gradient of 
        the input ``scale`` of ``forward()``."""
        return grad_output * ctx.scale, None


# scale_grad = ScaleGrad.apply
# Supply a dummy for documentation to render.
scale_grad = getattr(ScaleGrad, "apply", None)


