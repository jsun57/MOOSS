import copy
import json
import math
import os
import random
import sys
import time
import argparse
import dmc2gym
import gym
import numpy as np
import torch
import logging
logging.basicConfig(stream=sys.stdout, level=logging.INFO)

import utils
from utils import Logger
from utils import VideoRecorder
from utils import Config, cfg2dic, BASE_NAME
from utils import memory_usage_psutil
from models.mooss import MOOSSAgent


def set_global_seed(seed, reproducible=False):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if torch.cuda.is_available():
        if seed != -1:
            # reproducible but slower
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
        else:
            # not reproducible but faster
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.deterministic = False


def evaluate(env, agent, video, num_episodes, L, step, args):
    all_ep_rewards = []

    def run_eval_loop(sample_stochastically=True):
        start_time = time.time()
        prefix = 'stochastic_' if sample_stochastically else ''
        for i in range(num_episodes):
            obs = env.reset()

            video.init(enabled=(i == 0))
            done = False
            episode_reward = 0
            while not done:
                # center crop image
                if obs.shape[-1] != args.image_size[-1] or obs.shape[-1] != args.image_size[-2]:
                    obs = utils.center_crop_image(obs, args.image_size[-1])
                with utils.eval_mode(agent):
                    if sample_stochastically:
                        action = agent.sample_action(obs)
                    else:
                        action = agent.select_action(obs)
                obs, reward, done, _ = env.step(action)
                video.record(env)
                episode_reward += reward

            video.save('%d.mp4' % step)
            all_ep_rewards.append(episode_reward)

        L.log('eval/' + prefix + 'eval_time', time.time() - start_time, step)
        mean_ep_reward = np.mean(all_ep_rewards)
        best_ep_reward = np.max(all_ep_rewards)
        std_ep_reward = np.std(all_ep_rewards)
        median_ep_reward = np.median(all_ep_rewards)
        L.log('eval/' + prefix + 'mean_episode_reward', mean_ep_reward, step)
        L.log('eval/' + prefix + 'std_episode_reward', std_ep_reward, step)
        L.log('eval/' + prefix + 'median_episode_reward', median_ep_reward, step)
        L.log('eval/' + prefix + 'best_episode_reward', best_ep_reward, step)

    run_eval_loop(sample_stochastically=False)
    L.dump(step)
    return all_ep_rewards


def make_agent(obs_shape, action_shape, args, device):
    return MOOSSAgent(
        obs_shape=obs_shape,
        action_shape=action_shape,
        # actor-critic parameters
        ac_hidden_dim=args.ac_hidden_dim,
        encoder_feature_dim=args.encoder_feature_dim,
        num_layers=args.num_layers,
        patch_size=args.patch_size,
        in_channels=3*args.frame_stack,         # RGB=3, frame_stack=3
        actor_log_std_min=args.actor_log_std_min,
        actor_log_std_max=args.actor_log_std_max,
        # STM: masker and transition decoder parameters
        clip_frames=args.clip_frames,
        block_size=args.block_size,
        augmentation=args.augmentation,
        aug_prob=args.aug_prob,
        num_dec_layers=args.num_dec_layers, 
        num_dec_heads=args.num_dec_heads, 
        mask_ratio=args.mask_ratio,
        proj_hidden_dim=args.proj_hidden_dim,
        # RL hyper-parameters
        discount=args.discount,
        init_temperature=args.init_temperature,
        alpha_lr=args.alpha_lr,
        alpha_beta=args.alpha_beta,
        actor_lr=args.actor_lr,
        actor_beta=args.actor_beta,
        actor_update_freq=args.actor_update_freq,
        critic_lr=args.critic_lr,
        critic_beta=args.critic_beta,
        critic_tau=args.critic_tau,
        critic_target_update_freq=args.critic_target_update_freq,
        encoder_tau=args.encoder_tau,
        auxiliary_task_lr=args.auxiliary_task_lr,
        momentum_tau=args.momentum_tau,
        log_interval=args.log_interval,
        detach_encoder=args.detach_encoder,
        device=device,
        configs=args,
    )


def config_repeat_w_act(config):
    config.init_steps *= config.action_repeat
    config.log_interval *= config.action_repeat
    config.actor_update_freq *= config.action_repeat
    config.critic_target_update_freq *= config.action_repeat    


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, default=None)
    parser.add_argument('--d', type=str, default=None)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--load', action='store_true', default=False)
    args, overrides = parser.parse_known_args()
    # everything about configuration
    if len(overrides) > 0:
        print('>>> Override with Caution!!! >>>')
        config = Config(args.cfg, args.d, overrides)
    else:
        config = Config(args.cfg, args.d)
    # set environment variables
    os.environ['MUJOCO_GL'] = 'egl'
    os.environ['MUJOCO_EGL_DEVICE_ID'] = str(args.gpu)
    # seeding
    reproducible = True
    if args.seed == -1:
        args.seed = np.random.randint(1, 1000000)
        reproducible = False
    set_global_seed(args.seed, reproducible)
    # load asserting
    if args.load:
        assert config.save_buffer and config.save_model and config.save_tb
    # device
    device = torch.device('cuda', index=args.gpu) if torch.cuda.is_available() else torch.device('cpu')
    # action repeat
    config_repeat_w_act(config)
    # env setup
    domain_name = config.env_name.split('/')[0]
    task_name = config.env_name.split('/')[1]
    env = dmc2gym.make(domain_name=domain_name, task_name=task_name, seed=args.seed, visualize_reward=False, from_pixels=True, 
        height=config.pre_transform_image_size[0], width=config.pre_transform_image_size[1], frame_skip=config.action_repeat)
    env.seed(args.seed)
    env = utils.FrameStack(env, k=config.frame_stack)

    # make directory 
    if config.exp_id:
        exp_id = config.exp_id
    else:
        exp_id = args.cfg if args.cfg not in BASE_NAME else ''
    domain_task = domain_name + '_' + task_name
    exp_name = exp_id + '_' + domain_task
    work_dir = os.path.join('results', exp_name)
    utils.make_dir(work_dir, load=args.load)
    if not args.load:
        config.write_yaml(work_dir) 
    video_dir = utils.make_dir(os.path.join(work_dir, 'video'), load=args.load)
    model_dir = utils.make_dir(os.path.join(work_dir, 'model'), load=args.load)
    buffer_dir = utils.make_dir(os.path.join(work_dir, 'buffer'), load=args.load)
    video = VideoRecorder(video_dir if config.save_video else None)

    # setup replay buffer
    action_shape = env.action_space.shape
    obs_shape = (3 * config.frame_stack, config.image_size[0], config.image_size[1])
    pre_aug_obs_shape = (3 * config.frame_stack, config.pre_transform_image_size[0], config.pre_transform_image_size[1])

    replay_buffer = utils.ReplayBuffer(
        obs_shape=pre_aug_obs_shape,
        action_shape=action_shape,
        capacity=config.replay_buffer_capacity,
        batch_size=config.batch_size,
        image_size=config.image_size,
        auxiliary_task_batch_size=config.auxiliary_task_batch_size,
        jumps=config.clip_frames-1,
        device=device,
    )

    # setup agent
    agent = make_agent(obs_shape=obs_shape, action_shape=action_shape, args=config, device=device)  # agent focus on model input shape
    replay_buffer.add_agent(agent)
    L = Logger(work_dir, use_tb=config.save_tb, use_wandb=False, loaded=args.load)

    curr_env_step, episode, episode_reward, done = 0, 0, 0, True
    start_time = time.time()

    if args.load:
        curr_env_step, additional_data = agent.load(model_dir)
        replay_buffer.load(buffer_dir)
        L.load()
        print('Agent, replay buffer and logger loaded. Start from step:', curr_env_step)
        episode = additional_data['episode']

    # RAM logging 
    max_memory = 0
    current_memory = memory_usage_psutil()
    max_memory = max(max_memory, current_memory)

    for step in range(curr_env_step, config.num_env_steps, config.action_repeat):
        # evaluate agent periodically, and saving if necessary
        if step % config.eval_freq == 0:
            if (step != curr_env_step and args.load) or (not args.load):
                L.log('eval/episode', episode, step)
                all_rewards = evaluate(env, agent, video, config.num_eval_episodes, L, step, config)
                print("Episode:", episode, "Step:", step, "Rwds:", [round(num, 2) for num in all_rewards])
                if config.save_model and (step == 100000 or step == 500000):
                    agent.save(model_dir, step, episode=episode)
                if config.save_buffer:
                    replay_buffer.save(buffer_dir)
                if config.save_tb:
                    L.save()
        
        # done in the middle with proper logging and reset
        if done:
            if (step > 0 and not args.load) or (args.load and step != curr_env_step):
                if step % config.log_interval == 0:
                    L.log('train/duration', time.time() - start_time, step)
                    L.dump(step)
                start_time = time.time()
            if step % config.log_interval == 0:
                L.log('train/episode_reward', episode_reward, step)

            obs = env.reset()
            done = False
            episode_reward = 0
            episode_step = 0
            episode += 1
            if step % config.log_interval == 0:
                L.log('train/episode', episode, step)

        # sample action for data collection
        if step < config.init_steps:
            action = env.action_space.sample()
        else:
            with utils.eval_mode(agent):
                action = agent.sample_action(obs)

        # run training update
        if step >= config.init_steps:
            num_updates = 1
            for _ in range(num_updates):
                agent.update(replay_buffer, L, step)

        next_obs, reward, done, _ = env.step(action)    # BGR not RGB

        done_bool = 0 if episode_step + 1 == env._max_episode_steps else float(done)
        episode_reward += reward
        replay_buffer.add(obs, action, reward, next_obs, done_bool)

        obs = next_obs
        episode_step += 1

        current_memory = memory_usage_psutil()
        max_memory = max(max_memory, current_memory)
    print(f"Maximum program memory used: {max_memory} GB")

if __name__ == '__main__':
    start_time = time.time()
    torch.cuda.empty_cache()
    torch.multiprocessing.set_start_method('spawn')
    main()
    end_time = time.time()
    duration = end_time - start_time
    # Convert to hours, minutes, seconds
    hours = int(duration // 3600)
    minutes = int((duration % 3600) // 60)
    seconds = round(duration % 60)
    print(f"Total program runtime: {hours:02d}:{minutes:02d}:{seconds:02d}")