# Copyright 2020
# Author: Christian Leininger <info2016frei@gmail.com>

import os
import sys
import cv2
import gym
import time
import torch
import random
import numpy as np
from collections import deque
from datetime import datetime
from replay_buffer import ReplayBuffer
from agent import TQC
from torch.utils.tensorboard import SummaryWriter
from PIL import Image
from gym_grasping.envs.robot_sim_env import RobotSimEnv
from helper import FrameStack, mkdir, write_into_file


def train_agent(config):
    """ In this function der Agent interact with env to improve policy
    Args:
        param1:(config) param from json file
    """
    now = datetime.now()
    dt_string = now.strftime("%d_%m_%Y_%H:%M:%S")
    torch.manual_seed(config["seed"])
    np.random.seed(config["seed"])
    file_name = config["res_path"] + "/" .format(str(config["env_name"]))
    tensorboard_name = config["res_path"] + '/runs/' + dt_string
    print(tensorboard_name)
    writer = SummaryWriter(tensorboard_name)
    size = config["size"]
    env = gym.make(config["env_name"], renderer='egl')
    env = FrameStack(env, config)
    obs = env.reset()
    print("state ", obs.shape)
    state_dim = 200
    print("State dim ", state_dim)
    action_dim = 5
    print("action_dim ", action_dim)
    max_action = 1
    config["target_entropy"] = -np.prod(action_dim)
    obs_shape = (config["history_length"], size, size)
    action_shape = (action_dim, )
    print("obs", obs_shape)
    print("act", action_shape)
    policy = TQC(state_dim, action_dim, max_action, config)
    replay_buffer = ReplayBuffer(obs_shape, action_shape, int(config["buffer_size"]), config["image_pad"], config["device"])
    total_timesteps = 0
    timesteps_since_eval = 0
    episode_num = 0
    done = True
    t0 = time.time()
    scores_window = deque(maxlen=100)
    episode_reward = 0
    evaluations = []
    tb_update_counter = 0
    evaluations.append(evaluate_policy(policy, writer, total_timesteps, size, env))
    save_model = config["model_path"] + '/model-{}reward_{:.2f}'.format(episode_num, evaluations[-1])
    policy.save(save_model)
    done_counter = deque(maxlen=100)
    text_file = os.path.join(config["res_path"], str(dt_string))
    print("start training")
    while total_timesteps < config["max_timesteps"]:
        tb_update_counter += 1
        # If the episode is done
        if done:
            episode_num += 1
            scores_window.append(episode_reward)
            average_mean = np.mean(scores_window)
            if tb_update_counter > config["tensorboard_freq"]:
                print("Write tensorboard")
                tb_update_counter = 0
                writer.add_scalar('Reward', episode_reward, total_timesteps)
                writer.add_scalar('Reward mean ', average_mean, total_timesteps)
                writer.flush()
            # If we are not at the very beginning, we start the training process of the model
            if total_timesteps != 0:
                if episode_timesteps < 150:
                    done_counter.append(1)
                else:
                    done_counter.append(0)
                goals = sum(done_counter)
                text = "Total Timesteps: {} Episode Num: {} ".format(total_timesteps, episode_num)
                text += "Episode steps {} ".format(episode_timesteps)
                text += "Goal last 100 ep : {} ".format(goals)
                text += "Reward: {:.2f}  Average Re: {:.2f} Time: {}".format(episode_reward, np.mean(scores_window), time_format(time.time()-t0))
                writer.add_scalar('Goal_freq', goals, total_timesteps)

                print(text)
                write_into_file(text_file, text)
            # We evaluate the episode and we save the policy
            if timesteps_since_eval >= config["eval_freq"]:
                timesteps_since_eval %= config["eval_freq"]
                evaluations.append(evaluate_policy(policy, writer, total_timesteps, size,  env))
                save_model = config["model_path"] + '/model-{}reward_{:.2f}'.format(episode_num, evaluations[-1])
                print("Save model to {}".format(save_model))
                policy.save(save_model)
            # When the training step is done, we reset the state of the environment
            obs = env.reset()

            # Set the Done to False
            done = False
            # Set rewards and episode timesteps to zero
            episode_reward = 0
            episode_timesteps = 0
        # Before 10000 timesteps, we play random actions
        if total_timesteps < config["start_timesteps"]:
            action = env.action_space.sample()
        else:  # After 10000 timesteps, we switch to the model
            action = policy.select_action(obs)
        # The agent performs the action in the environment, then reaches the next state and receives the reward
        new_obs, reward, done, _ = env.step(action)
        done = float(done)

        # We check if the episode is done
        done_bool = 0 if episode_timesteps + 1 == config["max_episode_steps"] else float(done)
        if episode_timesteps + 1 == config["max_episode_steps"]:
            done = True
        episode_reward += reward
        # We store the new transition into the Experience Replay memory (ReplayBuffer)
        replay_buffer.add(obs, action, reward, new_obs, done, done_bool)
        # We update the state, the episode timestep, the total timesteps, and the timesteps since the evaluation of the policy
        obs = new_obs
        if total_timesteps > config["start_timesteps"]:
            for i in range(1):
                policy.train(replay_buffer, writer, 1)
        episode_timesteps += 1
        total_timesteps += 1
        timesteps_since_eval += 1
