import os
import sys
import gym
import json
import time
import random
import numpy as np
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from collections import deque
from agent import TQC
import cv2
from PIL import Image
from gym_grasping.envs.robot_sim_env import RobotSimEnv
from helper import FrameStack


def evaluate_policy(policy, args, env, episode=25):
    """
    Args:
       param1(policy): policy
       param2(args): args
       param3(env): gym env
       param4(episode): episode default
    """
    size = args.size
    obs_shape = (args.history_length, size, size)
    action_shape = (args.action_dim, )
    path = "eval"
    goals = 0
    avg_reward = 0
    total_steps = 0
    episodes = 5
    for episode in range(episodes):
        obs = env.reset()
        done = False
        episode_reward = 0
        for step in range(1,args.max_episode_steps + 1):
            action = policy.select_action(np.array(obs))
            obs, reward, done, info = env.step(action)
            episode_reward += reward
            frame = cv2.imwrite("images/{}wi{}.png".format(episode, step), cv2.resize(np.array(info),(300,300)))
            if done:
                if step < 150:
                    total_steps += step
                    goals += 1
                break
            avg_reward += reward
        print("Episode {}  reward {}  goal {} ".format(episode, episode_reward, goals))
    avg_reward /= episodes
    print("reached goal {} of {}".format(goals, episode))
    if goals != 0:
        print("Average step if reached {} ".format(float(total_steps) / goals))
    print("------------------------------------------------------------------------------------------")
    print("Average Reward over the Evaluation Step: {} of {} episodes".format(avg_reward, episodes))
    print("------------------------------------------------------------------------------------------")
    return avg_reward


def main(args):
    """ Evaluates policy starts different tests

    Args:
        param1(args): args

    """
    with open(args.param, "r") as f:
        config = json.load(f)

    env = gym.make(args.env_name, renderer='egl')
    env = FrameStack(env, config)
    state = env.reset()
    state_dim = 200
    action_dim = 5
    args.action_dim = action_dim
    max_action = float(1)
    min_action = float(-1)
    config["target_entropy"] = -np.prod(action_dim)

    policy = TQC(state_dim, action_dim, max_action, config)
    directory = "models/"
    if args.agent is None:
        filename = "kuka_block_grasping-v0-76721reward_-0.88-agentTCQ"  # 91 %
        filename = "kuka_block_grasping-v0-97133reward_-1.05-agentTCQ"  # 93 %
    else:
        filename = args.agent

    filename = directory + filename
    print("Load ", filename)
    policy.load(filename)
    policy.actor.training = False
    evaluate_policy(policy, args,  env, args.epi)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--env-name', default="kuka_block_stacking-v0", type=str, help='Name of a environment (set it to any Continous environment you want')
    parser.add_argument('--param', default="param.json", type=str)
    parser.add_argument('--seed', default=True, type=bool, help='use different seed for each episode')
    parser.add_argument('--epi', default=25, type=int)
    parser.add_argument('--max_episode_steps', default=150, type=int)
    parser.add_argument('--lr-critic', default=0.0005, type=int)               # Total number of iterations/timesteps
    parser.add_argument('--lr-actor', default=0.0005, type=int)               # Total number of iterations/timesteps
    parser.add_argument('--lr_alpha', default=3e-4, type=float)
    parser.add_argument('--lr_decoder', default=1e-4, type=float)      # Divide by 5
    parser.add_argument('--save_model', default=True, type=bool)     # Boolean checker whether or not to save the pre-trained model
    parser.add_argument('--batch_size', default=256, type=int)      # Size of the batch
    parser.add_argument('--discount', default=0.99, type=float)      # Discount factor gamma, used in the calculation of the total discounted reward
    parser.add_argument('--tau', default=0.005, type=float)        # Target network update rate
    parser.add_argument('--size', default=84, type=int)
    parser.add_argument('--num_q_target', default=4, type=int)    # amount of qtarget nets
    parser.add_argument('--tensorboard_freq', default=5000, type=int)    # every nth episode write in to tensorboard
    parser.add_argument('--device', default='cuda', type=str)    # amount of qtarget nets
    parser.add_argument("--n_quantiles", default=25, type=int)
    parser.add_argument("--top_quantiles_to_drop_per_net", default=2, type=int)
    parser.add_argument("--n_nets", default=5, type=int)
    parser.add_argument('--history_length', default=3, type=int)
    parser.add_argument('--image_pad', default=4, type=int)     #
    parser.add_argument('--actor_clip_gradient', default=1., type=float)     # Maximum value of the Gaussian noise added to
    parser.add_argument('--locexp', type=str)     # Maximum value
    parser.add_argument('--agent', type=str)     # Maximum value
    parser.add_argument('--debug', default=False, type=bool)
    parser.add_argument('--eval', type=bool, default=True)
    parser.add_argument('--buffer_size', default=3.5e5, type=int)
    arg = parser.parse_args()
    main(arg)
