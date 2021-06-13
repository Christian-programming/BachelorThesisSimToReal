import os
import cv2
import torch
import numpy as np
from gym.spaces import Box
from gym import Wrapper
from collections import deque


class FrameStack(Wrapper):
    r"""Observation wrapper that stacks the observations in a rolling manner.
    For example, if the number of stacks is 4, then the returned observation contains
    the most recent 4 observations. For environment 'Pendulum-v0', the original observation
    is an array with shape [3], so if we stack 4 observations, the processed observation
    has shape [4, 3].
    .. note::
        To be memory efficient, the stacked observations are wrapped by :class:`LazyFrame`.
    .. note::
        The observation space must be `Box` type. If one uses `Dict`
        as observation space, it should apply `FlattenDictWrapper` at first.
    Example::
        >>> import gym
        >>> env = gym.make('PongNoFrameskip-v0')
        >>> env = FrameStack(env, 4)
        >>> env.observation_space
        Box(4, 210, 160, 3)
    Args:
        env (Env): environment object
        num_stack (int): number of stacks
        lz4_compress (bool): use lz4 to compress the frames internally
    """
    def __init__(self, env, config):
        """ Wrapper to stack n different states to a single one
        Args:
            param1(env): env gym enviroment
            param2(config): config param from json file

        """
        super(FrameStack, self).__init__(env)
        self.state_buffer = deque([], maxlen=config["history_length"])
        self.env = env
        self.size = config["size"]
        self.device = config["device"]
        self.history_length = config["history_length"]

    def step(self, action):
        """  Computes reward stacked mext state and done signal
        Args:
            param(numpy array) action for the env
        """
        observation, reward, done, info = self.env.step(action)
        img = observation
        state = self._create_next_obs(observation["img"])
        return state, reward, done, img

    def reset(self, **kwargs):
        """  resets the env and returns stacked state
        """
        observation = self.env.reset(**kwargs)
        state = self._stacked_frames(observation["img"])
        return state

    def _create_next_obs(self, state):
        """ Take RGB image from step env
            computes the gray scale of it resize it
            and stacks it to the last 2 states

        Args:
            param1(numpy array) image of env
        """
        state = cv2.cvtColor(state, cv2.COLOR_BGR2GRAY)
        state = cv2.resize(state, (self.size, self.size))
        state = torch.tensor(state, dtype=torch.uint8, device=self.device)
        self.state_buffer.append(state)
        state = torch.stack(list(self.state_buffer), 0)
        state = state.cpu()
        obs = np.array(state)
        return obs

    def _stacked_frames(self, state):
        """ creates a for reset the stackt state by adding
            2 zeros matrices at the beginning
            computes the gray scale of it resize it
            and stacks it to the last 2 states

        Args:
            param1(numpy array) image of reset
        """
        state = cv2.cvtColor(state, cv2.COLOR_BGR2GRAY)
        state = cv2.resize(state, (self.size, self.size))
        state = torch.tensor(state, dtype=torch.uint8, device=self.device)
        zeros = torch.zeros_like(state)
        for idx in range(self.history_length - 1):
            self.state_buffer.append(zeros)
        self.state_buffer.append(state)

        state = torch.stack(list(self.state_buffer), 0)
        state = state.cpu()
        obs = np.array(state)
        return obs


def mkdir(base, name):
    """
    Creates a direction if its not exist
    Args:
       param1(string): base first part of pathname
       param2(string): name second part of pathname
    Return: pathname
    """
    path = os.path.join(base, name)
    if not os.path.exists(path):
        os.makedirs(path)
    return path


def write_into_file(pathname, text):
    """ Writes given text in to file
    Args:
       param1(string): pathname name of the file
       param2(string): text to be saved
    """
    with open(pathname + ".txt", "a") as myfile:
        myfile.write(text)
        myfile.write('\n')
