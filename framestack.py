import os
import cv2
import numpy as np
from gym import Wrapper


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
        super(FrameStack, self).__init__(env)
        self.env = env
        self.size = config["size"]
        self.device = config["device"]

    def step(self, action):
        """ takes action and changes the next state in desired shape
        Args:
             param1(np.array): action
        Return: next_state, reward, done, info

        """
        observation, reward, done, info = self.env.step(action)
        observation = cv2.resize(observation, (self.size, self.size))
        observation = np.array(observation, dtype=np.uint8)
        observation = observation.transpose(2, 0, 1)
        info = ""
        return observation, reward, done, info

    def reset(self, **kwargs):
        """ changes reset state in to desired shape """
        observation = self.env.reset(**kwargs)
        observation = cv2.resize(observation, (self.size, self.size))
        observation = np.array(observation, dtype=np.uint8)
        observation = observation.transpose(2, 0, 1)
        return observation


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
    """ creates new textfile in pathname with text

    Args:
       param1(string): pathname
       param2(string): text

    """
    with open(pathname+".txt", "a") as myfile:
        myfile.write(text)
        myfile.write('\n')
