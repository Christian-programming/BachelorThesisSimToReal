import kornia
import sys
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class ReplayBuffer(object):
    """Buffer to store environment transitions."""
    def __init__(self, obs_shape, action_shape, capacity, image_pad, device):
        """ Create a replay buffer for images with data augmentation

        Args:
            param1(tuple): shape of images to save
            param2(tuple): shape of actions to save
            param3(int): amount of traj to save
            param4(int): amount of pixel for random shifts
            param5(str): device used

        """
        self.capacity = capacity
        self.device = device
        self.aug_trans = nn.Sequential(
                nn.ReplicationPad2d(image_pad),
                kornia.augmentation.RandomCrop((obs_shape[-1], obs_shape[-1])))
        self.obses = np.empty((capacity, *obs_shape), dtype=np.uint8)
        self.next_obses = np.empty((capacity, *obs_shape), dtype=np.uint8)
        self.actions = np.empty((capacity, *action_shape), dtype=np.float32)
        self.rewards = np.empty((capacity, 1), dtype=np.float32)
        self.not_dones = np.empty((capacity, 1), dtype=np.bool)
        self.not_dones_no_max = np.empty((capacity, 1), dtype=np.bool)
        self.idx = 0
        self.full = False

    def __len__(self):
        return self.capacity if self.full else self.idx

    def add(self, obs, action, reward, next_obs, done, done_no_max):
        """ Add traj to the memory
        Args:
            param1(numpy array): obs images to save
            param2(numpy array): action actions to save
            param3(int):  reward reward to save
            param4(numpy array): next_obs images of the next_state
            param5(bool): done signal
            param5(bool): done_no_max signal
        """
        np.copyto(self.obses[self.idx], obs)
        np.copyto(self.actions[self.idx], action)
        np.copyto(self.rewards[self.idx], reward)
        np.copyto(self.next_obses[self.idx], next_obs)
        np.copyto(self.not_dones[self.idx], not done)
        np.copyto(self.not_dones_no_max[self.idx], not done_no_max)
        self.idx = (self.idx + 1) % self.capacity
        self.full = self.full or self.idx == 0

    def sample(self, batch_size):
        """ sample uniformly from memory return amount of batch_size
            Return 2 augmentated states and next state
        Args:
            param1(int): batch_size

        Return
        """
        idxs = np.random.randint(0, self.capacity if self.full else self.idx, size=batch_size)
        obses = self.obses[idxs]
        next_obses = self.next_obses[idxs]
        obses_aug = obses.copy()
        next_obses_aug = next_obses.copy()
        obses = torch.as_tensor(obses, device=self.device).float()
        next_obses = torch.as_tensor(next_obses, device=self.device).float()
        obses_aug = torch.as_tensor(obses_aug, device=self.device).float()
        next_obses_aug = torch.as_tensor(next_obses_aug, device=self.device).float()
        actions = torch.as_tensor(self.actions[idxs], device=self.device)
        rewards = torch.as_tensor(self.rewards[idxs], device=self.device)
        not_dones_no_max = torch.as_tensor(self.not_dones_no_max[idxs], device=self.device)
        obses = self.aug_trans(obses)
        next_obses = self.aug_trans(next_obses)
        obses_aug = self.aug_trans(obses)
        next_obses_aug = self.aug_trans(next_obses)

        return obses, actions, rewards, next_obses, not_dones_no_max, obses_aug, next_obses_aug

    def save_memory(self, filename):
        """
        Use numpy save function to store the data in a given file
        """
        # create folder if not exist
        os.makedirs(filename)

        with open(filename + '/obses.npy', 'wb') as f:
            np.save(f, self.obses)

        with open(filename + '/actions.npy', 'wb') as f:
            np.save(f, self.actions)

        with open(filename + '/next_obses.npy', 'wb') as f:
            np.save(f, self.next_obses)

        with open(filename + '/rewards.npy', 'wb') as f:
            np.save(f, self.rewards)

        with open(filename + '/not_dones.npy', 'wb') as f:
            np.save(f, self.not_dones)

        with open(filename + '/not_dones_no_max.npy', 'wb') as f:
            np.save(f, self.not_dones_no_max)

        with open(filename + '/index.txt', 'w') as f:
            f.write("{}".format(self.idx))


        print("Save memory of size {} to filename {} ".format(self.idx, filename))

    def load_memory(self, filename):
        """
        Use numpy load function to store the data in a given file
        """
        with open(filename + '/obses.npy', 'rb') as f:
            self.obses = np.load(f)

        with open(filename + '/actions.npy', 'rb') as f:
            self.actions = np.load(f)

        with open(filename + '/next_obses.npy', 'rb') as f:
            self.next_obses = np.load(f)

        with open(filename + '/rewards.npy', 'rb') as f:
            self.rewards = np.load(f)

        with open(filename + '/not_dones.npy', 'rb') as f:
            self.not_dones = np.load(f)

        with open(filename + '/not_dones_no_max.npy', 'rb') as f:
            self.not_dones_no_max = np.load(f)

        with open(filename + '/index.txt', 'r') as f:
            self.idx = int(f.read())


        print("Load memory of size {} from filename {} ".format(self.idx, filename))
