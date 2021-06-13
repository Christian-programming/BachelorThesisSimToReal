import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn import Module, Linear
from torch.distributions import Distribution, Normal
from torch.nn.functional import relu, logsigmoid


class Actor(Module):
    def __init__(self, state_dim, action_dim, config):
        """  Actor network predict mean and var
             of a gaussian of the actio

        Args:
            param1(int): state_dim vector size of feature latent size
            param2(int): action_dim vector size of action
            param2(config): param from json file
        """
        super().__init__()
        self.device = config["device"]
        self.log_std_min_max = (-20, 2)
        self.action_dim = action_dim
        self.net = Mlp(state_dim, [256, 256], 2 * action_dim)

    def forward(self, obs):
        """  forward path of the network
        Args:
            param1(numpy array): latent vector of image to predict action

        Return mean of action, log
        """
        mean, log_std = self.net(obs).split([self.action_dim, self.action_dim], dim=1)
        log_std = log_std.clamp(*self.log_std_min_max)
        if self.training:
            std = torch.exp(log_std)
            tanh_normal = TanhNormal(mean, std, self.device)
            action, pre_tanh = tanh_normal.rsample()
            log_prob = tanh_normal.log_prob(pre_tanh)
            log_prob = log_prob.sum(dim=1, keepdim=True)
        else:  # deterministic eval without log_prob computation
            action = torch.tanh(mean)
            log_prob = None
        return action, log_prob

    def select_action(self, obs):
        """  Returns action to the given latent vector
        Args:
            param1(numpy array): latent vector of image to predict action

        Return  actions vector of action dim size
        """
        action, _ = self.forward(obs)
        action = action[0].cpu().detach().numpy()
        return action


class Critic(Module):
    """  Implements the Ensemble of Distribution Critic proposed in the paper """
    def __init__(self, state_dim, action_dim, config):
        """
        Args:
            param1(int): state_dim vector size of feature latent size
            param2(int): action_dim vector size of action
            param2(config): param from json file
        """
        super().__init__()
        self.nets = []
        self.n_quantiles = config["n_quantiles"]
        self.n_nets = config["n_nets"]
        for i in range(self.n_nets):
            net = Mlp(state_dim + action_dim, [512, 512, 512], self.n_quantiles)
            self.add_module(f'qf{i}', net)
            self.nets.append(net)

    def forward(self, state, action):
        """ Computes forward path to get the values of the different quantiles of
            the  q-value distribution
        Args:
            param1(int): state_dim vector size of feature latent size
            param2(int): action_dim vector size of action
        """
        sa = torch.cat((state, action), dim=1)
        quantiles = torch.stack(tuple(net(sa) for net in self.nets), dim=1)
        return quantiles


class TanhNormal(Distribution):
    def __init__(self, normal_mean, normal_std, device):
        super().__init__()
        self.normal_mean = normal_mean
        self.normal_std = normal_std
        self.standard_normal = Normal(torch.zeros_like(self.normal_mean, device=device), torch.ones_like(self.normal_std, device=device))
        self.normal = Normal(normal_mean, normal_std)

    def log_prob(self, pre_tanh):
        log_det = 2 * np.log(2) + logsigmoid(2 * pre_tanh) + logsigmoid(-2 * pre_tanh)
        result = self.normal.log_prob(pre_tanh) - log_det
        return result

    def rsample(self):
        pretanh = self.normal_mean + self.normal_std * self.standard_normal.sample()
        return torch.tanh(pretanh), pretanh


class Mlp(Module):
    """  Single Fully Connected Network from the Ensemble """
    def __init__(self, input_size, hidden_sizes, output_size):
        """  Set parameter of the class
        Args:
            param1(int): input_size vector size of feature latent size
            param2(list): hidden_sizes amount of hidden layers and nodes for each
            param3(int): amount of atoms to represent the distribution
        """
        super().__init__()
        self.fcs = []
        in_size = input_size
        for i, next_size in enumerate(hidden_sizes):
            fc = Linear(in_size, next_size)
            self.add_module(f'fc{i}', fc)
            self.fcs.append(fc)
            in_size = next_size
        self.last_fc = Linear(in_size, output_size)

    def forward(self, h):
        """ Comutes forward path to network
        Args:
            param1(numpy array): state vector from image
        """

        for fc in self.fcs:
            h = relu(fc(h))
        output = self.last_fc(h)
        return output


class Encoder(nn.Module):
    """ Extracts features from image """
    def __init__(self, config, D_out=200, conv_channels=[16, 32], kernel_sizes=[8, 4], strides=[4, 2]):
        """ creates encoder netowrk wit cnn
        Args:
            param1(config): param dict from json file
            param2(int): dim of latent space
            param3(list): conv_channels of the different layers
            param4(list): kernel_sizes
            param5(list): strides
        """
        super(Encoder, self).__init__()
        # Defining the first Critic neural network
        channels = config["history_length"]
        self.conv_1 = torch.nn.Conv2d(channels, conv_channels[0], kernel_sizes[0], strides[0])
        self.relu_1 = torch.nn.ReLU()
        self.conv_2 = torch.nn.Conv2d(conv_channels[0], conv_channels[1], kernel_sizes[1], strides[1])
        self.relu_2 = torch.nn.ReLU()
        self.flatten = torch.nn.Flatten()
        self.Linear = torch.nn.Linear(2592, D_out)
        self.relu_3 = torch.nn.ReLU()

    def create_vector(self, obs):
        """  Computes latent feature of given image
        Args:
            param1(numpy array): images of observation
        """
        obs_shape = obs.size()
        if_high_dim = (len(obs_shape) == 5)
        if if_high_dim:  # case of RNN input
            obs = obs.view(-1, *obs_shape[2:])
        x = self.conv_1(obs)
        x = self.relu_1(x)
        x = self.conv_2(x)
        x = self.relu_2(x)
        x = self.flatten(x)
        obs = self.relu_3(self.Linear(x))

        if if_high_dim:
            obs = obs.view(obs_shape[0], obs_shape[1], -1)
        return obs
