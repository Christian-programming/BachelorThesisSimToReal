import os
import sys
import torch
import copy
import numpy as np
from tqc_models import Actor, Critic, Encoder, quantile_huber_loss_f
import torch.nn as nn
import torch.nn.functional as F
from helper import mkdir


# Building the whole Training Process into a class

class TQC(object):
    """  Implements the algorithm "Controlling Overestimation Bias with Truncated Mixture of Continuous
         Distributional Quantile Critics " combined with "Images Augmentations is all you need"
    """
    def __init__(self, state_dim, action_dim, actor_input_dim, config):
        """   set parameter from json file
        Args:
            param1(int): state_dim    vector size of output of encoder
            param2(int): action_dim
            param3(int): actor_input_dim
            param4(config): param from json file

        """
        input_dim = [config["history_length"], config["size"], config["size"]]
        self.actor = Actor(state_dim, action_dim, config).to(config["device"])
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), config["lr_actor"])
        self.critic = Critic(state_dim, action_dim, config).to(config["device"])
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), config["lr_critic"])
        self.target_critic = Critic(state_dim, action_dim, config).to(config["device"])
        self.target_critic.load_state_dict(self.critic.state_dict())
        self.encoder = Encoder(config).to(config["device"])
        self.encoder_optimizer = torch.optim.Adam(self.encoder.parameters(), config["lr_encoder"])
        self.target_encoder = Encoder(config).to(config["device"])
        self.target_encoder.load_state_dict(self.encoder.state_dict())
        self.batch_size = int(config["batch_size"])
        self.discount = config["discount"]
        self.tau = config["tau"]
        self.device = config["device"]
        self.write_tensorboard = False
        self.top_quantiles_to_drop = config["top_quantiles_to_drop_per_net"] * config["n_nets"] * 2
        self.target_entropy = config["target_entropy"]
        self.quantiles_total = self.critic.n_quantiles * self.critic.n_nets * 2
        self.log_alpha = torch.zeros((1,), requires_grad=True, device=config["device"])
        self.alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=config["lr_alpha"])
        self.total_it = 0
        self.step = 0

    def train(self, replay_buffer,  writer, iterations):
        """  updates the weights of Encoder, Critic and Actor

        Args:
            param1(replay_buffer): memory class from replay buffer
            param2(tensorboard):   writer save metric to tensorboard
            param3(int): amount of SGD updates of samples of batchsize
        """
        self.step += 1
        if self.step % 1000 == 0:
            self.write_tensorboard = 1 - self.write_tensorboard
        for it in range(iterations):
            # Step 4: We sample a batch of transitions (s, s’, a, r) from the memoy
            obs, action, reward, next_obs, not_done, obs_aug, obs_next_aug = replay_buffer.sample(self.batch_size)

            # for augment 1
            # compute the features from the augmented images
            obs = obs.div_(255)
            next_obs = next_obs.div_(255)
            state = self.encoder.create_vector(obs)
            detach_state = state.detach()
            next_state = self.target_encoder.create_vector(next_obs)

            # for augment 2
            obs_aug = obs_aug.div_(255)
            next_obs_aug = obs_next_aug.div_(255)
            state_aug = self.encoder.create_vector(obs_aug)
            detach_state_aug = state_aug.detach()
            next_state_aug = self.target_encoder.create_vector(next_obs_aug)

            alpha = torch.exp(self.log_alpha)
            with torch.no_grad():
                # Step 5: Get policy action
                new_next_action, next_log_pi = self.actor(next_state)

                # compute quantile at next state
                next_z = self.target_critic(next_state, new_next_action)

                # again for augment
                new_next_action_aug, next_log_pi_aug = self.actor(next_state_aug)
                next_z_aug = self.target_critic(next_state_aug, new_next_action_aug)
                next_z_all = torch.cat((next_z, next_z_aug), dim=1)
                sorted_z, _ = torch.sort(next_z_all.reshape(self.batch_size, -1))
                sorted_z_part = sorted_z[:, :self.quantiles_total - self.top_quantiles_to_drop]
                target = reward + not_done * self.discount * (sorted_z_part - alpha * next_log_pi)
            # -----------------update-critic-------------------------------------------------------
            cur_z = self.critic(state, action)

            # wasserstein metric to compute loss to be minimized
            critic_loss = quantile_huber_loss_f(cur_z, target, self.device)

            # for augment
            cur_z_aug = self.critic(state_aug, action)
            critic_loss += quantile_huber_loss_f(cur_z_aug, target, self.device)
            critic_loss * 0.5
            writer.add_scalar('Critic_loss', critic_loss, self.step)
            self.critic_optimizer.zero_grad()
            self.encoder_optimizer.zero_grad()
            critic_loss.backward()
            self.encoder_optimizer.step()
            self.critic_optimizer.step()

            for param, target_param in zip(self.critic.parameters(), self.target_critic.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

            for param, target_param in zip(self.encoder.parameters(), self.target_encoder.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

            # -------------------------Update-policy-and-alpha-------------------------------------------------------
            new_action, log_pi = self.actor(detach_state)
            alpha_loss = -self.log_alpha * (log_pi + self.target_entropy).detach().mean()
            actor_loss = (alpha * log_pi - self.critic(detach_state, new_action).mean(2).mean(1, keepdim=True)).mean()

            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            writer.add_scalar('Actor_loss', actor_loss, self.step)
            self.actor_optimizer.step()

            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            writer.add_scalar('Alpha_loss', actor_loss, self.step)
            self.alpha_optimizer.step()
            self.total_it += 1

    def select_action(self, obs):
        """  Computes action from actor
        Args:
            param1:(numpy array) obs single input
        """
        obs = torch.FloatTensor(obs).to(self.device)
        obs = obs.div_(255)
        state = self.encoder.create_vector(obs.unsqueeze(0))
        return self.actor.select_action(state)

    def quantile_huber_loss_f(self, quantiles, samples):
        """  Compute wasserstein loss proposed in the paper
        Args:
            param1:(numpy array) quantiles atoms of critc estimate
            param1:(numpy array) samples  atoms of the target estimate

            Return int
        """
        pairwise_delta = samples[:, None, None, :] - quantiles[:, :, :, None]  # batch x nets x quantiles x samples
        abs_pairwise_delta = torch.abs(pairwise_delta)
        huber_loss = torch.where(abs_pairwise_delta > 1, abs_pairwise_delta - 0.5, pairwise_delta ** 2 * 0.5)
        n_quantiles = quantiles.shape[2]
        tau = torch.arange(n_quantiles, device=self.device).float() / n_quantiles + 1 / 2 / n_quantiles
        loss = (torch.abs(tau[None, None, :, None] - (pairwise_delta < 0).float()) * huber_loss).mean()
        return loss

    def save(self, filename):
        """ Save weigths and optimizer parameter of the Encoder, Critic , Alpha and Actor
        Args:
            param1:(str) filename
        """
        mkdir("", filename)
        torch.save(self.critic.state_dict(), filename + "_critic")
        torch.save(self.critic_optimizer.state_dict(), filename + "_critic_optimizer")

        torch.save(self.actor.state_dict(), filename + "_actor")
        torch.save(self.actor_optimizer.state_dict(), filename + "_actor_optimizer")

        torch.save(self.encoder.state_dict(), filename + "_encoder")
        torch.save(self.encoder_optimizer.state_dict(), filename + "_encoder_optimizer")

        torch.save(self.log_alpha, filename + "_alpha")
        torch.save(self.alpha_optimizer.state_dict(), filename + "_alpha_optimizer")
        print("Saved weights and optimizer param to {}".format(filename))

    def load(self, filename):
        """ load weigths of the Encoder, Critic, Alpha and Actor
        Args:
            param1:(str) filename
        """
        self.critic.load_state_dict(torch.load(filename + "_critic"))
        self.critic_optimizer.load_state_dict(torch.load(filename + "_critic_optimizer"))
        self.critic_target = copy.deepcopy(self.critic)
        self.actor.load_state_dict(torch.load(filename + "_actor"))
        self.actor_optimizer.load_state_dict(torch.load(filename + "_actor_optimizer"))
        self.actor_target = copy.deepcopy(self.actor)

        self.encoder.load_state_dict(torch.load(filename + "_encoder"))
        self.encoder_optimizer.load_state_dict(torch.load(filename + "_encoder_optimizer"))
        self.log_alpha = torch.load(filename + "_alpha")
        self.alpha_optimizer.load_state_dict(torch.load(filename + "_alpha_optimizer"))
        print("Load weights and optimizer param to {}".format(filename))
