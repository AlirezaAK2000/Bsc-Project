import os
import torch as T
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.normal import Normal
import numpy as np
from torch.distributions.categorical import Categorical
from torch.distributions import MultivariateNormal
import torch.nn.functional as F


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class ActorCriticPPOAtari(nn.Module):
    def __init__(self, action_dim, has_continuous_action_space, action_std_init, device, num_frames=4):
        super(ActorCriticPPOAtari, self).__init__()

        self.has_continuous_action_space = has_continuous_action_space
        self.device = device
        if has_continuous_action_space:
            self.action_dim = action_dim
            self.action_var = torch.full((action_dim,), action_std_init * action_std_init).to(self.device)
        # actor
        if has_continuous_action_space:
            self.actor = nn.Sequential(
                layer_init(nn.Conv2d(num_frames, 32, 8, stride=4)),
                nn.Tanh(),
                layer_init(nn.Conv2d(32, 64, 4, stride=2)),
                nn.Tanh(),
                layer_init(nn.Conv2d(64, 64, 3, stride=1)),
                nn.Tanh(),
                nn.Flatten(),
                layer_init(nn.Linear(3136, 512)),
                nn.Tanh(),
                layer_init(nn.Linear(512, action_dim)),
            )
        else:
            self.actor = nn.Sequential(
                layer_init(nn.Conv2d(num_frames, 32, 8, stride=4)),
                nn.Tanh(),
                layer_init(nn.Conv2d(32, 64, 4, stride=2)),
                nn.Tanh(),
                layer_init(nn.Conv2d(64, 64, 3, stride=1)),
                nn.Tanh(),
                nn.Flatten(),
                layer_init(nn.Linear(3136, 512)),
                nn.Tanh(),
                layer_init(nn.Linear(512, action_dim)),
                nn.Softmax(dim=-1)
            )
        # critic
        self.critic = nn.Sequential(
            layer_init(nn.Conv2d(num_frames, 32, 8, stride=4)),
            nn.Tanh(),
            layer_init(nn.Conv2d(32, 64, 4, stride=2)),
            nn.Tanh(),
            layer_init(nn.Conv2d(64, 64, 3, stride=1)),
            nn.Tanh(),
            nn.Flatten(),
            layer_init(nn.Linear(3136, 512)),
            nn.Tanh(),
            layer_init(nn.Linear(512, 1), std=1)
        )

    def set_action_std(self, new_action_std):
        if self.has_continuous_action_space:
            self.action_var = torch.full((self.action_dim,), new_action_std * new_action_std).to(self.device)
        else:
            print("--------------------------------------------------------------------------------------------")
            print("WARNING : Calling ActorCriticPPO::set_action_std() on discrete action space policy")
            print("--------------------------------------------------------------------------------------------")

    def forward(self):
        raise NotImplementedError

    def act(self, state):

        if self.has_continuous_action_space:
            action_mean = self.actor(state)
            cov_mat = torch.diag(self.action_var).unsqueeze(dim=0)
            dist = MultivariateNormal(action_mean, cov_mat)
        else:
            action_probs = self.actor(state)
            dist = Categorical(action_probs)

        action = dist.sample()
        action_logprob = dist.log_prob(action)
        state_val = self.critic(state)

        return action.detach(), action_logprob.detach(), state_val.detach()

    def evaluate(self, state, action):

        if self.has_continuous_action_space:
            action_mean = self.actor(state)

            action_var = self.action_var.expand_as(action_mean)
            cov_mat = torch.diag_embed(action_var).to(self.device)
            dist = MultivariateNormal(action_mean, cov_mat)

            # For Single Action Environments.
            if self.action_dim == 1:
                action = action.reshape(-1, self.action_dim)
        else:
            action_probs = self.actor(state)
            dist = Categorical(action_probs)
        action_logprobs = dist.log_prob(action)
        dist_entropy = dist.entropy()
        state_values = self.critic(state)
        return action_logprobs, state_values, dist_entropy


class DeepQNetworkAtari(nn.Module):
    def __init__(self, n_action,
                 name='q_network', chkpt_dir='tmp/dqn',
                 num_frames=4):
        super(DeepQNetworkAtari, self).__init__()
        self.n_action = n_action

        self.network = nn.Sequential(
            nn.Conv2d(num_frames, 32, 8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(3136, 512),
            nn.ReLU(),
            nn.Linear(512, n_action),
        )

        self.name = name
        self.checkpoint_dir = chkpt_dir
        self.checkpoint_file = os.path.join(self.checkpoint_dir, name + '_dqn')

    def forward(self, state):
        output = self.network(state)

        return output

    def save_checkpoint(self):
        T.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        self.load_state_dict(T.load(self.checkpoint_file))


# SAC
def sac_layer_init(layer, bias_const=0.0):
    nn.init.kaiming_normal_(layer.weight)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class SACActorAtari(nn.Module):
    """Actor (Policy) Model."""

    def __init__(self, action_dim, num_frames=4):
        super(SACActorAtari, self).__init__()

        self.network = nn.Sequential(
            sac_layer_init(nn.Conv2d(num_frames, 32, 8, stride=4)),
            nn.ReLU(),
            sac_layer_init(nn.Conv2d(32, 64, 4, stride=2)),
            nn.ReLU(),
            sac_layer_init(nn.Conv2d(64, 64, 3, stride=1)),
            nn.ReLU(),
            nn.Flatten(),
            sac_layer_init(nn.Linear(3136, 512)),
            nn.ReLU(),
            sac_layer_init(nn.Linear(512, action_dim)),
            nn.Softmax(dim=-1)
        )

    def forward(self, state):
        action_probs = self.network(state)
        return action_probs

    def evaluate(self, state, epsilon=1e-6):
        action_probs = self.forward(state)

        dist = Categorical(action_probs)
        action = dist.sample().to(state.device)
        # Have to deal with situation of 0.0 probabilities because we can't do log 0
        z = action_probs == 0.0
        z = z.float() * 1e-8
        log_action_probabilities = torch.log(action_probs + z)
        return action.detach().cpu(), action_probs, log_action_probabilities

    def get_action(self, state):
        """
        returns the action based on a squashed gaussian policy. That means the samples are obtained according to:
        a(s,e)= tanh(mu(s)+sigma(s)+e)
        """
        action_probs = self.forward(state)

        dist = Categorical(action_probs)
        action = dist.sample().to(state.device)
        # Have to deal with situation of 0.0 probabilities because we can't do log 0
        z = action_probs == 0.0
        z = z.float() * 1e-8
        log_action_probabilities = torch.log(action_probs + z)
        return action.detach().cpu(), action_probs, log_action_probabilities

    def get_det_action(self, state):
        action_probs = self.forward(state)
        dist = Categorical(action_probs)
        action = dist.sample().to(state.device)
        return action.detach().cpu()


class SACCriticAtari(nn.Module):
    """Critic (Value) Model."""

    def __init__(self, action_dim, num_frames=4, seed=1):
        super(SACCriticAtari, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.network = nn.Sequential(
            sac_layer_init(nn.Conv2d(num_frames, 32, 8, stride=4)),
            nn.ReLU(),
            sac_layer_init(nn.Conv2d(32, 64, 4, stride=2)),
            nn.ReLU(),
            sac_layer_init(nn.Conv2d(64, 64, 3, stride=1)),
            nn.ReLU(),
            nn.Flatten(),
            sac_layer_init(nn.Linear(3136, 512)),
            nn.ReLU(),
            sac_layer_init(nn.Linear(512, action_dim)),
        )

    def forward(self, state):
        """Build a critic (value) network that maps (state, action) pairs -> Q-values."""
        return self.network(state)
