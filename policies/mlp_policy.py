import numpy as np
import torch

from torch import nn, optim, distributions
from policies.base_policy import BasePolicy
from infrastructure import pytorch_utils as ptu


class MLPPolicy(BasePolicy, nn.Module):

    def __init__(self, ob_dim, ac_dim, n_hidden_layers, hidden_size,
                 learning_rate, **kwargs):

        super().__init__(**kwargs)

        self.ob_dim = ob_dim
        self.ac_dim = ac_dim
        self.n_hidden_layers = n_hidden_layers
        self.hidden_size = hidden_size
        self.learning_rate = learning_rate

        # discrete env
        self.policy = ptu.build_mlp(self.ob_dim, self.ac_dim, self.n_hidden_layers, self.hidden_size)
        self.policy.to(ptu.device)
        self.optimizer = optim.Adam(self.policy.parameters(), self.learning_rate)

        # self.baseline = ptu.build_mlp(self.ob_dim, 1, self.n_hidden_layers, self.hidden_size)
        # self.baseline.to(ptu.device)
        # self.baseline_optimizer = optim.Adam(self.baseline.parameters(), self.learning_rate)
        # self.baseline_loss = nn.MSELoss()


    def get_action(self, ob: np.ndarray, action_list: list) ->int:
        """Queries the policy with observation(s) to get selected action(s)"""
        if len(ob.shape) > 1:
            observation = ob[0] # separate state and adjacent matrix
        else:
            observation = ob[None]

        observation = ptu.from_numpy(observation.astype(np.float32))
        action_dist = self.forward(observation)
        # action = action_dist.sample()

        # get action from the action list
        mask = torch.zeros(action_dist.param_shape)
        mask[action_list] = 1
        action_probs = action_dist.probs
        action = torch.multinomial(action_probs*mask, 1)

        return ptu.to_numpy(action)[0]


    def forward(self, observation: torch.FloatTensor):
        logits = self.policy(observation)
        action_dist = distributions.Categorical(logits=logits)

        return action_dist


    def update(self, obs: np.ndarray, acs: np.ndarray, rewards: np.ndarray, **kwargs):
        obs = obs[:,0,:]
        observations = ptu.from_numpy(obs)

        actions = ptu.from_numpy(acs)
        rewards = ptu.from_numpy(rewards)

        action_dist = self.forward(observations)
        loss = -torch.mul(action_dist.log_prob(actions), rewards).mean()

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        train_log = {"Training Loss": ptu.to_numpy(loss)}

        return train_log


    def save(self, filepath: str):
        torch.save(self.state_dict(), filepath)