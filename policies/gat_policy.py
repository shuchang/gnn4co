import numpy as np
import torch

from torch import nn, optim, distributions
from torch_geometric.data.batch import Batch

from policies.base_policy import BasePolicy
from infrastructure import pytorch_utils as ptu
from infrastructure.pytorch_utils import GAT


class GATPolicy(BasePolicy, nn.Module):

    def __init__(self, ob_dim, ac_dim, n_hidden_layers, hidden_size,
                 learning_rate, **kwargs):

        super().__init__(**kwargs)

        self.ob_dim = ob_dim
        self.ac_dim = ac_dim
        # self.n_hidden_layers = n_hidden_layers
        self.hidden_size = hidden_size
        self.learning_rate = learning_rate

        # discrete env
        self.policy = GAT(self.ob_dim, self.hidden_size, self.ac_dim, heads=1)
        self.policy.to(ptu.device)
        self.optimizer = optim.Adam(self.policy.parameters(), self.learning_rate)

        # self.baseline = ptu.build_mlp(self.ob_dim, 1, self.n_hidden_layers, self.hidden_size)
        # self.baseline.to(ptu.device)
        # self.baseline_optimizer = optim.Adam(self.baseline.parameters(), self.learning_rate)
        # self.baseline_loss = nn.MSELoss()


    def forward(self, observation, edge_index, batch_size=1):
        logits = self.policy(observation, edge_index).view(batch_size, -1)
        action_dist = distributions.Categorical(logits=logits)
        return action_dist


    def get_action(self, ob, action_list):
        """Queries the policy with observation(s) to get selected action(s)"""
        action_dist = self.forward(ob.x, ob.edge_index)

        # get action from the action list
        mask = torch.zeros(action_dist.param_shape)
        mask[0, action_list] = 1
        action_probs = action_dist.probs * mask
        # TODO: fix the over-fitting problem
        action = torch.multinomial(action_probs.squeeze(), num_samples=1)
        return ptu.to_numpy(action)[0]


    def update(self, obs, actions, rewards, **kwargs):
        """updates the weights of the policy\n
            params:
                obs: list\n
                acs: np.ndarray\n
                rews: np.ndarray\n
            returns:
                train_log: dict
        """
        batch_size = len(obs)
        loader = Batch.from_data_list(obs)
        actions = ptu.from_numpy(actions)
        rewards = ptu.from_numpy(rewards)

        action_dist = self.forward(loader.x, loader.edge_index, batch_size)
        loss = -torch.mul(action_dist.log_prob(actions), rewards).mean()

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return {"Training Loss": ptu.to_numpy(loss)}


    def save(self, filepath: str):
        torch.save(self.state_dict(), filepath)