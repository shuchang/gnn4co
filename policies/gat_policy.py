import torch

from torch import nn, optim, distributions
from torch_geometric.data.batch import Batch

from policies.base_policy import BasePolicy
from infrastructure import pytorch_utils as ptu
from infrastructure.pytorch_utils import GAT


class GATPolicy(BasePolicy, nn.Module):

    def __init__(self, hyperparameters):

        super().__init__()

        self.ob_dim = hyperparameters["ob_dim"]
        self.ac_dim = hyperparameters["ac_dim"]
        self.n_layers = hyperparameters["n_layers"]
        self.hidden_size = hyperparameters["hidden_size"]
        self.learning_rate = hyperparameters["learning_rate"]

        self.network = GAT(self.ob_dim, self.n_layers, self.hidden_size, self.ac_dim)
        self.network.to(ptu.device)
        self.optimizer = optim.Adam(self.network.parameters(), self.learning_rate)

        # self.baseline = ptu.build_mlp(self.ob_dim, 1, self.n_hidden_layers, self.hidden_size)
        # self.baseline.to(ptu.device)
        # self.baseline_optimizer = optim.Adam(self.baseline.parameters(), self.learning_rate)
        # self.baseline_loss = nn.MSELoss()


    def forward(self, obs, batch_size=1):
        """Returns the distributions of batches of actions
            params:
                obs.x: shape (n_nodes, ob_dim)\n
                obs.edge_index: shape (2, n_edges)\n
            returns:
                action_dist: shape (batch_size*n_nodes, ac_dim)->(batch_size, n_nodes)
        """
        logits = self.network(obs.x, obs.edge_index).view(batch_size, -1)
        action_dist = distributions.Categorical(logits=logits)
        return action_dist


    @torch.no_grad()
    def get_action(self, obs, action_list):
        """Queries the policy with observation(s) to get selected action(s)"""
        action_dist = self.forward(obs)
        batch_size = action_dist.batch_shape[0]

        if batch_size == 1:
            mask = ptu.to_device(torch.zeros(action_dist.param_shape))
            mask[0, action_list] = 1
            allowed_action_probs = action_dist.probs.masked_fill(mask==0, 0)
            actions = torch.multinomial(allowed_action_probs.squeeze(), num_samples=1)
            return ptu.to_numpy(actions)[0]
        else:
            raise NotImplementedError


    def update(self, obs, actions, rewards):
        """Runs a learning iteration for the policy"""
        assert len(obs) == actions.shape[0]
        batch_size = len(obs)
        loader = Batch.from_data_list(obs)
        actions = ptu.from_numpy(actions)
        rewards = ptu.from_numpy(rewards)

        action_dist = self.forward(loader, batch_size)
        loss = -torch.mul(action_dist.log_prob(actions), rewards).mean()

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return {"Training Loss": ptu.to_numpy(loss)}


    def save(self, filepath):
        torch.save(self.network.state_dict(), filepath)


class GATCritic(BasePolicy, nn.Module):

    def __init__(self, hyperparameters):

        super().__init__()

        self.ob_dim = hyperparameters["ob_dim"]
        self.ac_dim = hyperparameters["ac_dim"]
        self.n_layers = hyperparameters["n_layers"]
        self.hidden_size = hyperparameters["hidden_size"]
        self.learning_rate = hyperparameters["learning_rate"]

        self.double_q = hyperparameters["double_q"]
        self.grad_norm_clipping = hyperparameters["grad_norm_clipping"]
        self.gamma = hyperparameters["discount_rate"]

        self.network = GAT(self.ob_dim, self.n_layers, self.hidden_size, self.ac_dim)
        self.network.to(ptu.device)
        self.target_network = GAT(self.ob_dim, self.n_layers, self.hidden_size, self.ac_dim)
        self.target_network.to(ptu.device)

        self.optimizer = optim.Adam(self.network.parameters(), self.learning_rate)
        self.criterion = nn.SmoothL1Loss()


    def forward(self, obs, batch_size=1):
        """Returns the state values V(s_t)\n
            params:
                obs.x: shape (n_nodes, ob_dim)\n
                obs.edge_index: shape (2, n_edges)\n
            returns:
                state_values: shape (batch_size*n_nodes, ac_dim)->(batch_size, n_nodes)
        """
        state_values = self.network(obs.x, obs.edge_index).view(batch_size, -1)
        return state_values


    @torch.no_grad()
    def get_action(self, obs, action_list):
        """Queries the policy with observation(s) to get selected action(s)"""
        state_values = self.forward(obs)
        batch_size = state_values.shape[0]

        if batch_size == 1:
            mask = ptu.to_device(torch.zeros(state_values.shape))
            mask[0, action_list] = 1
            allowed_state_values = state_values.masked_fill(mask==0, -1e4)
        else:
            raise NotImplementedError

        if state_values.dim() == 1: # TODO: to be improved
            actions = allowed_state_values.argmax()
        else:
            actions = allowed_state_values.argmax(1, keepdim=True).squeeze(0)

        return ptu.to_numpy(actions)[0]


    def update(self, obs, actions, rewards, next_obs, dones):
        """Runs a learning iteration for the critic"""
        batch_size = len(obs)
        loader = Batch.from_data_list(obs)
        actions = ptu.from_numpy(actions).to(torch.long)
        loader_target = Batch.from_data_list(next_obs)
        rewards = ptu.from_numpy(rewards)
        dones = ptu.from_numpy(dones)

        q_values = self.forward(loader, batch_size).gather(1, actions.unsqueeze(1)).squeeze(1)
        # calculate the state-action values Q(s_t, a)

        # calculate the next state values V(s_{t+1})
        state_values_target = self.target_network(loader_target.x, loader_target.edge_index).view(batch_size, -1)

        if self.double_q:
            state_values = self.forward(loader, batch_size)
            greedy_actions = state_values.argmax(1, keepdim=True)
            q_values_target = state_values_target.gather(1, greedy_actions).squeeze(1)
        else:
            q_values_target = state_values_target.argmax(1, True).squeeze(1)
        td_target = rewards + self.gamma * q_values_target * (1 - dones)
        td_target.detach() # no grad through target value

        loss = self.criterion(q_values, td_target) # TD error

        self.optimizer.zero_grad()
        loss.backward()

        if self.grad_norm_clipping is not None: # Optional gradient clipping
            nn.utils.clip_grad_value_(self.network.parameters(), self.grad_norm_clipping)

        self.optimizer.step()
        return {"Training Loss": ptu.to_numpy(loss)}


    def update_target_network(self):
        for target_param, param in zip(
                self.target_network.parameters(), self.network.parameters()
        ):
            target_param.data.copy_(param.data)


    def save(self, filepath):
        torch.save(self.network.state_dict(), filepath)