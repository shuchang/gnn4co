import torch

from torch import nn, optim, distributions
from torch_geometric.data.batch import Batch

from policies.base_policy import BasePolicy
from infrastructure import pytorch_utils as ptu
from infrastructure.pytorch_utils import GAT


class GATPolicy(BasePolicy, nn.Module):

    def __init__(self, hparams):

        super().__init__()

        self.n_nodes = hparams["n_nodes"]
        self.ob_dim = hparams["ob_dim"]
        self.ac_dim = hparams["ac_dim"]
        self.n_layers = hparams["n_layers"]
        self.hidden_size = hparams["hidden_size"]
        self.learning_rate = hparams["learning_rate"]

        self.nn_baseline = hparams["nn_baseline"]

        self.network = GAT(self.ob_dim, self.n_layers, self.hidden_size, self.ac_dim)
        self.network.to(ptu.device)
        self.optimizer = optim.Adam(self.network.parameters(), self.learning_rate)

        if self.nn_baseline:
            # self.baseline = GAT(self.ob_dim, self.n_layers, self.hidden_size, self.ac_dim)
            self.baseline = ptu.build_mlp(self.n_nodes, 1, self.n_layers, self.hidden_size)
            self.baseline.to(ptu.device)
            self.baseline_optimizer = optim.Adam(self.baseline.parameters(), self.learning_rate)
            self.baseline_loss = nn.MSELoss()


    def forward(self, obs, batch_size=1):
        """Returns the distributions of batches of actions\n
            params:
                obs.x: tensor of shape (n_nodes, ob_dim)\n
                obs.edge_index: tensor of shape (2, n_edges)\n
            return:
                action_dist.logits: tensor of shape
                (batch_size*n_nodes, ac_dim)->(batch_size, n_nodes)
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
            mask = torch.zeros(action_dist.param_shape).to(ptu.device)
            mask[0, action_list] = 1
            allowed_action_probs = action_dist.probs.masked_fill(mask==0, 0)
            actions = torch.multinomial(allowed_action_probs.squeeze(), num_samples=1)
            return ptu.to_numpy(actions)[0]
        else:
            raise NotImplementedError


    def update(self, obs, actions, advantages, q_values):
        """Runs a learning iteration for the policy\n
            params:
                obs: list of length (batch_size)\n
                actions: np.ndarray of shape (batch_size, )\n
                advantages: np.ndarray of shape (batch_size, )\n
                q_values: np.ndarray of shape (batch_size,)\n
            return:
                train_log: dict
        """
        assert len(obs) == actions.shape[0]
        batch_size = len(obs)
        loader = Batch.from_data_list(obs)
        actions = ptu.from_numpy(actions)
        advantages = ptu.from_numpy(advantages)

        action_dist = self.forward(loader, batch_size)
        loss = -torch.mul(action_dist.log_prob(actions), advantages).mean()

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        train_log = {"loss": ptu.to_numpy(loss)}

        if self.nn_baseline:
            q_values_normalized = (q_values - q_values.mean())/q_values.std()
            q_values = ptu.from_numpy(q_values_normalized)

            observations = torch.cat([ob.x for ob in obs]).view(batch_size, -1)
            pred = self.baseline(observations)
            loss_baseline = self.baseline_loss(pred.squeeze(), q_values).sum()

            self.baseline_optimizer.zero_grad()
            loss_baseline.backward()
            self.baseline_optimizer.step()
            train_log.update({"loss_baseline": ptu.to_numpy(loss_baseline)})

        return train_log


    @torch.no_grad()
    def get_baseline_prediction(self, obs):
        """Runs the forward method of baseline to get the predicted value function\n
            param:
                obs: list of length (batch_size)
            return:
                pred: np.ndarray of shape (batch_size, )
        """
        batch_size = len(obs)
        observations = torch.cat([ob.x for ob in obs]).view(batch_size, -1)
        pred = self.baseline(observations)
        return ptu.to_numpy(pred.squeeze())


    def save(self, filepath):
        torch.save(self.network.state_dict(), filepath)


class GATCritic(BasePolicy, nn.Module):

    def __init__(self, hparams):

        super().__init__()

        self.ob_dim = hparams["ob_dim"]
        self.ac_dim = hparams["ac_dim"]
        self.n_layers = hparams["n_layers"]
        self.hidden_size = hparams["hidden_size"]
        self.learning_rate = hparams["learning_rate"]

        self.double_q = hparams["double_q"]
        self.grad_norm_clipping = hparams["grad_norm_clipping"]
        self.gamma = hparams["discount_rate"]

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
            mask = torch.zeros(state_values.shape).to(ptu.device)
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
        return {"loss": ptu.to_numpy(loss)}


    def update_target_network(self):
        for target_param, param in zip(
                self.target_network.parameters(), self.network.parameters()
        ):
            target_param.data.copy_(param.data)


    def save(self, filepath):
        torch.save(self.network.state_dict(), filepath)