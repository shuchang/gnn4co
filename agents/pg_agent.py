import numpy as np
from policies.gat_policy import GATPolicy
from agents.base_agent import BaseAgent


class PGAgent(BaseAgent):

    def __init__(self, config):

        BaseAgent.__init__(self, config)

        self.gamma = self.hparams["discount_rate"]
        self.reward_to_go = self.hparams["reward_to_go"]
        self.actor = GATPolicy(self.hparams)


    def sample_from_replay_buffer(self, batch_size):
        """Draws recent data samples from the replay buffer"""
        return self.replay_buffer.sample_recent_data(batch_size, return_full_trajectory=True)


    def train(self, obs, acs, rews, next_obs, dones):
        """Trains the policy gradient agent\n
            params:
                obs: list\n
                acs: np.ndarray\n
                rews: list\n
                next_obs: list\n
                dones: np.ndarray\n
            returns:
                train_log: dict
        """
        q_values = self.calculate_q_values(rews)
        train_log = self.actor.update(obs, acs, q_values)
        return train_log


    def calculate_q_values(self, rews):
        """Monte Carlo estimation of the Q function"""
        if not self.reward_to_go:
            discounted_returns = [self._discounted_return(r) for r in rews]
            q_values = np.concatenate(discounted_returns).astype('float32')
        else:
            discounted_cumsums = [self._discounted_cumsum(r) for r in rews]
            q_values = np.concatenate(discounted_cumsums).astype('float32')

        return q_values


    def save(self, path):
        self.actor.save(path)


    def _discounted_return(self, reward):
        """Calculates the discounted returns for a trajectory\n
            Input:
                np.ndarray reward {r_0, r_1, ..., r_T} for a trajectory of len T\n
            Output:
                np.ndarray where each index t contains sum_{t'=0}^T gamma^{t'} r_{t'}
        """
        traj_len = reward.shape[0]
        discount = self.gamma ** np.arange(traj_len)
        discounted_reward = np.multiply(discount, reward)
        discounted_return = np.ones(traj_len) * np.sum(discounted_reward)
        return discounted_return


    def _discounted_cumsum(self, reward):
        """"""
        traj_len = reward.shape[0]
        discounted_cumsum = np.zeros(traj_len)

        for t in range(traj_len):
            discount = self.gamma ** (np.arange(t, traj_len) - t)
            discounted_reward_to_go = np.multiply(discount, reward[t:])
            discounted_cumsum[t] = np.sum(discounted_reward_to_go)

        return discounted_cumsum