import numpy as np
from policies.gat_policy import GATPolicy
from agents.base_agent import BaseAgent


class PGAgent(BaseAgent):

    def __init__(self, config):

        BaseAgent.__init__(self, config)

        self.gamma = self.hparams["discount_rate"]
        self.actor = GATPolicy(self.hparams)


    def sample_from_replay_buffer(self, batch_size):
        """Draws recent data samples from the replay buffer"""
        return self.replay_buffer.sample_recent_data(batch_size, return_full_trajectory=True)


    def train(self, obs, acs, rews, next_obs, dones):
        """Trains the policy gradient agent\n
            params:
                obs: list\n
                acs: np.ndarray\n
                rews: np.ndarray\n
                next_obs: list\n
                dones: np.ndarray\n
            returns:
                train_log: dict
        """
        discounted_return = self._calculate_discounted_return(rews)
        train_log = self.actor.update(obs, acs, discounted_return)
        return train_log


    def save(self, path):
        self.actor.save(path)


    def _calculate_discounted_return(self, rewards: np.ndarray) -> np.ndarray:
        """Calculates the discounted returns for list of trajectories\n
            Input:
                np.ndarray of rewards {r_0, r_1, ..., r_T} for a single trajectory of len T\n
            Output:
                np.ndarray where each index t contains sum_{t'=0}^T gamma^{t'} r_{t'}
        """
        discounts = self.gamma ** np.arange(rewards.shape[0])
        discounted_return = np.multiply(discounts, rewards)
        return discounted_return