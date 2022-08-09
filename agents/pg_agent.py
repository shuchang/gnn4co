import numpy as np
from policies.gat_policy import GATPolicy
from agents.base_agent import BaseAgent
from infrastructure.utils import normalize, unnormalize


class PGAgent(BaseAgent):

    def __init__(self, config):

        BaseAgent.__init__(self, config)

        self.gamma = self.hparams["discount_rate"]
        self.reward_to_go = self.hparams["reward_to_go"]
        self.nn_baseline = self.hparams["nn_baseline"]
        self.standardize_advantages = self.hparams["standardize_advantages"]
        self.gae_lambda = self.hparams["gae_lambda"]

        self.actor = GATPolicy(self.hparams)


    def sample_from_replay_buffer(self, batch_size):
        """Draws recent data samples from the replay buffer"""
        return self.replay_buffer.sample_recent_data(batch_size, return_full_traj=True)


    def train(self, obs, acs, rews, next_obs, dones):
        """Trains the policy gradient agent with estimated advantages\n
            params:
                obs: list of length (batch_size)\n
                acs: np.ndarray of shape (batch_size, )\n
                rews (unconcatenated): list of length (n_batch)\n
                next_obs: list of length (batch_size)\n
                dones: np.ndarray of shape (batch_size, )\n
            returns:
                train_log: dict
        """
        q_values = self.calculate_q_values(rews)
        advantages = self.estimate_advantage(obs, rews, q_values, dones)
        train_log = self.actor.update(obs, acs, advantages, q_values)
        return train_log


    def calculate_q_values(self, rews):
        """Runs the Monte Carlo estimation of the Q function\n
            param:
                rews (unconcatenated): list of length (n_batch)\n
            return:
                q_values: np.ndarray with shape (batch_size, )
        """
        # concatenate rewards with discounts
        if not self.reward_to_go:
            discounted_returns = [self._discounted_return(r) for r in rews]
            q_values = np.concatenate(discounted_returns).astype('float32')
        else:
            discounted_cumsums = [self._discounted_cumsum(r) for r in rews]
            q_values = np.concatenate(discounted_cumsums).astype('float32')

        return q_values


    def estimate_advantage(self, obs, rews, q_values, dones):
        """Computes advantages using GAE or the Q function minus the value function"""
        if self.nn_baseline:
            # estimate the value function with nn_baseline
            values = self.actor.get_baseline_prediction(obs)
            assert values.ndim == q_values.ndim
            values_normalized = normalize(values, values.mean(), values.std())
            values = unnormalize(values_normalized, q_values.mean(), q_values.std())

            if self.gae_lambda is None:
                advantages = q_values - values
            else:
                values = np.append(values, [0])
                rews = np.concatenate(rews)
                batch_size = len(obs)
                advantages = np.zeros(batch_size + 1)

                for i in reversed(range(batch_size)):
                    if dones[i] == 1:
                        delta = rews[i] - values[i]
                    else:
                        delta = rews[i] + self.gamma*values[i+1] - values[i]

                    advantages[i] = delta + self.gamma*self.gae_lambda*advantages[i+1]

                advantages = advantages[:-1] # remove dummy advantage

        else:
            advantages = q_values.copy()

        if self.standardize_advantages:
            advantages = normalize(advantages, advantages.mean(), advantages.std())

        return advantages


    def save(self, path):
        self.actor.save(path)


    def _discounted_return(self, reward):
        """Calculates the discounted returns for a trajectory\n
            param:
                reward: np.ndarray reward {r_0, r_1, ..., r_T}
                for a trajectory of len T\n
            return:
                discounted_return: np.ndarray where each index t
                contains sum_{t'=0}^T gamma^{t'} r_{t'}
        """
        traj_len = reward.shape[0]
        discount = self.gamma ** np.arange(traj_len)
        discounted_reward = np.multiply(discount, reward)
        discounted_return = np.ones(traj_len) * np.sum(discounted_reward)
        return discounted_return


    def _discounted_cumsum(self, reward):
        """Calculates the discounted cummulated sum for a trajectory\n
            param:
                reward: np.ndarray reward {r_0, r_1, ..., r_T}
                for a trajectory of len T\n
            return:
                discounted_cumsum: np.ndarray where each index t
                contains sum_{t'=t}^T gamma^(t'-t) * r_{t'}
        """
        traj_len = reward.shape[0]
        discounted_cumsum = np.zeros(traj_len)

        for t in range(traj_len):
            discount = self.gamma ** (np.arange(t, traj_len) - t)
            discounted_reward_to_go = np.multiply(discount, reward[t:])
            discounted_cumsum[t] = np.sum(discounted_reward_to_go)

        return discounted_cumsum