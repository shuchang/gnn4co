from agents.base_agent import BaseAgent
from policies.gat_policy import GATCritic


class DQNAgent(BaseAgent):

    def __init__(self, config):

        BaseAgent.__init__(self, config)

        self.actor = GATCritic(self.hyperparameters)


    def sample_from_replay_buffer(self, batch_size):
        """Draws random data samples from the replay buffer"""
        return self.replay_buffer.sample_random_data(batch_size)


    def train(self, obs, acs, rews, next_obs, dones):
        """Trains the DQN agent\n
            params:
                obs: list\n
                acs: np.ndarray\n
                rews: np.ndarray\n
                next_obs: list\n
                dones: np.ndarray\n
            returns:
                train_log: dict
        """
        train_log = self.actor.update(obs, acs, rews, next_obs, dones)
        return train_log


    def save(self, path):
        self.actor.save(path)