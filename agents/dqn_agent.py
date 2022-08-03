from agents.base_agent import BaseAgent
from policies.gat_policy import GATCritic


class DQNAgent(BaseAgent):

    def __init__(self, config):

        BaseAgent.__init__(self, config)

        self.actor = GATCritic(self.hparams)
        self.learning_starts = self.hparams["learning_starts"]
        self.update_freq = self.hparams["update_freq"]
        self.target_update_freq = self.hparams["target_update_freq"]
        self.train_steps = 0
        self.n_param_updates = 0


    def sample_from_replay_buffer(self, batch_size):
        """Draws random data samples from the replay buffer"""
        return self.replay_buffer.sample_random_data(batch_size)


    def train(self, obs, acs, rews, next_obs, dones):
        """Trains the DQN agent\n
            params:
                obs: list\n
                acs: np.ndarray\n
                rews: list\n
                next_obs: list\n
                dones: np.ndarray\n
            returns:
                train_log: dict
        """
        train_log = {}

        if (self.train_steps > self.learning_starts and
            self.train_steps % self.update_freq == 0):
            train_log = self.actor.update(obs, acs, rews, next_obs, dones)

            if self.n_param_updates % self.target_update_freq == 0:
                self.actor.update_target_network()

            self.n_param_updates += 1

        self.train_steps += 1

        return train_log


    def save(self, path):
        self.actor.save(path)