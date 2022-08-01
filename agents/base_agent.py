from infrastructure.replay_buffer import ReplayBuffer

class BaseAgent(object):

    def __init__(self, config):

        self.config = config
        self.hparams = config.hparams

        self.replay_buffer = ReplayBuffer(
            buffer_size=self.hparams["buffer_size"],
            batch_size=self.hparams["batch_size"])


    def add_to_replay_buffer(self, trajectories):
        """Adds collected experiences to the replay buffer"""
        self.replay_buffer.add_trajectories(trajectories)

    def sample_from_replay_buffer(self, batch_size):
        """Draws samples of trajectories from the replay buffer"""
        raise NotImplementedError

    def train(self, obs, acs, rews, next_obs, dones) -> dict:
        """Return a dictionary of logging information"""
        raise NotImplementedError

    def save(self, path):
        raise NotImplementedError