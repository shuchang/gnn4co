import numpy as np


class ReplayBuffer(object):
    """Replay buffer to store past trajectories"""

    def __init__(self, buffer_size, batch_size):

        self.buffer_size = buffer_size
        self.batch_size = batch_size

        # store each trajectories and its component arrays
        self.memory = []
        self.obs = None
        self.acs = None
        self.rews = None
        self.next_obs = None
        self.dones = None


    def add_trajectories(self, trajectories: list):
        """Adds trajectories and their component arrays into the replay buffer"""
        self.memory.extend(trajectories)
        obs, acs, rews, next_obs, dones = _trajectories_to_data(trajectories)

        if self.obs is None:
            self.obs = obs[-self.buffer_size:]
            self.acs = acs[-self.buffer_size:]
            self.rews = rews[-self.buffer_size:]
            self.next_obs = next_obs[-self.buffer_size:]
            self.dones = dones[-self.buffer_size:]
        else:
            self.obs.extend(obs)
            self.obs = self.obs[-self.buffer_size:]
            self.acs = np.concatenate([self.acs, acs])[-self.buffer_size:]
            self.rews = np.concatenate([self.rews, rews])[-self.buffer_size:]
            self.next_obs.extend(next_obs)
            self.next_obs = self.next_obs[-self.buffer_size:]
            self.dones = np.concatenate([self.dones, dones])[-self.buffer_size:]


    def sample_random_data(self, batch_size):
        """Draws a random sample from the replay buffer, for Value-based agents"""
        indices = np.random.permutation(self.acs.shape[0])[:batch_size]
        return (self.obs[indices],
                self.acs[indices],
                self.rews[indices],
                self.next_obs[indices],
                self.dones[indices])


    def sample_recent_data(self, batch_size, return_full_trajectory=True):
        """Draws a recent sample from the replay buffer
           Policy-based agents: data are from full episodes
        """
        if return_full_trajectory:
            n_recent_trajectories = 0
            index = -1
            n_data_so_far = 0

            while n_data_so_far < batch_size:
                recent_trajectory = self.memory[index]
                # n_data_so_far += len(recent_trajectory["reward"])
                n_data_so_far += recent_trajectory["reward"].shape[0]
                n_recent_trajectories += 1
                index -= 1

            trajectories = self.memory[-n_recent_trajectories:]
            obs, acs, rews, next_obs, dones = _trajectories_to_data(trajectories)
            return obs, acs, rews, next_obs, dones
        else:
            return (self.obs[-batch_size:],
                    self.acs[-batch_size:],
                    self.rews[-batch_size:],
                    self.next_obs[-batch_size:],
                    self.dones[-batch_size:])

    #####################################################
    #####################################################

    def sample_random_trajectories(self, n_trajectories):
        indices = np.random.permutation(len(self.memory))[:n_trajectories]
        return self.memory[indices]

    def sample_recent_trajectories(self, n_trajectories):
        return self.memory[-n_trajectories:]

    #####################################################
    ################## HELPER FUNCTIONS #################
    #####################################################

def _trajectories_to_data(trajectories: list):
    """Convert lists of trajectories to the training data\n
        param:
            trajectories: list of trajectories\n
        returns:
            obs: list\n
            acs: np.ndarray\n
            rews: np.ndarray\n
            next_obs: list\n
            dones: np.ndarray
    """
    obs, next_obs = [], []
    for t in trajectories:
        obs.extend(t["observation"])
        next_obs.extend(t["next_observation"])
    acs = np.concatenate([t["action"] for t in trajectories])
    rews = np.concatenate([t["reward"] for t in trajectories])
    dones = np.concatenate([t["done"] for t in trajectories])
    return obs, acs, rews, next_obs, dones