import numpy as np


class ReplayBuffer(object):
    """Replay buffer to store past trajectories that the agent can then use for training data"""

    def __init__(self, buffer_size, batch_size):

        self.buffer_size = buffer_size
        self.batch_size = batch_size

        # store each trajectories and its component arrays
        self.memory = []
        self.obs = None
        self.acs = None
        self.res = None
        self.next_obs = None
        self.dones = None

    def __len__(self):
        if self.observations: return self.observations.shape[0]
        else: return 0


    def add_trajectories(self, trajectories: list):
        """Adds new trajectories and their component arrays into the replay buffer"""
        self.memory.extend(trajectories)
        obs, acs, res, next_obs, dones = _trajectories_to_data(trajectories)

        if self.obs is None:
            self.obs = obs[-self.buffer_size:]
            self.acs = acs[-self.buffer_size:]
            self.res = res[-self.buffer_size:]
            self.next_obs = next_obs[-self.buffer_size:]
            self.dones = dones[-self.buffer_size:]
        else:
            self.obs = np.concatenate([self.obs, obs])[-self.buffer_size:]
            self.acs = np.concatenate([self.acs, acs])[-self.buffer_size:]
            self.res = np.concatenate([self.res, res])[-self.buffer_size:]
            self.next_obs = np.concatenate([self.next_obs, next_obs])[-self.buffer_size:]
            self.dones = np.concatenate([self.dones, dones])[-self.buffer_size:]

    #####################################################
    #####################################################

    def sample_random_data(self, batch_size):
        """Draws a random sample of data from the replay buffer, for Value-based agents"""
        indices = np.random.permutation(self.obs.shape[0])[:batch_size]
        return (self.obs[indices],
                self.acs[indices],
                self.res[indices],
                self.next_obs[indices],
                self.dones[indices])


    def sample_recent_data(self, batch_size, return_full_trajectory=True):
        """Draws a recent sample of data from the replay buffer
            Policy-based agents: data are from full episodes
        """
        if return_full_trajectory:
            n_recent_trajectories_to_return = 0
            index = -1
            n_data_so_far = 0

            while n_data_so_far < batch_size:
                recent_trajectory = self.memory[index]
                # n_data_so_far += len(recent_trajectory["reward"])
                n_data_so_far += recent_trajectory["reward"].shape[0]
                n_recent_trajectories_to_return += 1
                index -= 1

            trajectories_to_return = self.memory[-n_recent_trajectories_to_return:]
            obs, acs, res, next_obs, dones = _trajectories_to_data(trajectories_to_return)
            return obs, acs, res, next_obs, dones
        else:
            return (self.obs[-batch_size:],
                    self.acs[-batch_size:],
                    self.res[-batch_size:],
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
    """Convert lists of trajectories to training data \n
        param:
            trajectories: list of trajectories\n
        returns:
            obs: np.ndarray\n
            acs: np.ndarray\n
            res: np.ndarray\n
            next_obs: np.ndarray\n
            dones: np.ndarray
    """
    obs = np.concatenate([t["observation"] for t in trajectories])
    acs = np.concatenate([t["action"] for t in trajectories])
    res = np.concatenate([t["reward"] for t in trajectories])
    next_obs = np.concatenate([t["next_observation"] for t in trajectories])
    dones = np.concatenate([t["done"] for t in trajectories])

    return obs, acs, res, next_obs, dones