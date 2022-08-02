from infrastructure.utils import BarabasiAlbertGraphGenerator
# from envs.core import Env
import numpy as np
import random


class MaxCutEnv(object):
    """Environment for the MaxCut problem

    Observation:
        A graph data with node feature (x) and edge index

    Action:
        Adding a new node (v) to the partial solution
        turning v's node feature to 1: x_v = 1

    Reward:
        Change in the cut weight after taking action (v)

    Starting State:
        All node features are initialized to 0

    Episode Termination:
        Cut weight cannot be improved
    """


    class ActionSpace(object):
        """Implements the discrete action space"""
        def __init__(self, n_nodes):
            self.shape = [n_nodes, 1]
            self.action_list = None

        def contains(self, action):
            return True if action in self.action_list else False

        def remove(self, action):
            self.action_list.remove(action)

        def reset(self):
            self.action_list = list(range(self.shape[0]))

        def sample(self):
            return random.choice(self.action_list)


    class ObservationSpace(object):
        """Implements the multi-binary observation space"""
        def __init__(self, n_nodes):
            self.shape = [n_nodes, 1]
            self.state = None

        def reset(self):
            self.state = np.zeros(self.shape)


    def __init__(self, n_nodes, m_edges):

        self.n_nodes = n_nodes
        self.graph_generator = BarabasiAlbertGraphGenerator(n_nodes, m_edges)

        self.action_space = self.ActionSpace(self.n_nodes)
        self.observation_space = self.ObservationSpace(self.n_nodes)

        self.graph = None
        self.laplacian_matrix = None

        self.current_step = None
        self.best_score = None
        self.best_solution = None


    def step(self, action):
        """Runs one time step of the environment's dynamics"""
        reward = 0
        score = 0
        done = False

        assert self.action_space.contains(action), \
        "picked action must be in the action space"
        self.action_space.remove(action)

        self.current_step += 1
        state = np.copy(self.observation_space.state)
        last_score = self._calculate_score(state[:, 0], self.laplacian_matrix)

        ############################################################
        # 1. Performs the action and calculates the score change   #
        ############################################################
        state[action, 0] = 1
        self.observation_space.state = state
        self.graph = self.graph_generator.get(state)

        score = self._calculate_score(state[:, 0], self.laplacian_matrix)
        delta_score = score - last_score

        if score > self.best_score:
            self.best_score = score
            self.best_solution = state

        ############################################################
        # 2. Calculate the reward for the action                   #
        ############################################################
        reward = delta_score
        # if self.norm_rewards:
        #     reward /= self.n_nodes

        ############################################################
        # 3. Check termination criteria                            #
        ############################################################
        # TODO: fix termination criteria
        if len(self.action_space.action_list) == 0:
            done = True

        return (self.graph, reward, done,
            {"best_score": self.best_score,
             "best_solution": self.best_solution.T})


    def reset(self):
        """Resets the state of the environment and returns the
        `torch_geometric.data.Data` as an observation"""
        self.action_space.reset()
        self.observation_space.reset()
        self.graph_generator.reset()
        self.current_step = 0
        self.best_score = 0

        self.graph = self.graph_generator.get(self.observation_space.state)
        self.laplacian_matrix = self.graph_generator.get_laplacian()
        return self.graph


    def seed(self, seed):
        """Sets the seed for this env"""
        np.random.seed(seed)
        random.seed(seed)


    #####################################################
    ################## HELPER FUNCTIONS #################
    #####################################################

    def _calculate_score(self, state, laplacian_matrix):
        x = state*2 - 1 # convert 0 -> -1, 1 -> 1
        return (1/4) * np.dot(np.dot(x, laplacian_matrix), x)

    # def _calculate_score_change(self, next_state, action, laplacian_matrix):
    #     # raise NotImplementedError
    #     return np.matmul(next_state.T, laplacian_matrix[:, action])