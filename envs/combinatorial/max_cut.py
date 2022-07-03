from infrastructure.utils import BarabasiAlbertGraphGenerator
# from envs.core import Env
import numpy as np
import random


class MaxCutEnv(object):
    """Environment for the MaxCut problem

    Observation:
        An ordered list of nodes that represents a partial solution
        + adjacent matrix

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

    # Note these are defined at the class level of SpinSystem to ensure that SpinSystem
    # can be pickled.
    class Action_Space(object):
        """Implements the discrete action space"""
        def __init__(self, n_actions):
            self.n = n_actions
            self.action_list = list(range(self.n))

        def contains(self, action):
            return True if action in self.action_list else False

        def remove(self, action):
            self.action_list.remove(action)

        def reset(self):
            self.action_list = list(range(self.n))

        def sample(self):
            """Returns a randomly sampled action"""
            return random.choice(self.action_list)

    class Observation_Space(object):
        """Implements the multi-binary observation space"""
        def __init__(self, n_nodes):
            self.shape = [n_nodes, 1]


    def __init__(self, n_nodes, m_edges):

        self.n_nodes = n_nodes
        self.graph_generator = BarabasiAlbertGraphGenerator(n_nodes, m_edges)

        self.action_space = self.Action_Space(self.n_nodes)
        self.observation_space = self.Observation_Space(self.n_nodes)

        self.state = None
        self.laplacian_matrix = None

        self.current_step = None
        self.best_score = None
        self.best_solution = None


    def step(self, action: int):
        """Run one time step of the environment's dynamics"""
        reward = 0
        score = 0
        done = False

        assert self.action_space.contains(action)
        self.action_space.remove(action)

        self.current_step += 1
        state = np.copy(self.state)
        last_score = self._calculate_score(state[0, :], self.laplacian_matrix)

        ############################################################
        # 1. Performs the action and calculates the score change   #
        ############################################################
        state[0, action] = 1
        self.state = state

        score = self._calculate_score(state[0, :], self.laplacian_matrix)
        delta_score = score - last_score

        # if score > self.best_score:
        #     self.best_score = score
        #     self.best_solution = state

        ############################################################
        # 2. Calculate the reward for the action                   #
        ############################################################
        reward = delta_score
        # if self.norm_rewards:
        #     reward /= self.n_nodes

        ############################################################
        # 3. Check termination criteria                            #
        ############################################################
        if len(self.action_space.action_list) == 1:
            done = True

        return np.vstack((self.state, self.laplacian_matrix)), reward, done, {}


    def reset(self):
        """Resets the state of the environment and returns an initial observation"""
        self.current_step = 0
        self.best_score = 0

        # reset action space
        self.action_space.reset()
        # reset observation space
        self.state = np.zeros((self.observation_space.shape[1], self.n_nodes))
        self.laplacian_matrix = self.graph_generator.get_laplacian()

        return np.vstack((self.state, self.laplacian_matrix))


    def seed(self, seed):
        """Sets the seed for this env's random number generator(s)"""
        self.seed = seed
        np.random.seed(seed)
        random.seed(seed)


    #####################################################
    ################## HELPER FUNCTIONS #################
    #####################################################

    def _calculate_score(self, state, laplacian_matrix):
        x = state*2 - 1 # convert 0 -> -1, 1 -> 1
        return (1/4)*np.sum(np.multiply(laplacian_matrix, 1-np.outer(x, x)))

    # def _calculate_score_change(self, next_state, action, laplacian_matrix):
    #     # raise NotImplementedError
    #     return np.matmul(next_state.T, laplacian_matrix[:, action])
