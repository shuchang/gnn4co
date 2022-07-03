from envs.combinatorial.max_cut import MaxCutEnv


def make(id, *args, **kwargs):
    """Instantiates an instance of the environment"""
    if id == "MaxCut":
        env = MaxCutEnv(*args, **kwargs)
    elif id =="GraphPartition":
        raise NotImplementedError
    else:
        raise NotImplementedError
    return env


class Env(object):
    """Encapsulates an environment with arbitrary behind-the-scenes dynamics"""

    # Set this in SOME subclasses
    # reward_range = (-float('inf'), float('inf'))
    # spec = None

    # Set these in ALL subclasses
    # action_space = None
    # observation_space = None

    def step(self, action):
        """Run one time step of the environment's dynamics. When end of
        episode is reached, you are responsible for calling `reset()`
        to reset this environment's state.

        Accepts an action and returns a tuple (observation, reward, done, info).

        Args:
            action (object): an action provided by the environment

        Returns:
            observation (object): agent's observation of the current environment
            reward (float) : amount of reward returned after previous action
            done (boolean): whether the episode has ended, in which case further step() calls will return undefined results
            info (dict): contains auxiliary diagnostic information (helpful for debugging, and sometimes learning)
        """
        raise NotImplementedError

    def reset(self):
        """Resets the state of the environment and returns an initial observation.

        Returns: observation (object): the initial observation of the
            space.
        """
        raise NotImplementedError

    def seed(self, seed=None):
        """Sets the seed for this env's random number generator(s).

        Note:
            Some environments use multiple pseudorandom number generators.
            We want to capture all such seeds used in order to ensure that
            there aren't accidental correlations between multiple generators.

        Returns:
            list<bigint>: Returns the list of seeds used in this env's random
              number generators. The first value in the list should be the
              "main" seed, or the value which a reproducer should pass to
              'seed'. Often, the main seed equals the provided 'seed', but
              this won't be true if seed=None, for example.
        """
        raise NotImplementedError