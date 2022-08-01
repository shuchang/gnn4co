class Config(object):
    """Object to hold the config requirements for an agent"""

    def __init__(self):
        self.seed = None
        self.env_name = None
        self.n_episodes = None
        self.n_steps_per_episode = None
        self.log_metrics = None
        self.update_learning_rate = True
        self.update_exploration = True
        self.hyperparameters = None
        self.use_GPU = None
        self.which_GPU = 0
        self.randomize_random_seed = True