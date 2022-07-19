class Config(object):
    """Object to hold the config requirements for an agent/game"""
    def __init__(self):
        self.seed = None
        self.env_name = None
        # self.requirements_to_solve_game = None
        self.n_episodes_to_run = None
        self.n_steps_per_episode = None
        self.log_metrics = None
        # self.file_to_save_results_data = None
        # self.file_to_save_results_graph = None
        # self.runs_per_agent = None
        # self.visualize_overall_results = None
        # self.visualize_individual_results = None
        self.hyperparameters = None
        self.use_GPU = None
        self.which_GPU = 0
        # self.overwrite_existing_results_file = None
        # self.save_model = False
        # self.standard_deviation_results = 1.0
        self.randomize_random_seed = True
        # self.show_solution_score = False