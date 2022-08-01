import sys
import os
from os.path import dirname, abspath, join
sys.path.append(dirname(dirname(abspath(__file__))))
import time
import comet_ml

from agents.dqn_agent import DQNAgent
from infrastructure.rl_trainer import RLTrainer
from infrastructure.config import Config


config = Config()
config.seed = 1
config.env_name = "MaxCut"
config.n_episodes = 100000
config.n_steps_per_episode = 1

config.update_learning_rate = True
config.update_exploration = True

config.log_metrics = True
config.use_GPU = True
config.which_GPU = 0
config.randomize_random_seed = True
config.hparams = {
    "buffer_size": 5000,
    "learning_starts": 500,
    "batch_size": 64,

    "hidden_size": 64,
    "n_layers": 3,
    "learning_rate": 1e-4,

    "update_freq": 32,
    "target_update_freq": 500,
    "discount_rate": 0.99,
    "initial_exploration_rate": 1,
    "final_exploration_rate": 0.05,
    "final_exploration_step": 150000,
    "double_q": True,
    "grad_norm_clipping": None}


def main():
    data_path = join(dirname(dirname(abspath(__file__))), 'data')
    if not (os.path.exists(data_path)): os.makedirs(data_path)
    log_dir = config.env_name + '_' + time.strftime("%m-%d_%H-%M")
    log_dir = join(data_path, log_dir)
    config.log_dir = log_dir
    print("\n\n\nLOGGING TO: ", log_dir, "\n\n\n")

    comet_ml.init(project_name='gnn4co')
    AGENTS = DQNAgent
    trainer = RLTrainer(config, AGENTS)
    trainer.run_training_loop()


if __name__ == "__main__":
    main()