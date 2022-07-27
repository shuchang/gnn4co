import sys
from os.path import dirname, abspath
sys.path.append(dirname(dirname(abspath(__file__))))

from agents.dqn_agent import DQNAgent
from infrastructure.rl_trainer import RLTrainer
from infrastructure.config import Config


config = Config()
config.seed = 1
config.env_name = "MaxCut"
config.n_episodes = 300
config.n_steps_per_episode = 1
config.log_metrics = True
config.use_GPU = False
config.which_GPU = 0
config.randomize_random_seed = True
config.hyperparameters = {
    "batch_size": 64,
    "buffer_size": 10000000,
    "learning_rate": 0.01,
    "hidden_size": 64,
    "n_hidden_layers": 2,
    "activation": "tanh",
    "output_activation": "identity",
    "discount_rate": 0.99,
    "exploration_rate": 0.8,
    "double_q": True,
    "grad_norm_clipping":10}


if __name__ == "__main__":
    AGENTS = DQNAgent
    trainer = RLTrainer(config, AGENTS)
    trainer.run_training_loop()