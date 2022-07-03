import sys
from os.path import dirname, abspath
sys.path.append(dirname(dirname(abspath(__file__))))

from agents.pg_agent import PGAgent
from agents.rl_trainer import Trainer
from infrastructure.config import Config


config = Config()
config.seed = 1
config.env_name = "MaxCut"
config.num_episodes_to_run = 30
config.num_steps_per_episode = 1
config.use_GPU = False
config.which_GPU = 0
config.randomize_random_seed = True
config.hyperparameters = {
    "batch_size": 16,
    "buffer_size": 10000000,
    "learning_rate": 0.05,
    "hidden_size": 30,
    "n_hidden_layers": 2,
    "activation": "tanh",
    "output_activation": "identity",
    "discount_rate": 0.99}


if __name__ == "__main__":
    AGENTS = PGAgent # class import from agents.xxx
    trainer = Trainer(config, AGENTS)
    trainer.run_training_loop()