# import logging
import os
import torch
import random
import time
import numpy as np
from infrastructure import pytorch_utils as ptu
# from tensorboardX import SummaryWriter
import envs.core as co_env


class Trainer(object):
    """Runs experiments for given agents, and optionally visualizes and saves the results"""

    def __init__(self, config, agent_class):

        #############
        ## INIT
        #############
        if config.randomize_random_seed:
            config.seed = random.randint(0, 2**32-2)
        self.config = config
        # self.logger =
        self.set_random_seeds(config.seed)
        ptu.init_gpu(config.use_GPU, config.which_GPU)

        #############
        ## ENV
        #############
        self.env = co_env.make(config.env_name, n_nodes=3, m_edges=2)
        self.env.seed(config.seed)

        # observation and action space size
        ob_dim = self.env.observation_space.shape[0]
        ac_dim = self.env.action_space.n
        config.hyperparameters["ob_dim"] = ob_dim
        config.hyperparameters["ac_dim"] = ac_dim

        #############
        ## AGENT
        #############
        self.agent = agent_class(config)


    def run_training_loop(self):
        """Runs a set of training loops for the agent"""
        batch_size = self.config.hyperparameters["batch_size"]
        self.total_env_steps = 0
        self.start_time = time.time()

        for ep in range(self.config.num_episodes_to_run):
            print("\n\n********** Episode %i ************"%ep)
            # self.log_metrics = True
            trajectories, env_steps = self.collect_n_trajectories(batch_size)
            self.total_env_steps += env_steps

            self.agent.add_to_replay_buffer(trajectories)
            all_logs = self.train_agent(batch_size)

            print(all_logs[0]["Training Loss"])
            # if self.log_metrics:
            #     self.perform_logging(all_logs)


    def collect_n_trajectories(self, batch_size):
        """Collect a batch of trajectories for training"""
        env_steps = 0
        trajectories = []

        while env_steps < batch_size:
            trajectory = self.collect_trajectory(batch_size)
            # env_steps += len(trajectory["reward"])
            env_steps += trajectory["reward"].shape[0]
            trajectories.append(trajectory)

        return trajectories, env_steps


    def collect_trajectory(self, batch_size) -> dict:
        """Collects one trajectory by letting the agent interact with the env"""
        obs, acs, res, next_obs, dones = [], [], [], [], []
        steps = 0
        ob = self.env.reset()

        while True:
            # agent's turn to interact
            action_list = self.env.action_space.action_list
            ac = self.agent.actor.get_action(ob, action_list)
            obs.append(ob)
            acs.append(ac)

            # env's turn to interact
            ob, re, done, _ = self.env.step(ac)
            steps += 1
            res.append(re)
            next_obs.append(ob)

            rollout_done = 1 if done or steps == batch_size else 0
            dones.append(rollout_done)
            if rollout_done:
                break

        return {"observation" : np.array(obs, dtype=np.float32),
                "action" : np.array(acs, dtype=np.float32),
                "reward" : np.array(res, dtype=np.float32),
                "next_observation": np.array(next_obs, dtype=np.float32),
                "done": np.array(dones, dtype=np.float32)}


    def train_agent(self, batch_size):
        """Samples from the replay buffer and trains the agent"""
        all_logs = []

        for _ in range(self.config.num_steps_per_episode):
            obs, acs, res, next_obs, dones = self.agent.sample_from_replay_buffer(batch_size)
            train_log = self.agent.train(obs, acs, res, next_obs, dones)
            all_logs.append(train_log)

        return all_logs


    def set_random_seeds(self, random_seed):
        """Sets all possible random seeds so results can be reproduced"""
        os.environ['PYTHONHASHSEED'] = str(random_seed)
        random.seed(random_seed)
        np.random.seed(random_seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.manual_seed(random_seed) # for CPU

        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(random_seed) # for current GPU
            torch.cuda.manual_seed(random_seed) # for all GPU