from collections import OrderedDict
import os
import torch
import random
import time
import numpy as np
from agents.dqn_agent import DQNAgent
from infrastructure import pytorch_utils as ptu
from tensorboardX import SummaryWriter
import envs.core as co_env


class RLTrainer(object):
    """Runs experiments for given agents and saves the results"""

    def __init__(self, config, agent_class):

        #############
        ## INIT
        #############
        self.writer = SummaryWriter(config.log_dir, comet_config={"disabled": False})
        self.writer.add_hparams(config.hparams, metric_dict={})

        if config.randomize_random_seed:
            config.seed = random.randint(0, 2**32-2)
        self.set_random_seeds(config.seed)
        ptu.init_gpu(config.use_GPU, config.which_GPU)

        self.n_episodes = config.n_episodes
        self.n_steps_per_episode = config.n_steps_per_episode
        self.log_metrics = config.log_metrics
        self.batch_size = config.hparams["batch_size"]
        self.update_learning_rate = config.update_learning_rate
        self.update_exploration = False

        if agent_class == DQNAgent:
            self.update_exploration = config.update_exploration
            self.initial_exp_rate = config.hparams["initial_exploration_rate"]
            self.final_exp_rate = config.hparams["final_exploration_rate"]
            self.final_exp_step = config.hparams["final_exploration_rate"]
            self.epsilon = config.hparams["initial_exploration_rate"]
            self.learning_starts = config.hparams["learning_starts"]

        #############
        ## ENV
        #############
        self.env = co_env.make(config.env_name, n_nodes=20, m_edges=4)
        self.env.seed(config.seed)
        self.max_traj_len = 10

        config.hparams["n_nodes"] = self.env.n_nodes
        config.hparams["ob_dim"] = self.env.observation_space.shape[1]
        config.hparams["ac_dim"] = self.env.action_space.shape[1]

        #############
        ## AGENT
        #############
        self.agent = agent_class(config)


    def run_training_loop(self):
        """Runs a set of training loops for the agent"""
        self.total_env_steps = 0
        self.start_time = time.time()
        print_freq = 100 if isinstance(self.agent, DQNAgent) else 1

        for ep in range(self.n_episodes):

            # if self.update_learning_rate:
                # self.update_lr(ep)
            if self.update_exploration:
                self.update_epsilon(ep)

            if ep % print_freq == 0:
                print("\n\n********** Episode %i ************"%ep)
                print("\nCollecting data for train ...")

            trajectories = self.collect_trajectories()
            self.agent.add_to_replay_buffer(trajectories)

            if ep % print_freq == 0:
                print("\nTraining agent ...")

            train_logs = self.train_agent()

            if self.log_metrics:
                # print('\nBeginning logging procedure...')
                if isinstance(self.agent, DQNAgent):
                    self.perform_dqn_logging(train_logs)
                else:
                    self.perform_logging(ep, train_logs)


    def collect_trajectories(self):
        """Collects a batch of trajectories"""
        env_steps_this_batch = 0
        trajectories = []

        while env_steps_this_batch < self.batch_size:
            trajectory = self.collect_trajectory()
            env_steps_this_batch += trajectory["reward"].shape[0]
            trajectories.append(trajectory)

        return trajectories


    def collect_trajectory(self):
        """Collects one trajectory by letting the agent interact with the env"""
        obs, acs, rews, next_obs, dones = [], [], [], [], []
        step = 0
        ob = self.env.reset()

        while True:
            action_list = self.env.action_space.action_list

            if not isinstance(self.agent, DQNAgent):
                ac = self.agent.actor.get_action(ob, action_list)
            else:
                if (self.total_env_steps < self.learning_starts or
                    np.random.random() < self.epsilon):
                    ac = self.env.action_space.sample()
                else:
                    ac = self.agent.actor.get_action(ob, action_list)

            obs.append(ob)
            acs.append(ac)

            ob, rew, done, info = self.env.step(ac)
            rews.append(rew)
            next_obs.append(ob)
            rollout_done = 1 if done or step == self.max_traj_len else 0
            dones.append(rollout_done)
            step += 1
            self.total_env_steps += 1

            if rollout_done:
                break

        return {"observation" : obs,
                "action" : np.array(acs),
                "reward" : np.array(rews),
                "next_observation": next_obs,
                "done": np.array(dones)}


    def train_agent(self):
        """Samples data from the replay buffer and trains the agent"""
        all_logs = []

        for _ in range(self.n_steps_per_episode):
            transitions = self.agent.sample_from_replay_buffer(self.batch_size)
            obs, acs, rews, next_obs, dones = transitions
            train_log = self.agent.train(obs, acs, rews, next_obs, dones)
            all_logs.append(train_log)

        return all_logs


    def eval_agent(self):
        """Evaluates the agent by letting the agent interact with the env"""
        env_steps_this_batch = 0
        eval_best_scores = []

        while env_steps_this_batch < self.batch_size:
            ob = self.env.reset()

            while True:
                action_list = self.env.action_space.action_list
                ac = self.agent.actor.get_action(ob, action_list)
                ob, rew, done, info = self.env.step(ac)
                env_steps_this_batch += 1

                if done:
                    eval_best_scores.append(info["best_score"])
                    break

        return eval_best_scores


    def perform_dqn_logging(self, train_logs):
        """Performs logging for DQN agent"""
        logs = OrderedDict()
        # train_steps is usually less than total_env_steps as
        # sample_recent_data doesn't require full trajectories

        logs["Train_EnvstepsSoFar"] = self.agent.train_steps
        # logs["TimeSinceStart"] = time.time() - self.start_time
        logs.update(train_logs[-1])

        for key, value in logs.items():
            print('{} : {}'.format(key, value))
            self.writer.add_scalar('{}'.format(key), value, self.agent.train_steps)

        self.writer.flush()


    def perform_logging(self, ep, train_logs):
        """Returns the training and evaluating logs for each batch of data"""
        eval_best_scores = self.eval_agent()

        logs = OrderedDict()
        logs["Eval_AverageBestScore"] = np.mean(eval_best_scores)
        logs["Eval_StdBestScore"] = np.std(eval_best_scores)
        logs["Eval_MaxBestScore"] = np.max(eval_best_scores)
        logs["Eval_MinBestScore"] = np.min(eval_best_scores)

        # logs["Train_EnvstepsSoFar"] = self.total_env_steps
        # logs["TimeSinceStart"] = time.time() - self.start_time
        logs.update(train_logs[-1]) # for n_steps_per_episode > 1

        for key, value in logs.items():
            print('{}: {}'.format(key, value))
            self.writer.add_scalar('{}'.format(key), value, ep)

        self.writer.flush()


    # def update_lr(ep):

    def update_epsilon(self, ep):
        eps = self.initial_exp_rate + (self.final_exp_rate - self.initial_exp_rate)*(
            ep/self.final_exp_step)
        self.epsilon = max(eps, self.final_exp_rate)


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