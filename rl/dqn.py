import datetime
import logging
import os

import gym
import numpy as np
import torch
from tensorboardX import SummaryWriter
from torch.nn import MSELoss
from torch.optim import Adam

from .agent import Agent
from .model import MLP
from .replaybuffer import ReplayBuffer
from .utils import Utils


class DQN:
    def __init__(self,
                 env_name: str = "LunarLander-v2",
                 eps_begin: float = 1.0,
                 eps_final: float = 0.01,
                 eps_decay: float = 0.4,
                 n_episodes: int = 100_000,
                 sync_rate: int = 500,
                 gamma: float = 0.99,
                 learning_rate: float = 1.0e-3,
                 batch_size: int = 64,
                 replay_buffer_size: int = 100_000,
                 warm_start: int = 1_000,
                 avg_reward_len: int = 100,
                 seed: int = 42,
                 logdir: str = "logs"
                 ):
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.n_episodes = n_episodes
        self.env = gym.make(env_name)
        self.env.seed(seed)
        self.obs_shape = self.env.observation_space.shape
        self.n_states = self.obs_shape[0]
        self.n_actions = self.env.action_space.n
        self.net = MLP(self.n_states, self.n_actions)
        self.net.to(self.device)
        self.net_target = MLP(self.n_states, self.n_actions)
        self.net_target.to(self.device)
        self.epsilons = Utils.decay_schedule(eps_begin, eps_final, eps_decay, n_episodes)
        self.agent = Agent(self.net, self.n_actions, self.device)
        self.gamma = gamma
        self.lr = learning_rate
        self.batch_size = batch_size
        self.warm_start = warm_start
        self.buffer = ReplayBuffer(replay_buffer_size, batch_size)
        self.sync_rate = sync_rate
        self.avg_reward_len = avg_reward_len
        self.logdir = logdir
        self.rewards = []
        self.optimizer = Adam(self.net.parameters(), lr=self.lr)
        self.loss = MSELoss()
        self.loss.to(self.device)

    def populate(self) -> None:
        logging.info(f"Populating replay buffer with {self.warm_start} items.")
        if self.warm_start > 0:
            state = self.env.reset()
            for _ in range(self.warm_start):
                action = self.env.action_space.sample()
                next_state, reward, done, _ = self.env.step(action)
                self.buffer.append(state, action, reward, done, next_state)
                state = next_state
                if done:
                    state = self.env.reset()

    def learn(self) -> None:
        states, actions, rewards, terminals, next_states = self.buffer.sample()

        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        terminals = torch.BoolTensor(terminals).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)

        actions_v = actions.unsqueeze(-1)
        outputs = self.net(states)
        state_action_values = outputs.gather(1, actions_v)
        state_action_values = state_action_values.squeeze(-1)

        with torch.no_grad():
            next_state_values, _ = self.net_target(next_states).max(axis=1)
            next_state_values[terminals] = 0.0
            expected_state_action_values = rewards + self.gamma * next_state_values.detach()

        loss = self.loss(state_action_values, expected_state_action_values)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def train(self) -> None:
        now = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M")
        log_dir = os.path.join(self.logdir, now)
        logging.info(f"Tensorboard logdir={log_dir}")
        sw = SummaryWriter(log_dir=log_dir)
        try:
            self.populate()
            for e in range(self.n_episodes):
                self.agent.eps = self.epsilons[e]
                rewards = 0.0
                steps = 0
                done = False
                old_state = self.env.reset()
                while not done:
                    action = self.agent(old_state)
                    new_state, reward, done, _ = self.env.step(action)
                    self.buffer.append(old_state, action, reward, done, new_state)
                    self.learn()
                    old_state = new_state
                    rewards += reward
                    steps += 1

                if e % self.sync_rate == 0:
                    self.net_target.load_state_dict(self.net.state_dict())

                self.rewards.append(rewards)
                score_avg = np.average(self.rewards[-self.avg_reward_len:])
                sw.add_scalar("score/val", rewards, global_step=e)
                sw.add_scalar("score/avg", score_avg, global_step=e)
                sw.add_scalar("episode/eps", self.agent.eps, global_step=e)
                sw.add_scalar("episode/steps", steps, global_step=e)
        finally:
            sw.close()