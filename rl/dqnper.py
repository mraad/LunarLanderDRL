import datetime
import os

import gym
import numpy as np
import torch
from tensorboardX import SummaryWriter
from torch.optim import Adam

from .agent import Agent
from .checkpoint import Checkpoint
from .model import MLP
from .replaybuffer import PEReplayBuffer
from .utils import Utils


class DQNPER:
    def __init__(self,
                 env_name: str = "LunarLander-v2",
                 eps_begin: float = 1.0,
                 eps_final: float = 0.01,
                 eps_decay: float = 0.4,
                 n_episodes: int = 100_000,
                 gamma: float = 0.99,
                 learning_rate: float = 1.0e-3,
                 batch_size: int = 64,
                 replay_buffer_size: int = 100_000,
                 replay_buffer_alpha: float = 0.4,
                 replay_buffer_beta: float = 0.6,
                 replay_buffer_eps: float = 0.001,
                 target_update_freq: int = 1,
                 target_update_tau: float = 1.0e-3,
                 warm_start: int = 1_000,
                 avg_reward_len: int = 100,
                 seed: int = 42,
                 logs_dir: str = "logs",
                 ckpt_dir: str = "ckpt"
                 ) -> None:
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
        self.warm_start = warm_start
        self.target_update_freq = target_update_freq
        self.target_update_tau = target_update_tau
        self.avg_reward_len = avg_reward_len
        self.logs_dir = logs_dir
        self.ckpt_dir = ckpt_dir
        self.rewards = []
        self.optimizer = Adam(self.net.parameters(), lr=learning_rate)
        # self.loss = MSELoss()
        # self.loss.to(self.device)
        self.buffer = PEReplayBuffer(replay_buffer_size, batch_size, replay_buffer_alpha, replay_buffer_beta, replay_buffer_eps)

    def populate(self) -> None:
        if self.warm_start > 0:
            state = self.env.reset()
            for _ in range(self.warm_start):
                action = self.env.action_space.sample()
                next_state, reward, done, _ = self.env.step(action)
                self.buffer.append(state, action, reward, done, next_state)
                state = next_state
                if done:
                    state = self.env.reset()

    def update_target_copy(self) -> None:
        self.net_target.load_state_dict(self.net.state_dict())

    def update_target_soft(self) -> None:
        # for target_param, param in zip(target.parameters(), source.parameters()):
        #     target_param.data.copy_(
        #             target_param.data * (1.0 - tau) + param.data * tau
        #     )
        for target_param, param in zip(self.net_target.parameters(), self.net.parameters()):
            target_param.detach_()
            target_param.copy_(target_param * (1.0 - self.target_update_tau) + param * self.target_update_tau)

    def learn(self,
              old_state: np.array,
              action: int,
              reward: float,
              terminal: bool,
              new_state) -> None:
        self.buffer.append(old_state, action, reward, terminal, new_state)

        states, actions, rewards, terminals, next_states, indices, priorities = self.buffer.sample()

        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        terminals = torch.BoolTensor(terminals).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        priorities = torch.FloatTensor(priorities).to(self.device)

        q_values = self.net(states)
        q_values = q_values.gather(1, actions.unsqueeze(-1)).squeeze(-1)

        with torch.no_grad():
            next_q_values, _ = self.net_target(next_states).max(axis=1)
            next_q_values[terminals] = 0.0
            expected_q_values = rewards + self.gamma * next_q_values.detach()

        errors = q_values - expected_q_values

        buffer_priorities = errors.abs().add(self.buffer.eps).pow(self.buffer.alpha)
        self.buffer.update_priorities(indices, buffer_priorities.detach().numpy())

        weights = priorities. \
            mul(self.buffer.priority_factor). \
            add(1e-6). \
            pow(-self.buffer.beta)
        weights = weights / weights.max()
        losses = weights * errors ** 2.0
        loss = losses.mean()

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def train(self) -> None:
        self.update_target_copy()
        now = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M")
        checkpoint = Checkpoint(os.path.join(self.ckpt_dir, f"{now}.pth"))
        with SummaryWriter(os.path.join(self.logs_dir, now)) as sw:
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
                    self.learn(old_state, action, reward, done, new_state)
                    old_state = new_state
                    rewards += reward
                    steps += 1

                if e % self.target_update_freq == 0:
                    self.update_target_soft()

                self.rewards.append(rewards)
                score_avg = np.average(self.rewards[-self.avg_reward_len:])

                checkpoint.checkpoint(self.net, score_avg)

                sw.add_scalar("score/val", rewards, global_step=e)
                sw.add_scalar("score/avg", score_avg, global_step=e)
                sw.add_scalar("episode/eps", self.agent.eps, global_step=e)
                sw.add_scalar("episode/steps", steps, global_step=e)