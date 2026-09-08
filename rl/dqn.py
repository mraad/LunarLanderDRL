import datetime
import logging
import os

import gymnasium as gym
import numpy as np
import torch
from tensorboardX import SummaryWriter
from torch.nn import MSELoss
from torch.optim import Adam

from .agent import Agent
from .checkpoint import Checkpoint
from .model import MLP
from .replaybuffer import ReplayBuffer
from .utils import dec_schedule, env_shape

logger = logging.getLogger(__name__)


class DQN:
    """Baseline Deep Q-Network: uniform replay and a hard target net copy every ``sync_rate`` episodes."""

    def __init__(self,
                 env_name: str = "LunarLander-v3",
                 eps_begin: float = 1.0,
                 eps_final: float = 0.01,
                 eps_ratio: float = 0.4,
                 n_episodes: int = 100_000,
                 sync_rate: int = 500,
                 gamma: float = 0.99,
                 learning_rate: float = 1.0e-3,
                 batch_size: int = 64,
                 replay_buffer_size: int = 100_000,
                 warm_start: int = 1_000,
                 avg_reward_len: int = 100,
                 seed: int = 42,
                 logs_dir: str = "logs",
                 ckpt_dir: str = "ckpt"
                 ) -> None:
        assert warm_start >= batch_size, "warm_start must fill at least one batch"
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.n_episodes = n_episodes
        self.env = gym.make(env_name)
        self.env.reset(seed=seed)
        self.env.action_space.seed(seed)
        self.n_states, self.n_actions = env_shape(self.env)
        self.net = MLP(self.n_states, self.n_actions).to(self.device)
        self.net_target = MLP(self.n_states, self.n_actions).to(self.device)
        self.epsilons = dec_schedule(eps_begin, eps_final, eps_ratio, n_episodes)
        self.agent = Agent(self.net, self.n_actions, self.device)
        self.gamma = gamma
        self.warm_start = warm_start
        self.sync_rate = sync_rate
        self.avg_reward_len = avg_reward_len
        self.logs_dir = logs_dir
        self.ckpt_dir = ckpt_dir
        self.rewards = []
        self.optimizer = Adam(self.net.parameters(), lr=learning_rate)
        self.loss = MSELoss().to(self.device)
        self.buffer = ReplayBuffer(replay_buffer_size, batch_size, self.n_states)

    def populate(self) -> None:
        """Seed the replay buffer with random transitions so the first batch is not degenerate."""
        state, _ = self.env.reset()
        for _ in range(self.warm_start):
            action = self.env.action_space.sample()
            next_state, reward, terminated, truncated, _ = self.env.step(action)
            self.buffer.append(state, action, float(reward), terminated, next_state)
            state = self.env.reset()[0] if terminated or truncated else next_state

    def learn(self,
              old_state: np.ndarray,
              action: int,
              reward: float,
              terminal: bool,
              new_state: np.ndarray) -> None:
        self.buffer.append(old_state, action, reward, terminal, new_state)

        states, actions, rewards, terminals, next_states = self.buffer.sample()

        states = torch.from_numpy(states).to(self.device)
        actions = torch.from_numpy(actions).to(self.device)
        rewards = torch.from_numpy(rewards).to(self.device)
        terminals = torch.from_numpy(terminals).to(self.device)
        next_states = torch.from_numpy(next_states).to(self.device)

        state_action_values = self.net(states).gather(1, actions.unsqueeze(-1)).squeeze(-1)

        with torch.no_grad():
            next_state_values = self.net_target(next_states).max(dim=1).values
            next_state_values[terminals] = 0.0
            expected_state_action_values = rewards + self.gamma * next_state_values

        loss = self.loss(state_action_values, expected_state_action_values)

        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        self.optimizer.step()

    def update_net_target(self) -> None:
        self.net_target.load_state_dict(self.net.state_dict())

    def train(self) -> None:
        self.update_net_target()
        now = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M")
        checkpoint = Checkpoint(os.path.join(self.ckpt_dir, f"{now}.pth"))
        with SummaryWriter(os.path.join(self.logs_dir, now)) as sw:
            self.populate()
            for e in range(self.n_episodes):
                self.agent.eps = self.epsilons[e]
                rewards = 0.0
                steps = 0
                done = False
                old_state, _ = self.env.reset()
                while not done:
                    action = self.agent(old_state)
                    new_state, reward, terminated, truncated, _ = self.env.step(action)
                    done = terminated or truncated
                    # Only a real terminal zeroes the bootstrap: a time-limit truncation
                    # cuts an episode that still had value beyond the cut.
                    self.learn(old_state, action, float(reward), terminated, new_state)
                    old_state = new_state
                    rewards += float(reward)
                    steps += 1

                if e % self.sync_rate == 0:
                    self.update_net_target()

                self.rewards.append(rewards)
                score_avg = np.average(self.rewards[-self.avg_reward_len:])

                if checkpoint.checkpoint(self.net, score_avg):
                    logger.info("episode %d: saved %s at average score %.1f", e, checkpoint.path, score_avg)

                sw.add_scalar("score/val", rewards, global_step=e)
                sw.add_scalar("score/avg", score_avg, global_step=e)
                sw.add_scalar("episode/eps", self.agent.eps, global_step=e)
                sw.add_scalar("episode/steps", steps, global_step=e)
