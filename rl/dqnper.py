import datetime
import logging
import os
from collections import deque

import gymnasium as gym
import numpy as np
import torch
from tensorboardX import SummaryWriter
from torch.optim import Adam

from .agent import Agent
from .checkpoint import Checkpoint
from .model import MLP
from .replaybuffer import PEReplayBuffer
from .utils import dec_schedule, env_shape, inc_schedule

logger = logging.getLogger(__name__)


class DQNPER:
    """Deep Q-Network with prioritized experience replay and a Polyak averaged target net."""

    def __init__(self,
                 env_name: str = "LunarLander-v3",
                 eps_begin: float = 1.0,
                 eps_final: float = 0.01,
                 eps_ratio: float = 0.4,
                 n_episodes: int = 100_000,
                 gamma: float = 0.99,
                 learning_rate: float = 1.0e-3,
                 batch_size: int = 64,
                 replay_buffer_size: int = 100_000,
                 replay_buffer_alpha: float = 0.6,
                 replay_buffer_beta_begin: float = 0.4,
                 replay_buffer_beta_final: float = 1.0,
                 replay_buffer_beta_ratio: float = 0.1,
                 replay_buffer_eps: float = 0.001,
                 target_update_freq: int = 1,
                 target_update_tau: float = 1.0e-3,
                 grad_clip: float = 10.0,
                 n_step: int = 3,
                 warm_start: int = 1_000,
                 avg_reward_len: int = 100,
                 eval_freq: int = 25,
                 eval_episodes: int = 10,
                 solved_score: float = 260.0,
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
        # A separate environment for evaluation, so greedy rollouts never perturb the
        # training environment's RNG stream and training stays reproducible.
        self.eval_env = gym.make(env_name)
        self.eval_freq = eval_freq
        self.eval_episodes = eval_episodes
        self.solved_score = solved_score
        self.eval_seed = seed + 500
        self.n_states, self.n_actions = env_shape(self.env)
        self.net = MLP(self.n_states, self.n_actions).to(self.device)
        self.net_target = MLP(self.n_states, self.n_actions).to(self.device)
        self.agent = Agent(self.net, self.n_actions, self.device)
        self.gamma = gamma
        self.warm_start = warm_start
        self.target_update_freq = target_update_freq
        self.target_update_tau = target_update_tau
        self.grad_clip = grad_clip
        self.n_step = n_step
        # Rewards are folded over this sliding window before a transition is stored, so
        # the terminal landing bonus reaches earlier states n_step times faster.
        self.pending = deque(maxlen=n_step)
        self.avg_reward_len = avg_reward_len
        self.logs_dir = logs_dir
        self.ckpt_dir = ckpt_dir
        self.rewards = []
        self.steps = 0
        self.optimizer = Adam(self.net.parameters(), lr=learning_rate)
        self.buffer = PEReplayBuffer(replay_buffer_size,
                                     batch_size,
                                     self.n_states,
                                     replay_buffer_alpha,
                                     replay_buffer_beta_begin,
                                     replay_buffer_eps)
        self.epsilons = dec_schedule(eps_begin, eps_final, eps_ratio, n_episodes)
        self.betas = inc_schedule(replay_buffer_beta_begin, replay_buffer_beta_final, replay_buffer_beta_ratio, n_episodes)

    def _fold(self) -> tuple:
        """Collapse the pending window into one n-step transition.

        Returns ``(state, action, R, terminated, next_state, gamma ** k)`` where ``R`` is
        the discounted sum of the k rewards actually accumulated. Stopping early at a
        terminal keeps the return from crossing an episode boundary.
        """
        state, action = self.pending[0][0], self.pending[0][1]
        total, discount, terminated, next_state = 0.0, 1.0, False, self.pending[0][4]
        for _, _, reward, term, new_state in self.pending:
            total += discount * reward
            discount *= self.gamma
            next_state = new_state
            if term:
                terminated = True
                break
        return state, action, total, terminated, next_state, discount

    def remember(self,
                 state: np.ndarray,
                 action: int,
                 reward: float,
                 terminated: bool,
                 next_state: np.ndarray) -> None:
        """Feed one environment step through the n-step window into the replay buffer."""
        self.pending.append((state, action, reward, terminated, next_state))
        if len(self.pending) == self.n_step:
            self.buffer.append(*self._fold())

    def flush(self) -> None:
        """Drain the window at an episode boundary as shorter partial n-step returns."""
        while self.pending:
            self.buffer.append(*self._fold())
            self.pending.popleft()

    def populate(self) -> None:
        """Seed the replay buffer with random transitions so the first batch is not degenerate."""
        state, _ = self.env.reset()
        for _ in range(self.warm_start):
            action = self.env.action_space.sample()
            next_state, reward, terminated, truncated, _ = self.env.step(action)
            self.remember(state, action, float(reward), terminated, next_state)
            if terminated or truncated:
                self.flush()
                state = self.env.reset()[0]
            else:
                state = next_state
        self.flush()

    @torch.no_grad()
    def evaluate(self) -> float:
        """Mean reward of the greedy policy over a fixed set of seeds.

        This, not the running average of training episodes, is what the checkpoint is
        selected on. Training episodes are epsilon-greedy, so their average rewards a
        lucky streak of exploration rather than a genuinely better policy.
        """
        total = 0.0
        for i in range(self.eval_episodes):
            state, _ = self.eval_env.reset(seed=self.eval_seed + i)
            done = False
            while not done:
                pt_state = torch.as_tensor(state, dtype=torch.float32, device=self.device)
                action = int(self.net(pt_state).argmax().item())
                state, reward, terminated, truncated, _ = self.eval_env.step(action)
                done = terminated or truncated
                total += float(reward)
        return total / self.eval_episodes

    def update_target_copy(self) -> None:
        self.net_target.load_state_dict(self.net.state_dict())

    @torch.no_grad()
    def update_target_soft(self) -> None:
        tau = self.target_update_tau
        for target_param, param in zip(self.net_target.parameters(), self.net.parameters()):
            target_param.mul_(1.0 - tau).add_(param, alpha=tau)

    def learn(self,
              old_state: np.ndarray,
              action: int,
              reward: float,
              terminal: bool,
              new_state: np.ndarray) -> None:
        self.remember(old_state, action, reward, terminal, new_state)
        if len(self.buffer) < self.buffer.batch_size:
            return

        states, actions, rewards, terminals, next_states, discounts, indices, weights = self.buffer.sample()

        states = torch.from_numpy(states).to(self.device)
        actions = torch.from_numpy(actions).to(self.device)
        rewards = torch.from_numpy(rewards).to(self.device)
        terminals = torch.from_numpy(terminals).to(self.device)
        next_states = torch.from_numpy(next_states).to(self.device)
        discounts = torch.from_numpy(discounts).to(self.device)
        weights = torch.from_numpy(weights).to(self.device)

        q_values = self.net(states).gather(1, actions.unsqueeze(-1)).squeeze(-1)

        with torch.no_grad():
            # Double DQN: the online net picks the next action, the target net prices it.
            # A single net doing both takes a max over its own noise, and that bias
            # compounds through bootstrapping until the policy collapses.
            next_actions = self.net(next_states).argmax(dim=1, keepdim=True)
            next_q_values = self.net_target(next_states).gather(1, next_actions).squeeze(-1)
            next_q_values[terminals] = 0.0
            # discounts is gamma ** k, already folded over the n-step window.
            expected_q_values = rewards + discounts * next_q_values

        errors = q_values - expected_q_values
        # Huber, not squared error: PER deliberately over-samples large TD errors, and
        # squaring them lets one outlier dominate the batch gradient.
        losses = torch.nn.functional.smooth_l1_loss(q_values, expected_q_values, reduction='none')
        loss = (weights * losses).mean()

        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.net.parameters(), self.grad_clip)
        self.optimizer.step()

        self.buffer.update_priorities(indices, errors.detach().cpu().numpy())

        self.steps += 1
        if self.steps % self.target_update_freq == 0:
            self.update_target_soft()

    def train(self) -> None:
        self.update_target_copy()
        now = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M")
        checkpoint = Checkpoint(os.path.join(self.ckpt_dir, f"{now}.pth"))
        with SummaryWriter(os.path.join(self.logs_dir, now)) as sw:
            self.populate()
            for e in range(self.n_episodes):
                self.agent.eps = self.epsilons[e]
                self.buffer.beta = self.betas[e]
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
                self.flush()

                self.rewards.append(rewards)
                score_avg = np.average(self.rewards[-self.avg_reward_len:])

                sw.add_scalar("score/val", rewards, global_step=e)
                sw.add_scalar("score/avg", score_avg, global_step=e)
                sw.add_scalar("episode/eps", self.agent.eps, global_step=e)
                sw.add_scalar("episode/beta", self.buffer.beta, global_step=e)
                sw.add_scalar("episode/steps", steps, global_step=e)

                if (e + 1) % self.eval_freq == 0:
                    score_eval = self.evaluate()
                    sw.add_scalar("score/eval", score_eval, global_step=e)
                    if checkpoint.checkpoint(self.net, score_eval):
                        logger.info("episode %d: saved %s at greedy score %.1f (train avg %.1f)",
                                    e, checkpoint.path, score_eval, score_avg)
                    if score_eval >= self.solved_score:
                        logger.info("episode %d: greedy score %.1f reached the %.1f target, stopping",
                                    e, score_eval, self.solved_score)
                        break
