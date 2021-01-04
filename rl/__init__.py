# from typing import Any
#
# import numpy as np
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import torch.optim as optim
# from gym import Env
# from tqdm import tqdm_notebook as tqdm
#
#
# def decay_schedule(
#         start: float,
#         final: float,
#         decay: float,
#         steps: int,
#         log_start: float = -2.0) -> np.array:
#     decay_steps = int(steps * decay)
#     values = np.logspace(log_start, 0, decay_steps)[::-1]
#     v_min = values.min()
#     v_max = values.max()
#     values = (values - v_min) / (v_max - v_min)
#     values = final + (start - final) * values
#     pad = steps - decay_steps
#     values = np.pad(values, (0, pad), 'edge')
#     return values
#
#
# class HParams:
#     def __init__(self,
#                  lr: float = 0.0001,
#                  n_states: int = 8,
#                  n_actions: int = 4,
#                  gamma: float = 0.99,
#                  batch_size: int = 64,
#                  eps_start: float = 1.0,
#                  eps_final: float = 0.01,
#                  eps_decay: float = 0.1,
#                  mem_size: int = 100_000
#                  ) -> None:
#         self.lr = lr
#         self.n_states = n_states
#         self.n_actions = n_actions
#         self.gamma = gamma
#         self.batch_size = batch_size
#         self.eps_start = eps_start
#         self.eps_final = eps_final
#         self.eps_decay = eps_decay
#         self.mem_size = mem_size
#
#     def to_md(self) -> str:
#         arr = []
#
#         def append(key: str, val: Any) -> None:
#             arr.append(f"|{key}|{val}|")
#
#         append("Parameter", "Value")
#         append("--", "--")
#         append("lr", self.lr)
#         append("n_states", self.n_states)
#         append("n_actions", self.n_actions)
#         append("gamma", self.gamma)
#         append("batch_size", self.batch_size)
#         append("eps_start", self.eps_start)
#         append("eps_final", self.eps_final)
#         append("eps_decay", self.eps_decay)
#         append("mem_size", self.mem_size)
#
#         return "\n".join(arr)
#
#
# class DQN(nn.Module):
#     def __init__(self, hparams: HParams) -> None:
#         super(DQN, self).__init__()
#         self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
#         self.fc1 = nn.Linear(hparams.n_states, 256)
#         self.fc2 = nn.Linear(256, 128)
#         self.fc3 = nn.Linear(128, 64)
#         self.fc4 = nn.Linear(64, hparams.n_actions)
#         self.optimizer = optim.Adam(self.parameters(), lr=hparams.lr)
#         self.loss = nn.MSELoss()
#         self.to(self.device)
#
#     def forward(self, x):
#         x = F.relu(self.fc1(x))
#         x = F.relu(self.fc2(x))
#         x = F.relu(self.fc3(x))
#         return self.fc4(x)
#
#
# class Agent:
#     def __init__(self, hparams: HParams) -> None:
#         self.hparams = hparams
#         self.eps = hparams.eps_start
#         self.action_space = [i for i in range(hparams.n_actions)]
#         self.mem_index = 0
#         self.old_state_mem = np.zeros((hparams.mem_size, hparams.n_states), dtype=np.float32)
#         self.new_state_mem = np.zeros((hparams.mem_size, hparams.n_states), dtype=np.float32)
#         self.action_mem = np.zeros(hparams.mem_size, dtype=np.int32)
#         self.reward_mem = np.zeros(hparams.mem_size, dtype=np.float32)
#         self.done_mem = np.zeros(hparams.mem_size, dtype=np.bool)
#         self.dqm = DQN(hparams)
#
#     def store_transition(self, old_state, action, new_state, reward, done):
#         mem_index = self.mem_index % self.hparams.mem_size
#         self.old_state_mem[mem_index] = old_state
#         self.new_state_mem[mem_index] = new_state
#         self.action_mem[mem_index] = action
#         self.reward_mem[mem_index] = reward
#         self.done_mem[mem_index] = done
#         self.mem_index += 1
#
#     def select_action(self, np_state):
#         if np.random.random() > self.eps:
#             with torch.no_grad():
#                 pt_state = torch.tensor(np_state, device=self.dqm.device)
#                 actions = self.dqm(pt_state)
#                 action = torch.argmax(actions).item()
#         else:
#             action = np.random.choice(self.action_space)
#         return action
#
#     def learn(self):
#         if self.mem_index < self.hparams.batch_size:
#             return
#
#         mem_index = min(self.mem_index, self.hparams.batch_size)
#         batch = np.random.choice(mem_index, self.hparams.batch_size, replace=False)
#
#         old_state = torch.tensor(self.old_state_mem[batch], device=self.dqm.device)
#         new_state = torch.tensor(self.new_state_mem[batch], device=self.dqm.device)
#         reward = torch.tensor(self.reward_mem[batch], device=self.dqm.device)
#         done = torch.tensor(self.done_mem[batch], device=self.dqm.device)
#         gamma = self.hparams.gamma * (~done)
#
#         index = np.arange(mem_index, dtype=np.int32)
#         action = self.action_mem[batch]
#
#         q_eval = self.dqm(old_state)[index, action]
#         q_next = self.dqm(new_state)
#
#         q_target = reward + gamma * torch.max(q_next, axis=1)[0]
#         loss = self.dqm.loss(q_target, q_eval)
#
#         self.dqm.optimizer.zero_grad()
#         loss.backward()
#         self.dqm.optimizer.step()
#
#
# def q_learning(env: Env,
#                gamma: float = 1.0,
#                lr_start: float = 0.5,
#                lr_final: float = 0.01,
#                lr_decay: float = 0.5,
#                eps_start: float = 1.0,
#                eps_final: float = 0.1,
#                eps_decay: float = 0.9,
#                episodes: int = 3000):
#     ns = env.observation_space.n
#     na = env.action_space.n
#
#     state = 0
#     pi_hist = []
#
#     q = np.zeros((ns, na), dtype=np.float64)
#     q_hist = np.zeros((episodes, ns, na), dtype=np.float64)
#
#     def select_action(eps: float) -> int:
#         if np.random.random() > eps:
#             a = np.argmax(q[state])
#         else:
#             a = np.random.randint(na)
#         return a
#
#     decay_lr = decay_schedule(lr_start, lr_final, lr_decay, episodes)
#     decay_eps = decay_schedule(eps_start, eps_final, eps_decay, episodes)
#
#     for e in tqdm(range(episodes), leave=False):
#         state = env.reset()
#         done = False
#         while not done:
#             action = select_action(decay_eps[e])
#             next_state, reward, done, _ = env.step(action)
#             td_target = reward + gamma * q[next_state].max() * (not done)
#             td_error = td_target - q[state][action]
#             q[state][action] = q[state][action] + decay_lr[e] * td_error
#             state = next_state
#         q_hist[e] = q
#         pi_hist.append(np.argmax(q, axis=1))
#
#     v = np.max(q, axis=1)
#     pi = lambda s: {s: a for s, a in enumerate(np.argmax(q, axis=1))}[s]
#
#     return q, v, pi, q_hist, pi_hist