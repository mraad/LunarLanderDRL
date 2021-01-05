from rl.dqn import DQN

if __name__ == '__main__':
    dqn = DQN(n_episodes=100, sync_rate=10)
    dqn.train()