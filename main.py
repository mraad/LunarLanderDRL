import argparse
from rl.dqn import DQN


def _main_(args: argparse.Namespace) -> None:
    dqn = DQN(n_episodes=args.n_episodes,
              sync_rate=args.sync_rate)
    dqn.train()


if __name__ == '__main__':
    arg_parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    arg_parser.add_argument(
        '-ne',
        '--n_episodes',
        type=int,
        default=150_000,
        help='Number of episodes.')
    arg_parser.add_argument(
        '-sr',
        '--sync_rate',
        type=int,
        default=1_000,
        help='When to sync the network to target network.')
    _main_(arg_parser.parse_args())