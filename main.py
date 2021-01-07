import argparse
import logging
from rl.dqn import DQN


def _main_(args: argparse.Namespace) -> None:
    dqn = DQN(**vars(args))
    dqn.train()


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    arg_parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    arg_parser.add_argument(
        '-eb',
        '--eps_begin',
        type=float,
        default=1.0,
        help='Epsilon beginning value.')
    arg_parser.add_argument(
        '-ef',
        '--eps_final',
        type=float,
        default=0.1,
        help='Epsilon final value.')
    arg_parser.add_argument(
        '-ed',
        '--eps_decay',
        type=float,
        default=0.9,
        help='Epsilon decay ratio.')
    arg_parser.add_argument(
        '-ne',
        '--n_episodes',
        type=int,
        default=100_000,
        help='Number of episodes.')
    arg_parser.add_argument(
        '-sr',
        '--sync_rate',
        type=int,
        default=500,
        help='When to sync target network with the main network.')
    arg_parser.add_argument(
        '-g',
        '--gamma',
        type=float,
        default=0.99,
        help='Reward discount rate.')
    arg_parser.add_argument(
        '-lr',
        '--learning_rate',
        type=float,
        default=1.0e-3,
        help='Learning rate.')
    arg_parser.add_argument(
        '-bs',
        '--batch_size',
        type=int,
        default=64,
        help='Batch size.')
    arg_parser.add_argument(
        '-rbs',
        '--replay_buffer_size',
        type=int,
        default=100_000,
        help='Replay buffer size.')
    arg_parser.add_argument(
        '-ws',
        '--warm_start',
        type=int,
        default=1_000,
        help='Warming size.')
    arg_parser.add_argument(
        '-arl',
        '--avg_reward_len',
        type=int,
        default=100,
        help='Average reward length.')
    arg_parser.add_argument(
        '-s',
        '--seed',
        type=int,
        default=42,
        help='Seed value.')
    arg_parser.add_argument(
        '--logdir',
        default="logs",
        help='Tensorboard logging directory.')
    _main_(arg_parser.parse_args())