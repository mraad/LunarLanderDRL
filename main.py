import argparse
import logging
import os

from rl.dqnper import DQNPER


def _main_(args: argparse.Namespace) -> None:
    dqn = DQNPER(**vars(args))
    dqn.train()


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    arg_parser = argparse.ArgumentParser(
            description='Train a DQN with prioritized experience replay.',
            formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    arg_parser.add_argument(
            '-en',
            '--env_name',
            default="LunarLander-v3",
            help='Gym environment name.')
    arg_parser.add_argument(
            '-ne',
            '--n_episodes',
            type=int,
            default=100_000,
            help='Number of episodes.')
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
            '-er',
            '--eps_ratio',
            type=float,
            default=0.9,
            help='Ratio of episodes till final epsilon value.')
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
            '-rba',
            '--replay_buffer_alpha',
            type=float,
            default=0.6,
            help='Replay buffer alpha, 0 is uniform sampling and 1 is fully prioritized.')
    arg_parser.add_argument(
            '-rbbb',
            '--replay_buffer_beta_begin',
            type=float,
            default=0.4,
            help='Replay buffer beta begin.')
    arg_parser.add_argument(
            '-rbbf',
            '--replay_buffer_beta_final',
            type=float,
            default=1.0,
            help='Replay buffer beta final.')
    arg_parser.add_argument(
            '-rbbr',
            '--replay_buffer_beta_ratio',
            type=float,
            default=0.1,
            help='Ratio of episodes till final beta value.')
    arg_parser.add_argument(
            '-rbe',
            '--replay_buffer_eps',
            type=float,
            default=0.001,
            help='Replay buffer priority floor, keeps zero error transitions sampleable.')
    arg_parser.add_argument(
            '-tuf',
            '--target_update_freq',
            type=int,
            default=1,
            help='Steps between target network soft updates.')
    arg_parser.add_argument(
            '-tut',
            '--target_update_tau',
            type=float,
            default=1.0e-3,
            help='Target network soft update rate.')
    arg_parser.add_argument(
            '-gc',
            '--grad_clip',
            type=float,
            default=10.0,
            help='Max gradient norm, clipped before each optimizer step.')
    arg_parser.add_argument(
            '-ws',
            '--warm_start',
            type=int,
            default=1_000,
            help='Replay buffer warming size.')
    arg_parser.add_argument(
            '-arl',
            '--avg_reward_len',
            type=int,
            default=100,
            help='Average reward length.')
    arg_parser.add_argument(
            '-evf',
            '--eval_freq',
            type=int,
            default=25,
            help='Episodes between greedy evaluations.')
    arg_parser.add_argument(
            '-eve',
            '--eval_episodes',
            type=int,
            default=10,
            help='Greedy episodes per evaluation; the checkpoint is selected on their mean.')
    arg_parser.add_argument(
            '-ss',
            '--solved_score',
            type=float,
            default=260.0,
            help='Stop once the greedy evaluation mean reaches this.')
    arg_parser.add_argument(
            '-s',
            '--seed',
            type=int,
            default=42,
            help='Seed value.')
    arg_parser.add_argument(
            '--logs_dir',
            default=os.path.join(os.getcwd(), "logs"),
            help='Tensorboard logging directory.')
    arg_parser.add_argument(
            '--ckpt_dir',
            default=os.path.join(os.getcwd(), "ckpt"),
            help='Checkpoint directory.')
    _main_(arg_parser.parse_args())
