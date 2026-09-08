from typing import Tuple

import gymnasium as gym
import numpy as np


def env_shape(env: gym.Env) -> Tuple[int, int]:
    """Observation and action counts, asserting the spaces this agent actually handles.

    A DQN needs one Q value per action, so a continuous action space (``LunarLander``
    has a `Continuous` variant) would fail far from here, or worse, not at all.
    """
    assert isinstance(env.observation_space, gym.spaces.Box), env.observation_space
    assert isinstance(env.action_space, gym.spaces.Discrete), env.action_space
    return int(env.observation_space.shape[0]), int(env.action_space.n)


def _log_ramp(ratio: float,
              steps: int,
              log_start: float
              ) -> Tuple[np.ndarray, int]:
    """Log spaced ramp from 0 to 1 over ``ratio * steps`` values, plus the padding left over."""
    assert steps > 2, "schedule steps should be greater than 2"
    ramp_steps = max(2, min(steps, int(steps * ratio)))
    values = np.logspace(log_start, 0.0, ramp_steps)
    values = (values - values.min()) / (values.max() - values.min())
    return values, steps - ramp_steps


def dec_schedule(start: float,
                 final: float,
                 ratio: float,
                 steps: int,
                 log_start: float = -2.0
                 ) -> np.ndarray:
    """Decay ``start`` to ``final`` over ``ratio * steps``, then hold ``final``."""
    values, pad = _log_ramp(ratio, steps, log_start)
    return np.pad(final + (start - final) * values[::-1], (0, pad), 'edge')


def inc_schedule(start: float,
                 final: float,
                 ratio: float,
                 steps: int,
                 log_start: float = -2.0
                 ) -> np.ndarray:
    """Grow ``start`` to ``final`` over ``ratio * steps``, then hold ``final``."""
    values, pad = _log_ramp(ratio, steps, log_start)
    return np.pad(start + (final - start) * values, (0, pad), 'edge')
