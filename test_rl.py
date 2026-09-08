"""Self-checks: schedules, replay buffers, the sum tree, and a short end-to-end train.

Run with ``uv run pytest test_rl.py`` or ``uv run test_rl.py``.
"""
import numpy as np

from rl.replaybuffer import PEReplayBuffer, ReplayBuffer
from rl.sumtree import SumTree
from rl.utils import dec_schedule, inc_schedule


def test_sum_tree_sums_and_retrieves() -> None:
    tree = SumTree(4)
    for leaf, priority in enumerate([1.0, 2.0, 3.0, 4.0]):
        tree.update(leaf, priority)
    assert tree.sum_priorities == 10.0
    assert [tree.priority(i) for i in range(4)] == [1.0, 2.0, 3.0, 4.0]

    # Cumulative intervals: [0,1) -> 0, [1,3) -> 1, [3,6) -> 2, [6,10) -> 3.
    assert [tree.retrieve(s) for s in (0.5, 2.0, 4.0, 9.0)] == [0, 1, 2, 3]

    tree.update(0, 5.0)
    assert tree.sum_priorities == 14.0
    assert tree.retrieve(0.5) == 0
    assert tree.retrieve(6.0) == 1


def test_sum_tree_handles_odd_capacity() -> None:
    tree = SumTree(5)
    for leaf in range(5):
        tree.update(leaf, 1.0)
    assert tree.sum_priorities == 5.0
    counts = np.bincount([tree.retrieve(s) for s in np.arange(0.05, 5.0, 0.1)], minlength=5)
    assert (counts == 10).all(), counts


def _fill(buffer: ReplayBuffer, n: int) -> None:
    for i in range(n):
        buffer.append(np.full(3, i, dtype=np.float32), i % 2, float(i), i % 5 == 0,
                      np.full(3, -i, dtype=np.float32))


def test_replay_buffer_wraps_and_samples() -> None:
    buffer = ReplayBuffer(capacity=4, batch_size=2, n_states=3)
    _fill(buffer, 6)
    assert len(buffer) == 4
    # Slots 0 and 1 were overwritten by transitions 4 and 5.
    assert sorted(buffer.rewards.tolist()) == [2.0, 3.0, 4.0, 5.0]

    states, actions, rewards, terminals, next_states = buffer.sample()
    assert states.shape == (2, 3) and next_states.shape == (2, 3)
    assert actions.shape == rewards.shape == terminals.shape == (2,)
    assert states.dtype == np.float32 and actions.dtype == np.int64
    # Columns stay aligned: state == reward and next_state == -reward.
    assert (states[:, 0] == rewards).all()
    assert (next_states[:, 0] == -rewards).all()


def test_per_never_samples_unwritten_slots() -> None:
    buffer = PEReplayBuffer(capacity=1024, batch_size=32, n_states=3)
    _fill(buffer, 40)
    for _ in range(50):
        _, _, rewards, _, _, indices, weights = buffer.sample()
        assert (indices < buffer.size).all(), indices
        assert (rewards < 40.0).all()
        assert weights.max() == 1.0 and (weights > 0.0).all()


def test_per_favours_high_priority_transitions() -> None:
    np.random.seed(0)
    buffer = PEReplayBuffer(capacity=64, batch_size=8, n_states=3, alpha=1.0, eps=0.0)
    _fill(buffer, 64)
    # Slot 7 gets a large TD error, everything else a tiny one.
    errors = np.full(64, 0.01)
    errors[7] = 100.0
    buffer.update_priorities(np.arange(64), errors)

    counts = np.bincount(np.concatenate([buffer.sample()[5] for _ in range(20)]), minlength=64)
    assert counts[7] > 0.5 * counts.sum(), counts[7]

    # A newly appended transition inherits the highest priority seen, so it is replayed.
    buffer.append(np.zeros(3, np.float32), 0, 0.0, False, np.zeros(3, np.float32))
    assert buffer.tree.priority(0) == buffer.max_priority >= 100.0


def test_per_weights_are_inverse_to_priority() -> None:
    buffer = PEReplayBuffer(capacity=8, batch_size=8, n_states=3, alpha=1.0, eps=0.0, beta=1.0)
    _fill(buffer, 8)
    buffer.update_priorities(np.arange(8), np.arange(1.0, 9.0))
    _, _, _, _, _, indices, weights = buffer.sample()
    # w = (N * p / sum_p) ** -beta, so the rarest sample carries the largest weight.
    order = np.argsort(indices)
    assert (np.diff(weights[order]) <= 1e-6).all(), weights[order]


def test_schedules_span_the_range_then_hold() -> None:
    dec = dec_schedule(1.0, 0.1, 0.5, 100)
    assert len(dec) == 100
    assert np.isclose(dec[0], 1.0) and np.isclose(dec[-1], 0.1)
    assert (np.diff(dec) <= 1e-12).all()
    assert (dec[50:] == 0.1).all()

    inc = inc_schedule(0.4, 1.0, 0.25, 100)
    assert len(inc) == 100
    assert np.isclose(inc[0], 0.4) and np.isclose(inc[-1], 1.0)
    assert (np.diff(inc) >= -1e-12).all()
    assert (inc[25:] == 1.0).all()

    # ratio of 1 leaves no padding, and ratio above 1 must not produce a negative pad.
    assert len(dec_schedule(1.0, 0.0, 1.0, 50)) == 50
    assert len(inc_schedule(0.0, 1.0, 2.0, 50)) == 50


def test_training_runs_end_to_end(tmp_path=None) -> None:
    """Exercises the Gymnasium 5-tuple step API through a real environment."""
    import tempfile

    import torch

    from rl.dqnper import DQNPER

    out = tmp_path or tempfile.mkdtemp()
    dqn = DQNPER(n_episodes=3, warm_start=200, batch_size=32, replay_buffer_size=1000,
                 logs_dir=f"{out}/logs", ckpt_dir=f"{out}/ckpt")
    assert (dqn.n_states, dqn.n_actions) == (8, 4)
    dqn.train()
    assert len(dqn.rewards) == 3 and np.isfinite(dqn.rewards).all()
    assert dqn.steps > 0
    # Polyak averaging must have nudged the target off the copy it started from.
    key = "net.0.linear.weight"
    assert not torch.allclose(dqn.net.state_dict()[key], dqn.net_target.state_dict()[key])


if __name__ == '__main__':
    for name, test in sorted(globals().items()):
        if name.startswith("test_"):
            test()
            print(f"ok {name}")
