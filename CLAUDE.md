# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

The project is uv-only. There is no conda env, no `requirements.txt`, and nothing to
activate — always go through `uv run`.

```bash
uv sync                                   # whole setup, provisions Python 3.13 + Box2D
uv run pytest test_rl.py -q               # all tests
uv run pytest test_rl.py -q -k n_step     # a single test by name
uv run test_rl.py                         # same checks without pytest (asserts + __main__)
uv run --with pyright pyright .           # type check; keep this at 0 errors
uv run main.py --help                     # every hyperparameter is a flag
uv run render.py                          # rollout the shipped checkpoint to LunarLander.mp4
uv run tensorboard --logdir logs
```

A short training run that reaches a solved policy in ~10 minutes on CPU:

```bash
uv run main.py --n_episodes 4000 --eps_ratio 0.35 --eps_final 0.01 \
               --learning_rate 5e-4 --warm_start 5000 --n_step 3 \
               --eval_episodes 20 --solved_score 275
```

## Architecture

`main.py` (argparse) constructs `DQNPER` and calls `train()`. `render.py` (click) loads a
checkpoint and replays it to MP4. Everything else lives in `rl/`.

`rl/dqnper.py` is the agent. It owns the episode loop, the n-step staging window,
greedy evaluation, and the learn step. `ReplayBuffer` is the storage base class;
`PEReplayBuffer` adds the priority tree on top. For a uniform-sampling ablation use
`PEReplayBuffer(alpha=0)` rather than reintroducing a second training loop.

Data flows: env step → `remember()` folds it into the n-step window → `PEReplayBuffer`
(flat numpy columns + `SumTree` of priorities) → `sample()` returns a batch plus IS
weights → `learn()` computes the Double-DQN target and writes TD errors back as
priorities.

### Invariants that fail silently if broken

These are the things that will not raise, will not fail a test you did not write, and
will quietly cost 100+ points of score.

- **`terminated` vs `truncated`.** Only `terminated` zeroes the bootstrap. A `truncated`
  episode (hit the 1000-step limit) still has value beyond the cut. The training loop
  ends on `terminated or truncated` but stores `terminated`.
- **The bootstrap discount is per transition, not `self.gamma`.** With n-step returns a
  transition bootstraps with `gamma ** k` where `k` is how many rewards were actually
  folded. That is why the buffers carry a `discounts` column and the target is
  `rewards + discounts * next_q_values`. Substituting `self.gamma` mis-scales every
  n-step target.
- **The n-step window must be flushed at every episode boundary** (`flush()`), and
  `_fold()` must stop at a terminal inside the window. Without the flush, the last
  `n_step - 1` transitions of each episode — the ones containing the touchdown — are
  never stored, which is exactly the data that teaches landing.
- **Checkpoints select on `evaluate()`, never on the training average.** Training
  episodes are epsilon-greedy, so their running mean measures exploration noise as much
  as policy quality — in both directions. It has saved a model reporting 239.8 that
  scored 160.6 greedy, and would have discarded one scoring 269.1 greedy whose training
  average read 78.2.
- **`evaluate()` runs on `self.eval_env`, a second environment.** Evaluating on the
  training env would perturb its RNG stream and break reproducibility.
- **Selection seeds and reporting seeds must stay disjoint.** `evaluate()` uses
  `seed + 500`; benchmark numbers in the README come from seeds 1000-1029. Reporting a
  score on the seeds you selected against is not a held-out number.

### Deliberate choices that look wrong

- `SumTree.tree` is a plain Python list, not an ndarray. Every access is a scalar on a
  root-to-leaf walk, and list indexing skips numpy's scalar boxing — measured ~2.5x
  faster for both `update` and `retrieve`.
- Replay buffers store flat numpy columns rather than tuples in a deque, so a sampled
  batch is contiguous and `torch.from_numpy` is zero-copy. Uniform sampling uses
  `randint`, not `np.random.choice(..., replace=False)`, which built a full permutation
  of the whole buffer on every step (934 → 6.7 us).
- Huber loss rather than MSE specifically because PER over-samples large TD errors, and
  squaring them lets one outlier dominate the batch gradient.
- `LinearRelu` in `rl/model.py` looks like pointless indirection but its removal changes
  `state_dict` keys and invalidates `models/lunarlander-v3.pth`.

## Artifacts

`ckpt/`, `logs/`, `*.pth` and `*.mp4` are gitignored build outputs, and `ckpt/` gets
wiped between training runs. The two committed artifacts are `models/lunarlander-v3.pth`
(the released checkpoint, un-ignored by a `!models/*.pth` negation, and the default for
`render.py`) and `docs/lunarlander.gif` (README demo; the README documents the
render + ffmpeg pipeline that regenerates it).

When a training change wins, regenerate both from the new weights rather than leaving
the README showing a policy that no longer exists.

## Conventions

- `test_rl.py` must keep working under both `pytest` and `python test_rl.py` — the
  `__main__` block calls every `test_*` in `globals()`, so a test taking a required
  argument other than pytest's optional `tmp_path` breaks the standalone runner.
- Claims about performance in the README are measured numbers. If you change training
  and quote a result, run the held-out evaluation rather than quoting the training log,
  and say when a comparison is a before/after rather than a controlled ablation.
