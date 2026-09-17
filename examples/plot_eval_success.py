#!/usr/bin/env python3
"""Aggregate eval_results.jsonl (written by train_rlpd.py's --eval_checkpoint_step
path) across however many checkpoints you've evaluated, and plot a sliding-window
average of success rate vs. checkpoint step.

Usage:
    python plot_eval_success.py --checkpoint_path experiments/flexiv_task/first_run
    python plot_eval_success.py --checkpoint_path .../first_run --window 5
    python plot_eval_success.py --checkpoint_path .../first_run --out success_curve.png

Run this after collecting results for multiple checkpoints via e.g.:
    ./run_actor.sh --eval_checkpoint_step=5000  --eval_n_trajs=20 --deterministic_eval
    ./run_actor.sh --eval_checkpoint_step=10000 --eval_n_trajs=20 --deterministic_eval
    ...
each of which appends one line to <checkpoint_path>/eval_results.jsonl.
"""

import json
import os

import matplotlib
matplotlib.use("Agg")  # offscreen rendering only, no GUI backend needed
import matplotlib.pyplot as plt
import numpy as np
from absl import app, flags

FLAGS = flags.FLAGS
flags.DEFINE_string("checkpoint_path", None, "Same --checkpoint_path used for training/eval; eval_results.jsonl lives under it.", required=True)
flags.DEFINE_integer("window", 3, "Sliding window size, in number of evaluated checkpoints (not step count).")
flags.DEFINE_string("out", None, "Output plot path. Defaults to <checkpoint_path>/eval_success.png.")


def _moving_average(values: np.ndarray, window: int) -> np.ndarray:
    """Centered moving average with shrinking window at the edges (no NaNs,
    no assumption that `window` evenly divides len(values))."""
    n = len(values)
    out = np.empty(n, dtype=np.float64)
    half = window // 2
    right_extra = 1 if window % 2 == 1 else 0  # odd window: symmetric + center; even: one-sided
    for i in range(n):
        lo = max(0, i - half)
        hi = min(n, i + half + right_extra)
        out[i] = values[lo:hi].mean()
    return out


def main(_):
    results_path = os.path.join(os.path.abspath(FLAGS.checkpoint_path), "eval_results.jsonl")
    assert os.path.isfile(results_path), f"No eval_results.jsonl found at {results_path} -- run some --eval_checkpoint_step evals first."

    rows = []
    with open(results_path, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))

    # Collapse duplicate steps (re-evaluating the same checkpoint) by
    # averaging their success rates -- each unique step becomes one point in
    # the checkpoint-index sequence the sliding window operates over.
    by_step = {}
    for r in rows:
        by_step.setdefault(r["step"], []).append(r)

    steps = sorted(by_step.keys())
    success_rates = np.array([
        np.mean([r["success_rate"] for r in by_step[s]]) for s in steps
    ])
    n_runs_per_step = [len(by_step[s]) for s in steps]

    if len(steps) < 2:
        print(f"Only {len(steps)} unique checkpoint(s) evaluated so far -- need at least 2 for a meaningful trend.")

    smoothed = _moving_average(success_rates, FLAGS.window)

    print(f"{'step':>10} {'n_evals':>8} {'success_rate':>13} {'smoothed':>10}")
    for s, n_evals, sr, sm in zip(steps, n_runs_per_step, success_rates, smoothed):
        print(f"{s:>10} {n_evals:>8} {sr:>13.3f} {sm:>10.3f}")

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(steps, success_rates, "o", alpha=0.4, label="per-checkpoint success rate")
    ax.plot(steps, smoothed, "-", linewidth=2, label=f"sliding window avg (window={FLAGS.window}, by checkpoint index)")
    ax.set_xlabel("checkpoint step")
    ax.set_ylabel("success rate")
    ax.set_ylim(-0.05, 1.05)
    ax.set_title("Eval success rate over training")
    ax.legend()
    ax.grid(alpha=0.3)

    out_path = FLAGS.out or os.path.join(os.path.abspath(FLAGS.checkpoint_path), "eval_success.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"saved plot to {out_path}")


if __name__ == "__main__":
    app.run(main)
