#!/usr/bin/env python3
"""Offline BC policy evaluation: MSE between the trained policy's deterministic
(mode) action prediction and the actual recorded action, across every
non-idle transition in demo_data/*.pkl. No hardware needed -- this measures
fit to the demo data itself, not live rollout success.
"""

import glob
import os
import pickle as pkl

import jax
import numpy as np
from absl import app, flags
from flax.training import checkpoints

from experiments.mappings import CONFIG_MAPPING
from serl_launcher.utils.launcher import make_bc_agent

FLAGS = flags.FLAGS
flags.DEFINE_string("exp_name", None, "Name of experiment corresponding to folder.")
flags.DEFINE_string("bc_checkpoint_path", None, "Path to the trained BC checkpoint dir.")
flags.DEFINE_string("path", None, "Path to a single specific demo_data/*.pkl file. Takes precedence over --demo_path.")
flags.DEFINE_string("demo_path", None, "Glob for demo pkl files. Defaults to ./demo_data/*.pkl. Ignored if --path is set.")
flags.DEFINE_integer("batch_size", 256, "Forward-pass batch size (memory/speed only, doesn't affect the result).")
flags.DEFINE_integer("seed", 42, "Seed for BCAgent init and (if --single_transition) picking the random transition.")
flags.DEFINE_boolean(
    "single_transition", False,
    "Instead of averaging over the whole dataset, compute MSE at one random non-idle "
    "transition from the middle 50% of a random episode -- a quick spot-check rather "
    "than a full dataset metric."
)


def stack_obs(obs_list):
    # each obs is a pytree (nested dict) of arrays already shaped like
    # env.observation_space.sample() (incl. ChunkingWrapper's leading dim) --
    # stack a list of them into one batched pytree with a new leading axis.
    return jax.tree.map(lambda *xs: np.stack(xs, axis=0), *obs_list)


def split_into_episodes(transitions: list) -> list:
    """record_demos.py saves one flat list of transitions with all accepted
    episodes concatenated together -- split back into per-episode lists using
    each transition's `dones` flag. Must run on the UNFILTERED transition
    list: the idle-action filter can remove the very transition a `dones`
    flag lives on, which would silently merge two episodes together."""
    episodes, current = [], []
    for t in transitions:
        current.append(t)
        if t["dones"]:
            episodes.append(current)
            current = []
    if current:
        episodes.append(current)
    return episodes


def main(_):
    assert FLAGS.exp_name in CONFIG_MAPPING, "Experiment folder not found."
    assert FLAGS.bc_checkpoint_path, "--bc_checkpoint_path is required."
    config = CONFIG_MAPPING[FLAGS.exp_name]()
    env = config.get_environment(fake_env=True, save_video=False, classifier=True)

    bc_agent = make_bc_agent(
        seed=FLAGS.seed,
        sample_obs=env.observation_space.sample(),
        sample_action=env.action_space.sample(),
        image_keys=config.image_keys,
        encoder_type=config.encoder_type,
    )
    ckpt_dir = os.path.abspath(FLAGS.bc_checkpoint_path)
    # restore_checkpoint silently no-ops (returns the untouched target) if
    # nothing's found at this path -- that would mean evaluating a randomly
    # initialized head with no error at all, so check explicitly first.
    if checkpoints.latest_checkpoint(ckpt_dir) is None:
        raise FileNotFoundError(f"No checkpoint found at {ckpt_dir}")
    ckpt = checkpoints.restore_checkpoint(ckpt_dir, bc_agent.state)
    bc_agent = bc_agent.replace(state=ckpt)

    if FLAGS.path:
        paths = [FLAGS.path]
    else:
        demo_glob = FLAGS.demo_path or os.path.join(os.getcwd(), "demo_data", "*.pkl")
        paths = glob.glob(demo_glob)
    assert paths, f"No demo files found (path={FLAGS.path!r}, demo_path glob={FLAGS.demo_path!r})"

    rng = np.random.default_rng(FLAGS.seed)

    # per-file, unfiltered -- needed as-is for --single_transition's episode
    # splitting (see split_into_episodes), and flattened+filtered below
    # otherwise.
    file_transitions = []
    for path in paths:
        with open(path, "rb") as f:
            file_transitions.append((path, pkl.load(f)))

    if FLAGS.single_transition:
        path, raw = file_transitions[rng.integers(len(file_transitions))]
        episodes = split_into_episodes(raw)
        ep_i = rng.integers(len(episodes))
        episode = episodes[ep_i]

        lo, hi = int(len(episode) * 0.25), int(len(episode) * 0.75)
        middle_idxs = list(range(lo, hi)) if hi > lo else list(range(len(episode)))
        # indices, not the dicts themselves -- transitions hold numpy arrays,
        # and dict equality on those raises ("truth value of an array is
        # ambiguous"), so anything relying on list.index()/`in` must avoid it.
        non_idle_idxs = [i for i in middle_idxs if np.linalg.norm(episode[i]["actions"]) > 0.0]
        pool_idxs = non_idle_idxs if non_idle_idxs else middle_idxs  # fall back if the whole middle is idle
        step_i = int(pool_idxs[rng.integers(len(pool_idxs))])
        transitions = [episode[step_i]]

        print(
            f"Picked 1 transition: {os.path.basename(path)}, episode {ep_i}/{len(episodes) - 1}, "
            f"step {step_i}/{len(episode) - 1} (middle range: [{lo}, {hi}))"
        )
    else:
        transitions = []
        for _, raw in file_transitions:
            for t in raw:
                # same idle-step filter train_bc.py uses when building the
                # training replay buffer, so this MSE reflects the same data
                # the policy was actually trained on.
                if np.linalg.norm(t["actions"]) > 0.0:
                    transitions.append(t)
        print(f"Loaded {len(transitions)} non-idle transitions from {len(paths)} file(s)")

    tanh_squash = bc_agent.config["tanh_squash_distribution"]
    action_dim = env.action_space.shape[-1]
    
    lin_scale, ang_scale, gripper_scale = env.unwrapped.config.ACTION_SCALE
    action_scale = np.array([lin_scale] * 3 + [ang_scale] * 3)[:action_dim]
    dim_units = ["m/s"] * 3 + ["rad/s"] * 3

    total_se = np.zeros(action_dim, dtype=np.float64)  # per-dimension summed squared error
    n = 0
    last_pred_actions, last_target_actions = None, None
    for start in range(0, len(transitions), FLAGS.batch_size):
        batch = transitions[start:start + FLAGS.batch_size]
        obs = stack_obs([t["observations"] for t in batch])
        # Truncate to action_dim (env.action_space.shape[-1], from the
        # exp_name's config) -- some demo files have recorded actions wider
        # than the current config (e.g. a leftover gripper dim from a since-
        # reverted action space), and np.stack would otherwise raise on the
        # inconsistent shapes rather than silently misalign dimensions.
        target_actions = np.stack([np.asarray(t["actions"])[:action_dim] for t in batch])
        if tanh_squash:
            target_actions = np.clip(target_actions, -1 + 1e-6, 1 - 1e-6)

        dist = bc_agent.forward_policy(obs, temperature=1.0)
        pred_actions = np.asarray(jax.device_get(dist.mode()))

        total_se += ((pred_actions - target_actions) ** 2).sum(axis=0)
        n += len(batch)
        last_pred_actions, last_target_actions = pred_actions, target_actions

    per_dim_mse = total_se / n
    dim_labels = ["x", "y", "z", "rx", "ry", "rz"][:action_dim]

    print()
    if FLAGS.single_transition:
        print(f"Policy action:  {np.round(last_pred_actions[0], 4)}")
        print(f"Actual action:  {np.round(last_target_actions[0], 4)}")
        gap = (last_pred_actions[0] - last_target_actions[0]) * action_scale
        gap_lin = np.linalg.norm(gap[:3])
        gap_str = f"|linear|={gap_lin:.4f} m/s"
        if action_dim > 3:
            gap_str += f"  |angular|={np.linalg.norm(gap[3:6]):.4f} rad/s"
        print(f"Physical gap (policy vs actual): {gap_str}")
        print()
    print(f"Evaluated on {n} transitions")
    print("Per-dimension MSE (normalized units, matches training loss):")
    for label, mse in zip(dim_labels, per_dim_mse):
        print(f"  {label:>3s}: {mse:.6f}")
    print(f"Overall MSE (mean across action dims): {per_dim_mse.mean():.6f}")
    print(f"Overall MSE (summed across action dims, matches training's per-sample metric): {per_dim_mse.sum():.6f}")

    per_dim_rmse_physical = np.sqrt(per_dim_mse) * action_scale
    print()
    print("Per-dimension RMSE (physical units -- typical velocity error):")
    for label, rmse, unit in zip(dim_labels, per_dim_rmse_physical, dim_units):
        print(f"  {label:>3s}: {rmse:.4f} {unit}")


if __name__ == "__main__":
    app.run(main)
