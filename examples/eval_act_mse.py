#!/usr/bin/env python3
"""Offline ACT policy evaluation: MSE between ACT's predicted action (first
step of its predicted chunk) and the actual recorded action, across every
non-idle transition in demo_data/*.pkl. No hardware needed -- this measures
fit to the demo data itself, not live rollout success. ACT analog of
eval_bc_mse.py.
"""

import glob
import os
import pickle as pkl

import numpy as np
from absl import app, flags

from experiments.mappings import CONFIG_MAPPING

FLAGS = flags.FLAGS
flags.DEFINE_string("exp_name", None, "Name of experiment corresponding to folder (used only for ACTION_SCALE).")
flags.DEFINE_string("act_checkpoint", None, "Path to a trained LeRobot ACT checkpoint dir.")
flags.DEFINE_string("path", None, "Path to a single specific demo_data/*.pkl file. Takes precedence over --demo_path.")
flags.DEFINE_string("demo_path", None, "Glob for demo pkl files. Defaults to ./demo_data/*.pkl. Ignored if --path is set.")
flags.DEFINE_integer("batch_size", 64, "Forward-pass batch size (memory/speed only, doesn't affect the result).")
flags.DEFINE_integer("seed", 42, "Seed for picking the random transition (only used with --single_transition).")
flags.DEFINE_boolean(
    "single_transition", False,
    "Instead of averaging over the whole dataset, compute MSE at one random non-idle "
    "transition from the middle 50% of a random episode -- a quick spot-check rather "
    "than a full dataset metric."
)


def load_act_policy(checkpoint_path: str):
    """Loads a LeRobot ACT checkpoint plus its bundled pre/post-processor
    pipelines (normalization stats baked in at train time)."""
    from lerobot.policies.act.modeling_act import ACTPolicy
    from lerobot.policies import make_pre_post_processors

    policy = ACTPolicy.from_pretrained(checkpoint_path)
    policy.eval()
    preprocessor, postprocessor = make_pre_post_processors(policy.config, pretrained_path=checkpoint_path)

    # e.g. "observation.image.wrist" -> "wrist" -- our own recorded obs dict
    # keys cameras by their short HIL-SERL name (REALSENSE_SERIALS keys).
    image_key_map = {key: key.rsplit(".", 1)[-1] for key in policy.config.image_features}
    return policy, preprocessor, postprocessor, image_key_map


def build_act_obs_batch(image_key_map: dict, obs_list: list):
    """Converts a list of recorded obs dicts (each already shaped like
    env.observation_space.sample(), incl. ChunkingWrapper's leading dim) into
    one batched LeRobot input dict."""
    import torch

    states = np.stack([np.asarray(o["state"]).flatten() for o in obs_list]).astype(np.float32)
    lerobot_obs = {"observation.state": torch.from_numpy(states)}
    for lerobot_key, cam_name in image_key_map.items():
        imgs = []
        for o in obs_list:
            img = np.asarray(o[cam_name])
            if img.ndim == 4:
                img = img[0]  # drop ChunkingWrapper's leading dim
            imgs.append(img)
        imgs = np.stack(imgs).astype(np.float32) / 255.0  # (B,H,W,3) uint8 -> float [0,1]
        imgs = imgs.transpose(0, 3, 1, 2)  # -> (B,3,H,W)
        lerobot_obs[lerobot_key] = torch.from_numpy(imgs)
    return lerobot_obs


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
    assert FLAGS.act_checkpoint, "--act_checkpoint is required."
    import torch

    config = CONFIG_MAPPING[FLAGS.exp_name]()
    env = config.get_environment(fake_env=True, save_video=False, classifier=True)

    policy, preprocessor, postprocessor, image_key_map = load_act_policy(FLAGS.act_checkpoint)

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
                # same idle-step filter train_bc.py/eval_bc_mse.py use, so
                # this MSE reflects the same non-idle data distribution.
                if np.linalg.norm(t["actions"]) > 0.0:
                    transitions.append(t)
        print(f"Loaded {len(transitions)} non-idle transitions from {len(paths)} file(s)")

    action_dim = env.action_space.shape[-1]

    # Actions are recorded in normalized [-1,1]-ish units (physical_velocity /
    # ACTION_SCALE), so MSE on its own is dimensionless. ACTION_SCALE converts
    # back: since it's a single positive scalar per 3-axis block (linear,
    # angular), it commutes with both differencing and RMS, so
    # sqrt(per_dim_mse) * action_scale is exactly the RMS physical velocity
    # error (m/s / rad/s), no approximation.
    lin_scale, ang_scale = env.unwrapped.config.ACTION_SCALE
    action_scale = np.array([lin_scale] * 3 + [ang_scale] * 3)[:action_dim]
    dim_units = ["m/s"] * 3 + ["rad/s"] * 3

    total_se = np.zeros(action_dim, dtype=np.float64)  # per-dimension summed squared error
    n = 0
    last_pred_actions, last_target_actions = None, None
    for start in range(0, len(transitions), FLAGS.batch_size):
        batch = transitions[start:start + FLAGS.batch_size]
        lerobot_obs = preprocessor(build_act_obs_batch(image_key_map, [t["observations"] for t in batch]))
        target_actions = np.stack([np.asarray(t["actions"]) for t in batch])

        with torch.no_grad():
            chunk = policy.predict_action_chunk(lerobot_obs)  # (B, chunk_size, action_dim)
        pred_actions = postprocessor(chunk[:, 0, :]).numpy()  # first step of the predicted chunk

        total_se += ((pred_actions - target_actions) ** 2).sum(axis=0)
        n += len(batch)
        last_pred_actions, last_target_actions = pred_actions, target_actions

    per_dim_mse = total_se / n
    dim_labels = ["x", "y", "z", "rx", "ry", "rz"][:action_dim]

    print()
    if FLAGS.single_transition:
        print(f"ACT action:    {np.round(last_pred_actions[0], 4)}")
        print(f"Actual action: {np.round(last_target_actions[0], 4)}")
        gap = (last_pred_actions[0] - last_target_actions[0]) * action_scale
        gap_lin = np.linalg.norm(gap[:3])
        gap_str = f"|linear|={gap_lin:.4f} m/s"
        if action_dim > 3:
            gap_str += f"  |angular|={np.linalg.norm(gap[3:6]):.4f} rad/s"
        print(f"Physical gap (ACT vs actual): {gap_str}")
        print()
    print(f"Evaluated on {n} transitions")
    print("Per-dimension MSE (normalized units):")
    for label, mse in zip(dim_labels, per_dim_mse):
        print(f"  {label:>3s}: {mse:.6f}")
    print(f"Overall MSE (mean across action dims): {per_dim_mse.mean():.6f}")
    print(f"Overall MSE (summed across action dims): {per_dim_mse.sum():.6f}")

    per_dim_rmse_physical = np.sqrt(per_dim_mse) * action_scale
    print()
    print("Per-dimension RMSE (physical units -- typical velocity error):")
    for label, rmse, unit in zip(dim_labels, per_dim_rmse_physical, dim_units):
        print(f"  {label:>3s}: {rmse:.4f} {unit}")


if __name__ == "__main__":
    app.run(main)
