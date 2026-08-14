#!/usr/bin/env python3
"""ACT (lerobot's Action Chunking Transformer) counterpart to train_bc.py.

Wired the same way as train_bc.py: same --exp_name/CONFIG_MAPPING env setup,
same demo_data/*.pkl source, same "hold SHIFT to start" live-eval loop, same
overall flag surface and train()/eval()/main() structure -- but the model and
training loop are pure PyTorch (lerobot's ACTPolicy), not JAX/flax's BCAgent.

This reads hil-serl's demo_data/*.pkl format (the same transitions train_bc.py
consumes), NOT the LeRobotDataset format that collect_demos/record_demos.py
writes today. If your demos live in a LeRobotDataset instead, training ACT on
those needs no new code -- run:
    collect_demos/train_me.sh <dataset_repo_id> <output_dir> act
which already shells out to lerobot's own training CLI. This script exists so
ACT can be trained/evaluated through the same harness as train_bc.py, for a
direct comparison against the BC baseline on identical data.

Note on chunking vs train_bc.py's idle-transition filter: BC treats
transitions as i.i.d. samples, so it can drop near-zero-action transitions
outright. ACT needs whole contiguous action sequences, so this script only
uses the idle filter to decide which timesteps are worth anchoring a training
sample on -- the action chunk pulled from a kept anchor is always the true,
unfiltered sequence that followed it in the episode.
"""

from __future__ import annotations

import glob
import os
import pickle as pkl
import shutil
import time
from typing import Dict, List

import numpy as np
import torch
import tqdm
from absl import app, flags
from gymnasium.wrappers import RecordEpisodeStatistics
from torch.utils.data import DataLoader, Dataset

from lerobot.configs.types import FeatureType, PolicyFeature
from lerobot.policies.act.configuration_act import ACTConfig
from lerobot.policies.act.modeling_act import ACTPolicy
from lerobot.policies.factory import make_pre_post_processors

from experiments.mappings import CONFIG_MAPPING
from experiments.config import DefaultTrainingConfig

FLAGS = flags.FLAGS

flags.DEFINE_string("exp_name", None, "Name of experiment corresponding to folder.")
flags.DEFINE_integer("seed", 42, "Random seed.")
flags.DEFINE_string("checkpoint_path", None, "Path to save checkpoints.")
flags.DEFINE_integer("eval_n_trajs", 0, "Number of trajectories to evaluate.")
flags.DEFINE_integer("train_steps", 20_000, "Number of pretraining steps.")
flags.DEFINE_bool("save_video", False, "Save video of the evaluation.")

flags.DEFINE_boolean(
    "debug", False, "Debug mode."
)  # debug mode will disable wandb logging

flags.DEFINE_integer(
    "batch_size", 8,
    "Training batch size. Much smaller than train_bc.py's default (256) -- "
    "ACT's transformer plus per-sample action chunks are far heavier than BC's MLP.",
)
flags.DEFINE_integer("chunk_size", 32, "Action chunk length ACT predicts per query.")
flags.DEFINE_integer(
    "save_interval", 1000,
    "Save a checkpoint every N training steps, in addition to a final checkpoint "
    "at the very end. Previously checkpoints only saved in the last 100 steps -- "
    "a crash/interrupt earlier in training lost all progress."
)
flags.DEFINE_integer(
    "resume_step", 0,
    "Resume training from checkpoint_<resume_step> under --checkpoint_path -- loads "
    "policy weights, the preprocessor/postprocessor's normalization stats (from that "
    "checkpoint, NOT recomputed from --demo_path -- keeping the model's normalization "
    "fixed across a DAgger-style continuation onto new/aggregated data is what actually "
    "matters, not matching whatever's in this run's demo file), and the optimizer's "
    "AdamW state (if saved), then continues stepping up to --train_steps total. "
    "0 = train from scratch (default)."
)
flags.DEFINE_integer(
    "keep_checkpoints", 5,
    "Keep only the N most recent checkpoint_* dirs under --checkpoint_path, deleting "
    "older ones after each save -- each one holds a full ACT policy (incl. ResNet18 "
    "backbone) plus pre/post processors, so these add up fast at --save_interval cadence. "
    "Set to 0 to disable pruning and keep everything."
)
flags.DEFINE_integer(
    "n_action_steps", 16,
    "Number of actions from each predicted chunk to actually execute before "
    "re-querying the policy for a fresh chunk. Must be <= chunk_size."
)
flags.DEFINE_float("optimizer_lr", 1e-5, "AdamW learning rate for ACT's transformer (everything except the vision backbone).")
flags.DEFINE_float(
    "optimizer_lr_backbone", None,
    "AdamW learning rate for the vision backbone specifically -- it's fine-tuned "
    "jointly, not frozen, so this is a real, separate knob (ACTPolicy.get_optim_params() "
    "puts backbone params in their own param group with this override). Defaults to "
    "--optimizer_lr's value if unset, matching ACTConfig's own default of using the same "
    "rate for both. Lower than --optimizer_lr is a common choice to avoid overwriting "
    "useful ImageNet-pretrained features early in training, especially with a small dataset."
)
flags.DEFINE_integer(
    "dim_model", 256,
    "Transformer main hidden dimension. ACTConfig's own default (512) is tuned for "
    "bimanual ALOHA fusing 3-4 camera views; smaller here since this setup has a single "
    "camera and a single 6-DOF arm to fuse, meaningfully less than that default was sized for."
)
flags.DEFINE_integer(
    "dim_feedforward", 1024,
    "Transformer feed-forward hidden dimension (expanded width inside each block's MLP, "
    "between attention sub-layers). Keeps roughly the same ~4x expansion ratio over "
    "--dim_model that ACTConfig's own default (3200 over dim_model=512, ~6.25x) uses."
)
flags.DEFINE_string(
    "demo_path", None,
    "Glob pattern (or single file path) for demo .pkl(s) to train on. "
    "Defaults to demo_data/*.pkl under the current working directory."
)

IMAGE_KEY_PREFIX = "observation.image."


def print_green(x):
    return print("\033[92m {}\033[00m".format(x))


def print_yellow(x):
    return print("\033[93m {}\033[00m".format(x))


def image_feature_key(camera_name: str) -> str:
    return f"{IMAGE_KEY_PREFIX}{camera_name}"


def make_wandb_logger(project: str, description: str, debug: bool):
    """Minimal, pure-Python stand-in for serl_launcher.utils.launcher.make_wandb_logger
    (that one pulls in jax/flax/agentlace as a side effect of import, which this
    script otherwise avoids entirely)."""
    if debug:
        return None
    import wandb

    wandb.init(project=project, name=description, tags=[description])
    return wandb


# ---------------------------------------------------------------------------
# Demo loading / chunked dataset
# ---------------------------------------------------------------------------


def _load_episodes(demo_paths: List[str]) -> List[List[dict]]:
    """Loads hil-serl's demo_data/*.pkl transition lists and splits each into
    per-episode runs (on `dones`), since ACT needs temporally contiguous
    action chunks, unlike BC's i.i.d. per-transition replay buffer."""
    episodes = []
    for path in demo_paths:
        with open(path, "rb") as f:
            transitions = pkl.load(f)
        current = []
        for transition in transitions:
            current.append(transition)
            if transition["dones"]:
                episodes.append(current)
                current = []
        if current:
            episodes.append(current)
    return episodes


class ACTChunkDataset(Dataset):
    """Turns hil-serl episodes into (observation, action_chunk) samples for ACT.

    Each sample anchors on one timestep's observation and pulls the true,
    contiguous `chunk_size` actions that follow it, padding past the episode's
    end by repeating its final action (masked out via `action_is_pad` --
    ACT's loss always ignores padded positions, so the pad value itself
    doesn't matter). Anchors with a near-zero action are skipped, mirroring
    train_bc.py's idle filter.
    """

    def __init__(
        self,
        episodes: List[List[dict]],
        image_keys: List[str],
        act_image_keys: Dict[str, str],
        chunk_size: int,
    ):
        self.episodes = episodes
        self.image_keys = image_keys
        self.act_image_keys = act_image_keys
        self.chunk_size = chunk_size
        self.samples = []  # (episode_idx, t)
        for ep_idx, episode in enumerate(episodes):
            for t, transition in enumerate(episode):
                if np.linalg.norm(transition["actions"]) > 0.0:
                    self.samples.append((ep_idx, t))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        ep_idx, t = self.samples[idx]
        episode = self.episodes[ep_idx]
        obs = episode[t]["observations"]

        item = {
            "observation.state": torch.from_numpy(
                np.asarray(obs["state"], dtype=np.float32).reshape(-1)
            )
        }
        for camera in self.image_keys:
            img_hwc = np.asarray(obs[camera], dtype=np.uint8)
            img_hwc = img_hwc.reshape(*img_hwc.shape[-3:])  # squeeze off obs-stacking dim
            image = torch.from_numpy(img_hwc).permute(2, 0, 1).float() / 255.0
            item[self.act_image_keys[camera]] = image

        chunk = np.stack(
            [episode[min(t + i, len(episode) - 1)]["actions"] for i in range(self.chunk_size)]
        ).astype(np.float32)
        is_pad = np.array([t + i >= len(episode) for i in range(self.chunk_size)], dtype=bool)
        item["action"] = torch.from_numpy(chunk)
        item["action_is_pad"] = torch.from_numpy(is_pad)
        return item


def compute_dataset_stats(
    episodes: List[List[dict]], image_keys: List[str], act_image_keys: Dict[str, str]
) -> dict:
    """Mean/std stats for ACT's Normalizer, computed directly from the demos.
    (train_bc.py needs nothing equivalent -- BC has no explicit normalization
    layer; it relies on the frozen encoder's built-in ImageNet normalization.)
    """
    states, actions = [], []
    images = {camera: [] for camera in image_keys}
    for episode in episodes:
        for transition in episode:
            obs = transition["observations"]
            states.append(np.asarray(obs["state"], dtype=np.float32).reshape(-1))
            actions.append(np.asarray(transition["actions"], dtype=np.float32))
            for camera in image_keys:
                images[camera].append(np.asarray(obs[camera], dtype=np.uint8).reshape(-1, 3))

    states = np.stack(states)
    actions = np.stack(actions)

    stats = {
        "observation.state": {
            "mean": torch.from_numpy(states.mean(0)),
            "std": torch.from_numpy(states.std(0) + 1e-6),
        },
        "action": {
            "mean": torch.from_numpy(actions.mean(0)),
            "std": torch.from_numpy(actions.std(0) + 1e-6),
        },
    }
    for camera in image_keys:
        pixels = np.concatenate(images[camera], axis=0).astype(np.float32) / 255.0
        stats[act_image_keys[camera]] = {
            "mean": torch.from_numpy(pixels.mean(0)),
            "std": torch.from_numpy(pixels.std(0) + 1e-6),
        }
    return stats


def _cycle(dataloader: DataLoader):
    while True:
        for batch in dataloader:
            yield batch


def _prune_old_checkpoints(checkpoint_path: str, keep: int) -> None:
    """Deletes all but the `keep` most-recent checkpoint_<step> dirs under
    checkpoint_path (sorted by the numeric step in the dir name, not mtime --
    robust to dirs written out of order). No-op if keep <= 0."""
    if keep <= 0:
        return
    ckpt_dirs = glob.glob(os.path.join(checkpoint_path, "checkpoint_*"))
    def _step(d):
        try:
            return int(os.path.basename(d).removeprefix("checkpoint_"))
        except ValueError:
            return -1  # non-numeric suffix -- sort first, never pruned-preferred
    ckpt_dirs = sorted((d for d in ckpt_dirs if os.path.isdir(d)), key=_step)
    # list[:-keep] is already [] whenever len(list) <= keep -- Python's negative
    # slicing clamps at the start, no extra length check needed.
    for d in ckpt_dirs[:-keep]:
        shutil.rmtree(d)


# ---------------------------------------------------------------------------
# Train / eval loops (mirrors train_bc.py's train()/eval()/main() structure)
# ---------------------------------------------------------------------------


def build_observation(obs, image_keys: List[str], act_image_keys: Dict[str, str]) -> dict:
    """Raw gym obs dict -> ACT's feature-keyed, batch-of-1 observation dict."""
    state = np.asarray(obs["state"], dtype=np.float32).reshape(-1)
    observation = {"observation.state": torch.from_numpy(state)}
    for camera in image_keys:
        img_hwc = np.asarray(obs[camera], dtype=np.uint8)
        img_hwc = img_hwc.reshape(*img_hwc.shape[-3:])
        image = torch.from_numpy(img_hwc).permute(2, 0, 1).float() / 255.0
        observation[act_image_keys[camera]] = image.unsqueeze(0)
    return observation


def eval(env, policy, preprocessor, postprocessor, image_keys, act_image_keys):
    """
    This is the actor loop, which runs when --eval_n_trajs > 0.
    """
    success_counter = 0
    time_list = []
    zero_action = np.zeros(env.action_space.sample().shape)
    for episode in range(FLAGS.eval_n_trajs):
        obs, _ = env.reset()

        # Wait for a SHIFT press before letting the policy act, same pattern as
        # train_bc.py / record_demos.py.
        print(f"[eval {episode}/{FLAGS.eval_n_trajs}] Hold SHIFT to let the policy start.")
        while True:
            obs, _, wait_done, _, info = env.step(zero_action)
            if info.get("start_recording"):
                break
            if wait_done:
                obs, _ = env.reset()

        policy.reset()  # clears ACT's internal action queue
        done = False
        start_time = time.time()
        while not done:
            observation = build_observation(obs, image_keys, act_image_keys)
            with torch.inference_mode():
                observation = preprocessor(observation)
                action = policy.select_action(observation)
                action = postprocessor(action)
            actions = action[0].detach().cpu().numpy()

            next_obs, reward, done, truncated, info = env.step(actions)
            obs = next_obs
            if done:
                if reward:
                    dt = time.time() - start_time
                    time_list.append(dt)
                    print(dt)
                success_counter += reward
                print(reward)
                print(f"{success_counter}/{episode + 1}")

    print(f"success rate: {success_counter / FLAGS.eval_n_trajs}")
    print(f"average time: {np.mean(time_list)}")


def train(policy, preprocessor, postprocessor, dataloader, config, optimizer, device, wandb_logger=None):
    data_iter = _cycle(dataloader)

    pbar = tqdm.tqdm(
        range(FLAGS.resume_step, FLAGS.train_steps),
        dynamic_ncols=True,
        desc="act_pretraining",
        initial=FLAGS.resume_step,
        total=FLAGS.train_steps,
    )
    for step in pbar:
        batch = next(data_iter)
        batch = {k: v.to(device) for k, v in batch.items()}
        batch = preprocessor(batch)

        policy.train()
        loss, loss_dict = policy.forward(batch)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), max_norm=10.0)
        optimizer.step()

        pbar.set_postfix(loss_dict)  # live loss on the progress bar, regardless of --debug/wandb

        if step % config.log_period == 0 and wandb_logger:
            wandb_logger.log({"act": loss_dict}, step=step)

        # Named by steps-COMPLETED (step + 1), not the raw loop index -- step
        # only ever reaches FLAGS.train_steps - 1, so naming by the raw index
        # would leave the final checkpoint as e.g. checkpoint_19990 instead of
        # checkpoint_20000, silently breaking the "already trained" guard
        # above (os.path.isdir check) that looks for checkpoint_{train_steps}.
        if (step + 1) % FLAGS.save_interval == 0 or step == FLAGS.train_steps - 1:
            ckpt_root = os.path.abspath(FLAGS.checkpoint_path)
            ckpt_dir = os.path.join(ckpt_root, f"checkpoint_{step + 1}")
            policy.save_pretrained(ckpt_dir)
            preprocessor.save_pretrained(ckpt_dir)
            postprocessor.save_pretrained(ckpt_dir)
            torch.save(optimizer.state_dict(), os.path.join(ckpt_dir, "optimizer.pt"))
            _prune_old_checkpoints(ckpt_root, FLAGS.keep_checkpoints)
    print_green("act pretraining done and saved checkpoint")


##############################################################################


def main(_):
    config: DefaultTrainingConfig = CONFIG_MAPPING[FLAGS.exp_name]()
    assert FLAGS.exp_name in CONFIG_MAPPING, "Experiment folder not found."
    eval_mode = FLAGS.eval_n_trajs > 0
    env = config.get_environment(fake_env=not eval_mode, save_video=FLAGS.save_video, classifier=True)
    env = RecordEpisodeStatistics(env)

    torch.manual_seed(FLAGS.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    image_keys = list(config.image_keys)
    act_image_keys = {camera: image_feature_key(camera) for camera in image_keys}

    state_dim = int(np.prod(env.observation_space["state"].shape[1:]))
    action_dim = int(env.action_space.shape[-1])

    input_features = {
        "observation.state": PolicyFeature(type=FeatureType.STATE, shape=(state_dim,)),
    }
    for camera in image_keys:
        h, w, c = env.observation_space[camera].shape[-3:]
        input_features[act_image_keys[camera]] = PolicyFeature(type=FeatureType.VISUAL, shape=(c, h, w))
    output_features = {"action": PolicyFeature(type=FeatureType.ACTION, shape=(action_dim,))}

    act_cfg = ACTConfig(
        chunk_size=FLAGS.chunk_size,
        n_action_steps=FLAGS.n_action_steps,
        optimizer_lr=FLAGS.optimizer_lr,
        optimizer_lr_backbone=FLAGS.optimizer_lr_backbone if FLAGS.optimizer_lr_backbone is not None else FLAGS.optimizer_lr,
        dim_model=FLAGS.dim_model,
        dim_feedforward=FLAGS.dim_feedforward,
        input_features=input_features,
        output_features=output_features,
        device=device,
    )
    act_cfg.validate_features()

    if not eval_mode:
        assert not os.path.isdir(
            os.path.join(FLAGS.checkpoint_path, f"checkpoint_{FLAGS.train_steps}")
        )

        demo_paths = glob.glob(FLAGS.demo_path) if FLAGS.demo_path else glob.glob(os.path.join(os.getcwd(), "demo_data", "*.pkl"))
        assert demo_paths, f"No demo .pkl files found (demo_path flag: {FLAGS.demo_path!r})"

        episodes = _load_episodes(demo_paths)
        print(f"loaded {len(episodes)} episodes, {sum(len(e) for e in episodes)} transitions")

        dataset = ACTChunkDataset(episodes, image_keys, act_image_keys, chunk_size=FLAGS.chunk_size)
        print(f"act training dataset size (non-idle anchors): {len(dataset)}")
        dataloader = DataLoader(
            dataset, batch_size=FLAGS.batch_size, shuffle=True, num_workers=0, drop_last=True
        )

        if FLAGS.resume_step > 0:
            resume_dir = os.path.join(os.path.abspath(FLAGS.checkpoint_path), f"checkpoint_{FLAGS.resume_step}")
            assert os.path.isdir(resume_dir), f"--resume_step={FLAGS.resume_step} but {resume_dir} doesn't exist"
            print_green(f"Resuming from {resume_dir}")
            policy = ACTPolicy.from_pretrained(resume_dir).to(device)
            # Reuse the ORIGINAL normalization stats from the checkpoint, not
            # ones recomputed from this run's --demo_path -- the model's inputs
            # need a fixed normalization across its whole training lifetime;
            # recomputing from a different/aggregated dataset here would shift
            # normalization out from under everything it already learned.
            preprocessor, postprocessor = make_pre_post_processors(policy_cfg=policy.config, pretrained_path=resume_dir)
            optimizer = torch.optim.AdamW(
                policy.get_optim_params(), lr=act_cfg.optimizer_lr, weight_decay=act_cfg.optimizer_weight_decay
            )
            optimizer_ckpt = os.path.join(resume_dir, "optimizer.pt")
            if os.path.isfile(optimizer_ckpt):
                optimizer.load_state_dict(torch.load(optimizer_ckpt, map_location=device))
                print_green("Restored optimizer (AdamW) state.")
            else:
                print_yellow(
                    f"No optimizer.pt found at {resume_dir} (checkpoint predates optimizer-state "
                    "saving) -- continuing with a freshly-initialized optimizer. Policy weights "
                    "still resume correctly; only Adam's running gradient-moment estimates restart."
                )
        else:
            dataset_stats = compute_dataset_stats(episodes, image_keys, act_image_keys)
            policy = ACTPolicy(config=act_cfg).to(device)
            preprocessor, postprocessor = make_pre_post_processors(policy_cfg=act_cfg, dataset_stats=dataset_stats)
            optimizer = torch.optim.AdamW(
                policy.get_optim_params(), lr=act_cfg.optimizer_lr, weight_decay=act_cfg.optimizer_weight_decay
            )

        wandb_logger = make_wandb_logger(project="hil-serl", description=FLAGS.exp_name, debug=FLAGS.debug)

        print_green("starting learner loop")
        train(policy, preprocessor, postprocessor, dataloader, config, optimizer, device, wandb_logger=wandb_logger)

    else:
        ckpt_dir = os.path.abspath(FLAGS.checkpoint_path)
        policy = ACTPolicy.from_pretrained(ckpt_dir).to(device)
        policy.eval()
        preprocessor, postprocessor = make_pre_post_processors(policy_cfg=policy.config, pretrained_path=ckpt_dir)

        print_green("starting actor loop")
        eval(env, policy, preprocessor, postprocessor, image_keys, act_image_keys)


if __name__ == "__main__":
    app.run(main)
