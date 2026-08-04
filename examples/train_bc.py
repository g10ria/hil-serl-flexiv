#!/usr/bin/env python3

import glob
import time
import jax
import jax.numpy as jnp
import numpy as np
import tqdm
from absl import app, flags
from flax.training import checkpoints
import os
import pickle as pkl
from gymnasium.wrappers import RecordEpisodeStatistics

from serl_launcher.agents.continuous.bc import BCAgent

from serl_launcher.utils.launcher import (
    make_bc_agent,
    make_trainer_config,
    make_wandb_logger,
)
from serl_launcher.data.data_store import MemoryEfficientReplayBufferDataStore

from experiments.mappings import CONFIG_MAPPING
from experiments.config import DefaultTrainingConfig
FLAGS = flags.FLAGS

flags.DEFINE_string("exp_name", None, "Name of experiment corresponding to folder.")
flags.DEFINE_integer("seed", 42, "Random seed.")
flags.DEFINE_string("ip", "localhost", "IP address of the learner.")
flags.DEFINE_string("bc_checkpoint_path", None, "Path to save checkpoints.")
flags.DEFINE_integer("eval_n_trajs", 0, "Number of trajectories to evaluate.")
flags.DEFINE_integer("train_steps", 20_000, "Number of pretraining steps.")
flags.DEFINE_string(
    "demo_path", None,
    "Glob pattern (or single file path) for demo .pkl(s) to train on. "
    "Defaults to demo_data/*.pkl under the current working directory."
)
flags.DEFINE_bool("save_video", False, "Save video of the evaluation.")
flags.DEFINE_bool(
    "argmax", False,
    "Eval rollout only: use the policy's deterministic mode instead of stochastically "
    "sampling from its action distribution. Off by default (matches prior behavior)."
)


flags.DEFINE_boolean(
    "debug", False, "Debug mode."
)  # debug mode will disable wandb logging


devices = jax.local_devices()
num_devices = len(devices)
# PositionalSharding was removed in newer jax; a NamedSharding over a trivial
# mesh with an empty PartitionSpec is the modern equivalent of "replicated
# across all devices" -- every call site below used PositionalSharding only
# via .replicate(), so `sharding` itself now stands in for that directly.
mesh = jax.sharding.Mesh(devices, axis_names=("x",))
sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())


def print_green(x):
    return print("\033[92m {}\033[00m".format(x))


def print_yellow(x):
    return print("\033[93m {}\033[00m".format(x))


##############################################################################

def eval(
    env,
    bc_agent: BCAgent,
    sampling_rng,
):
    """
    This is the actor loop, which runs when "--actor" is set to True.
    """
    success_counter = 0
    time_list = []
    zero_action = np.zeros(env.action_space.sample().shape)
    for episode in range(FLAGS.eval_n_trajs):
        obs, _ = env.reset()

        # Wait for a SHIFT press before letting the policy act -- gives a
        # moment to get near the SpaceMouse (still live as an override) or
        # just confirm the reset pose looks right before real motion starts.
        # Mirrors record_demos.py's free-movement-then-clutch pattern; since
        # zero_action still passes through FlexivSpacemouseIntervention,
        # touching the SpaceMouse during this wait repositions the arm same
        # as during collection.
        print(f"[eval {episode}/{FLAGS.eval_n_trajs}] Hold SHIFT to let the policy start.")
        while True:
            obs, _, wait_done, _, info = env.step(zero_action)
            if info.get("start_recording"):
                break
            if wait_done:
                obs, _ = env.reset()

        done = False
        start_time = time.time()
        while not done:
            rng, key = jax.random.split(sampling_rng)

            actions = bc_agent.sample_actions(observations=obs, seed=key, argmax=FLAGS.argmax)
            actions = np.asarray(jax.device_get(actions))
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


##############################################################################


def train(
    bc_agent: BCAgent,
    bc_replay_buffer,
    config: DefaultTrainingConfig,
    wandb_logger=None,
):

    bc_replay_iterator = bc_replay_buffer.get_iterator(
        sample_args={
            "batch_size": config.batch_size,
            "pack_obs_and_next_obs": False,
        },
        device=sharding,
    )
    
    # Pretrain BC policy to get started
    for step in tqdm.tqdm(
        range(FLAGS.train_steps),
        dynamic_ncols=True,
        desc="bc_pretraining",
    ):
        batch = next(bc_replay_iterator)
        bc_agent, bc_update_info = bc_agent.update(batch)
        if step % config.log_period == 0 and wandb_logger:
            wandb_logger.log({"bc": bc_update_info}, step=step)
        if step > FLAGS.train_steps - 100 and step % 10 == 0:
            checkpoints.save_checkpoint(
                os.path.abspath(FLAGS.bc_checkpoint_path), bc_agent.state, step=step, keep=5
            )
    print_green("bc pretraining done and saved checkpoint")


##############################################################################


def main(_):
    config: DefaultTrainingConfig = CONFIG_MAPPING[FLAGS.exp_name]()

    assert config.batch_size % num_devices == 0
    assert FLAGS.exp_name in CONFIG_MAPPING, "Experiment folder not found."
    eval_mode = FLAGS.eval_n_trajs > 0
    env = config.get_environment(
        fake_env=not eval_mode,
        save_video=FLAGS.save_video,
        classifier=True,
    )
    env = RecordEpisodeStatistics(env)

    bc_agent: BCAgent = make_bc_agent(
        seed=FLAGS.seed,
        sample_obs=env.observation_space.sample(),
        sample_action=env.action_space.sample(),
        image_keys=config.image_keys,
        encoder_type=config.encoder_type,
    )

    # replicate agent across devices
    # need the jnp.array to avoid a bug where device_put doesn't recognize primitives
    bc_agent: BCAgent = jax.device_put(
        jax.tree.map(jnp.array, bc_agent), sharding
    )

    if not eval_mode:
        assert not os.path.isdir(
            os.path.join(FLAGS.bc_checkpoint_path, f"checkpoint_{FLAGS.train_steps}")
        )

        bc_replay_buffer = MemoryEfficientReplayBufferDataStore(
            env.observation_space,
            env.action_space,
            capacity=config.replay_buffer_capacity,
            image_keys=config.image_keys,
        )

        # set up wandb and logging
        wandb_logger = make_wandb_logger(
            project="hil-serl",
            description=FLAGS.exp_name,
            debug=FLAGS.debug,
        )

        demo_path = glob.glob(FLAGS.demo_path) if FLAGS.demo_path else glob.glob(os.path.join(os.getcwd(), "demo_data", "*.pkl"))

        assert demo_path, f"No demo .pkl files found (demo_path flag: {FLAGS.demo_path!r})"

        for path in demo_path:
            with open(path, "rb") as f:
                transitions = pkl.load(f)
                for transition in transitions:
                    if np.linalg.norm(transition['actions']) > 0.0:
                        bc_replay_buffer.insert(transition)
        print(f"bc replay buffer size: {len(bc_replay_buffer)}")

        # learner loop
        print_green("starting learner loop")
        train(
            bc_agent=bc_agent,
            bc_replay_buffer=bc_replay_buffer,
            wandb_logger=wandb_logger,
            config=config,
        )

    else:
        rng = jax.random.PRNGKey(FLAGS.seed)
        sampling_rng = jax.device_put(rng, sharding)

        ckpt_dir = os.path.abspath(FLAGS.bc_checkpoint_path)
        # restore_checkpoint silently no-ops (returns the untouched target) if
        # nothing's found at this path -- that would mean rolling out a
        # randomly initialized head on real hardware with no error at all.
        if checkpoints.latest_checkpoint(ckpt_dir) is None:
            raise FileNotFoundError(f"No checkpoint found at {ckpt_dir}")
        bc_ckpt = checkpoints.restore_checkpoint(ckpt_dir, bc_agent.state)
        bc_agent = bc_agent.replace(state=bc_ckpt)

        print_green("starting actor loop")
        eval(
            env=env,
            bc_agent=bc_agent,
            sampling_rng=sampling_rng,
        )


if __name__ == "__main__":
    app.run(main)
