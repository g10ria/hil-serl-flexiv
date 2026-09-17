#!/usr/bin/env python3
"""Standalone rollout/eval for a single checkpoint -- no learner connection, no
TrainerClient, no implicit "restore latest checkpoint" step.

train_rlpd.py's --actor --eval_checkpoint_step path still goes through main(),
which *unconditionally* restores the latest checkpoint (blocking on
`input("Press Enter to resume training.")`) before the eval branch inside
actor() ever runs -- and if --eval_checkpoint_step isn't actually parsed as
nonzero, actor() falls through to the full training loop, which waits for a
learner connection. This script skips all of that: give it an exact
checkpoint step and it just rolls the policy out --eval_n_trajs times.

Usage:
    python eval_policy.py --exp_name=flexiv_task \
        --checkpoint_path=experiments/flexiv_task/first_run \
        --checkpoint_step=50000 --eval_n_trajs=20 --deterministic_eval

Appends one line per run to <checkpoint_path>/eval_results.jsonl, same schema
train_rlpd.py's eval path used -- so plot_eval_success.py works unchanged.
"""

import json
import os
import time

import jax
import jax.numpy as jnp
import numpy as np
from absl import app, flags
from flax.training import checkpoints

from serl_launcher.utils.launcher import (
    make_sac_pixel_agent,
    make_sac_pixel_agent_hybrid_single_arm,
    make_sac_pixel_agent_hybrid_dual_arm,
)

from experiments.mappings import CONFIG_MAPPING

FLAGS = flags.FLAGS

flags.DEFINE_string("exp_name", None, "Name of experiment corresponding to folder.", required=True)
flags.DEFINE_string("checkpoint_path", None, "Path to the run dir containing checkpoint_<step>/.", required=True)
flags.DEFINE_integer("checkpoint_step", None, "Exact checkpoint step to load. Required -- no 'latest' fallback.", required=True)
flags.DEFINE_integer("eval_n_trajs", 5, "Number of trajectories to roll out.")
flags.DEFINE_integer("seed", 42, "Random seed.")
flags.DEFINE_boolean(
    "deterministic_eval", False,
    "Take the policy's mean action (argmax) instead of sampling stochastically, so you "
    "see the checkpoint's learned behavior without exploration noise on top."
)
flags.DEFINE_boolean(
    "human_classifier", False,
    "Use a live human-typed 'Success? (1/0)' prompt for reward instead of the trained "
    "classifier checkpoint."
)

devices = jax.local_devices()
mesh = jax.sharding.Mesh(devices, axis_names=("x",))
sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())


def print_green(x):
    print("\033[92m {}\033[00m".format(x))


def main(_):
    config = CONFIG_MAPPING[FLAGS.exp_name]()

    ckpt_dir = os.path.abspath(FLAGS.checkpoint_path)
    ckpt_step_dir = os.path.join(ckpt_dir, f"checkpoint_{FLAGS.checkpoint_step}")
    if not os.path.isdir(ckpt_step_dir):
        available = sorted(d for d in os.listdir(ckpt_dir) if d.startswith("checkpoint_"))
        raise FileNotFoundError(f"No checkpoint at {ckpt_step_dir}. Available: {available}")

    rng = jax.random.PRNGKey(FLAGS.seed)
    rng, sampling_rng = jax.random.split(rng)
    sampling_rng = jax.device_put(sampling_rng, sharding)

    env = config.get_environment(
        fake_env=False,
        save_video=False,
        classifier=True,
        human_classifier=FLAGS.human_classifier,
    )

    if config.setup_mode in ("single-arm-fixed-gripper", "dual-arm-fixed-gripper"):
        agent = make_sac_pixel_agent(
            seed=FLAGS.seed,
            sample_obs=env.observation_space.sample(),
            sample_action=env.action_space.sample(),
            image_keys=config.image_keys,
            encoder_type=config.encoder_type,
            discount=config.discount,
        )
    elif config.setup_mode == "single-arm-learned-gripper":
        agent = make_sac_pixel_agent_hybrid_single_arm(
            seed=FLAGS.seed,
            sample_obs=env.observation_space.sample(),
            sample_action=env.action_space.sample(),
            image_keys=config.image_keys,
            encoder_type=config.encoder_type,
            discount=config.discount,
        )
    elif config.setup_mode == "dual-arm-learned-gripper":
        agent = make_sac_pixel_agent_hybrid_dual_arm(
            seed=FLAGS.seed,
            sample_obs=env.observation_space.sample(),
            sample_action=env.action_space.sample(),
            image_keys=config.image_keys,
            encoder_type=config.encoder_type,
            discount=config.discount,
        )
    else:
        raise NotImplementedError(f"Unknown setup mode: {config.setup_mode}")

    agent = jax.device_put(jax.tree.map(jnp.array, agent), sharding)

    ckpt = checkpoints.restore_checkpoint(ckpt_dir, agent.state, step=FLAGS.checkpoint_step)
    agent = agent.replace(state=ckpt)
    print_green(f"loaded checkpoint {FLAGS.checkpoint_step} from {ckpt_dir}")

    zero_action = np.zeros(env.action_space.sample().shape)
    obs = None

    def wait_for_shift():
        """Free-movement phase: teleoperate with the SpaceMouse (e.g. to reset the
        scene) with nothing recorded and no policy acting yet. Hold SHIFT to let the
        policy start -- mirrors train_rlpd.py's actor loop."""
        nonlocal obs
        print("Reposition with the SpaceMouse if needed. Hold SHIFT to let the policy start.")
        while True:
            obs, _, wait_done, wait_truncated, info = env.step(zero_action)
            if info.get("start_recording"):
                return
            if wait_done or wait_truncated:
                obs, _ = env.reset()

    try:
        success_counter = 0
        time_list = []

        for episode in range(FLAGS.eval_n_trajs):
            obs, _ = env.reset()
            wait_for_shift()
            done = False
            start_time = time.time()
            while not done:
                sampling_rng, key = jax.random.split(sampling_rng)
                actions = agent.sample_actions(
                    observations=jax.device_put(obs),
                    argmax=FLAGS.deterministic_eval,
                    seed=key,
                )
                actions = np.asarray(jax.device_get(actions))

                obs, reward, done, truncated, info = env.step(actions)

                if done:
                    if reward:
                        dt = time.time() - start_time
                        time_list.append(dt)
                        print(dt)
                    success_counter += reward
                    print(reward)
                    print(f"{success_counter}/{episode + 1}")

        success_rate = success_counter / FLAGS.eval_n_trajs
        avg_time = float(np.mean(time_list)) if time_list else None
        print(f"success rate: {success_rate}")
        print(f"average time: {avg_time}")

        results_path = os.path.join(ckpt_dir, "eval_results.jsonl")
        with open(results_path, "a") as f:
            f.write(json.dumps({
                "step": FLAGS.checkpoint_step,
                "n_trajs": FLAGS.eval_n_trajs,
                "successes": int(success_counter),
                "success_rate": success_rate,
                "avg_success_time": avg_time,
                "deterministic": FLAGS.deterministic_eval,
                "timestamp": time.time(),
            }) + "\n")
        print_green(f"appended result to {results_path}")
    finally:
        env.close()


if __name__ == "__main__":
    app.run(main)
