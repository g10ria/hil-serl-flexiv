#!/usr/bin/env python3

import glob
import json
import time
import jax
import jax.numpy as jnp
import numpy as np
import tqdm
from absl import app, flags
from flax.training import checkpoints
import os
import copy
import pickle as pkl
from gymnasium.wrappers import RecordEpisodeStatistics
from natsort import natsorted

from serl_launcher.agents.continuous.sac import SACAgent
from serl_launcher.agents.continuous.sac_hybrid_single import SACAgentHybridSingleArm
from serl_launcher.agents.continuous.sac_hybrid_dual import SACAgentHybridDualArm
from serl_launcher.utils.timer_utils import Timer
from serl_launcher.utils.train_utils import concat_batches

from agentlace.trainer import TrainerServer, TrainerClient
from agentlace.data.data_store import QueuedDataStore

from serl_launcher.utils.launcher import (
    make_sac_pixel_agent,
    make_sac_pixel_agent_hybrid_single_arm,
    make_sac_pixel_agent_hybrid_dual_arm,
    make_trainer_config,
    make_wandb_logger,
)
from serl_launcher.data.data_store import MemoryEfficientReplayBufferDataStore

from experiments.mappings import CONFIG_MAPPING

FLAGS = flags.FLAGS

flags.DEFINE_string("exp_name", None, "Name of experiment corresponding to folder.")
flags.DEFINE_integer("seed", 42, "Random seed.")
flags.DEFINE_boolean("learner", False, "Whether this is a learner.")
flags.DEFINE_boolean("actor", False, "Whether this is an actor.")
flags.DEFINE_string("ip", "localhost", "IP address of the learner.")
flags.DEFINE_multi_string("demo_path", None, "Path to the demo data.")
flags.DEFINE_string("checkpoint_path", None, "Path to save checkpoints.")
flags.DEFINE_integer("eval_checkpoint_step", 0, "Step to evaluate the checkpoint.")
flags.DEFINE_integer("eval_n_trajs", 0, "Number of trajectories to evaluate.")
flags.DEFINE_boolean(
    "deterministic_eval", False,
    "For --eval_checkpoint_step runs: take the policy's mean action (argmax) instead of "
    "sampling stochastically, so you see the checkpoint's actual learned behavior without "
    "exploration noise on top. Defaults off (stochastic, matching normal rollout) since "
    "that's also useful to see. Has no effect outside --eval_checkpoint_step."
)
flags.DEFINE_boolean("save_video", False, "Save video.")
flags.DEFINE_boolean(
    "human_classifier", False,
    "Use a live human-typed 'Success? (1/0)' prompt (HumanClassifierWrapper) for reward "
    "instead of the trained classifier checkpoint. Leave off for real online RLPD training "
    "-- a human can't label every actor step fast enough; this is mainly for --actor "
    "--eval_checkpoint_step eval runs where you want a manual read instead."
)

flags.DEFINE_boolean(
    "demo_buffer_offline_only", False,
    "Keep the demo buffer restricted to offline demos only, excluding SpaceMouse "
    "interventions. On the actor: interventions still override the executed action and "
    "still get logged to the online replay buffer as usual, but are NOT also inserted "
    "into the demo buffer. On the learner: skips loading checkpoint_path/demo_buffer/*.pkl "
    "(past interventions dumped to disk) -- demo buffer is built from --demo_path only. "
    "Pass it on both actor and learner to keep them consistent."
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


##############################################################################


def actor(agent, data_store, intvn_data_store, env, sampling_rng):
    """
    This is the actor loop, which runs when "--actor" is set to True.
    """
    zero_action = np.zeros(env.action_space.sample().shape)

    def wait_for_shift():
        """Free-movement phase: teleoperate with the SpaceMouse (e.g. to reposition
        the peg in the gripper) with nothing recorded and no policy acting yet.
        Hold SHIFT to let the policy start. Mirrors record_demos.py's own
        reposition-then-clutch flow. If `done`/`truncated` fires while still just
        repositioning (MAX_EPISODE_LENGTH doesn't apply pre-SHIFT, but the reward
        classifier still runs every step and could false-positive, or ESC/quit
        could fire) it's discarded here and we just keep waiting."""
        nonlocal obs
        print("Reposition with the SpaceMouse if needed. Hold SHIFT to let the policy start.")
        while True:
            obs, _, wait_done, wait_truncated, info = env.step(zero_action)
            if info.get("start_recording"):
                return
            if wait_done or wait_truncated:
                obs, _ = env.reset()

    if FLAGS.eval_checkpoint_step:
        success_counter = 0
        time_list = []

        ckpt = checkpoints.restore_checkpoint(
            os.path.abspath(FLAGS.checkpoint_path),
            agent.state,
            step=FLAGS.eval_checkpoint_step,
        )
        agent = agent.replace(state=ckpt)

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
                    seed=key
                )
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

        success_rate = success_counter / FLAGS.eval_n_trajs
        avg_time = float(np.mean(time_list)) if time_list else None
        print(f"success rate: {success_rate}")
        print(f"average time: {avg_time}")

        # Append this checkpoint's result so multiple --eval_checkpoint_step
        # runs (across different checkpoints) can be aggregated afterward --
        # e.g. plot_eval_success.py's sliding-window average over checkpoints.
        # One line per run (not overwritten), so re-running the same
        # checkpoint just adds another data point rather than losing history.
        results_path = os.path.join(os.path.abspath(FLAGS.checkpoint_path), "eval_results.jsonl")
        with open(results_path, "a") as f:
            f.write(json.dumps({
                "step": FLAGS.eval_checkpoint_step,
                "n_trajs": FLAGS.eval_n_trajs,
                "successes": int(success_counter),
                "success_rate": success_rate,
                "avg_success_time": avg_time,
                "deterministic": FLAGS.deterministic_eval,
                "timestamp": time.time(),
            }) + "\n")
        print_green(f"appended result to {results_path}")
        return  # after done eval, return and exit
    
    start_step = (
        int(os.path.basename(natsorted(glob.glob(os.path.join(FLAGS.checkpoint_path, "buffer/*.pkl")))[-1])[12:-4]) + 1
        if FLAGS.checkpoint_path and os.path.exists(FLAGS.checkpoint_path)
        else 0
    )

    datastore_dict = {
        "actor_env": data_store,
        "actor_env_intvn": intvn_data_store,
    }

    client = TrainerClient(
        "actor_env",
        FLAGS.ip,
        make_trainer_config(),
        data_stores=datastore_dict,
        wait_for_server=True,
        timeout_ms=3000,
    )

    # Function to update the agent with new params
    weights_received_count = 0
    last_weights_received_time = time.time()

    def update_params(params):
        nonlocal agent, weights_received_count, last_weights_received_time
        agent = agent.replace(state=agent.state.replace(params=params))
        weights_received_count += 1
        now = time.time()
        elapsed = now - last_weights_received_time
        last_weights_received_time = now
        # This callback fires on the broadcast client's own background thread
        # (recv_network_callback -> async_start), not the main loop's thread --
        # tqdm.write (not plain print) is what avoids garbling the live pbar
        # line when a message lands mid-redraw from another thread.
        tqdm.tqdm.write(
            f"\033[92m[actor] received new weights from learner "
            f"(update #{weights_received_count}, {elapsed:.1f}s since last)\033[00m"
        )

    client.recv_network_callback(update_params)

    transitions = []
    demo_transitions = []

    obs, _ = env.reset()
    wait_for_shift()
    done = False

    # training loop
    timer = Timer()
    running_return = 0.0
    already_intervened = False
    intervention_count = 0
    intervention_steps = 0

    pbar = tqdm.tqdm(range(start_step, config.max_steps), dynamic_ncols=True)
    try:
        for step in pbar:
            timer.tick("total")

            with timer.context("sample_actions"):
                if step < config.random_steps:
                    actions = env.action_space.sample()
                else:
                    sampling_rng, key = jax.random.split(sampling_rng)
                    actions = agent.sample_actions(
                        observations=jax.device_put(obs),
                        seed=key,
                        argmax=False,
                    )
                    actions = np.asarray(jax.device_get(actions))

            # Step environment
            with timer.context("step_env"):

                next_obs, reward, done, truncated, info = env.step(actions)
                if "left" in info:
                    info.pop("left")
                if "right" in info:
                    info.pop("right")

                # override the action with the intervention action
                if "intervene_action" in info:
                    actions = info.pop("intervene_action")
                    intervention_steps += 1
                    if not already_intervened:
                        intervention_count += 1
                    already_intervened = True
                else:
                    already_intervened = False

                running_return += reward
                transition = dict(
                    observations=obs,
                    actions=actions,
                    next_observations=next_obs,
                    rewards=reward,
                    masks=1.0 - done,
                    dones=done,
                )
                if 'grasp_penalty' in info:
                    transition['grasp_penalty']= info['grasp_penalty']
                data_store.insert(transition)
                transitions.append(copy.deepcopy(transition))
                if already_intervened and not FLAGS.demo_buffer_offline_only:
                    intvn_data_store.insert(transition)
                    demo_transitions.append(copy.deepcopy(transition))

                obs = next_obs
                if done or truncated:
                    info["episode"]["intervention_count"] = intervention_count
                    info["episode"]["intervention_steps"] = intervention_steps
                    stats = {"environment": info}  # send stats to the learner to log
                    client.request("send-stats", stats)
                    pbar.set_description(f"last return: {running_return}")
                    running_return = 0.0
                    intervention_count = 0
                    intervention_steps = 0
                    already_intervened = False
                    client.update()
                    obs, _ = env.reset()
                    wait_for_shift()

            if step > 0 and config.buffer_period > 0 and step % config.buffer_period == 0:
                # dump to pickle file
                buffer_path = os.path.join(FLAGS.checkpoint_path, "buffer")
                demo_buffer_path = os.path.join(FLAGS.checkpoint_path, "demo_buffer")
                if not os.path.exists(buffer_path):
                    os.makedirs(buffer_path)
                if not os.path.exists(demo_buffer_path):
                    os.makedirs(demo_buffer_path)
                with open(os.path.join(buffer_path, f"transitions_{step}.pkl"), "wb") as f:
                    pkl.dump(transitions, f)
                    transitions = []
                with open(
                    os.path.join(demo_buffer_path, f"transitions_{step}.pkl"), "wb"
                ) as f:
                    pkl.dump(demo_transitions, f)
                    demo_transitions = []

            timer.tock("total")

            if step % config.log_period == 0:
                stats = {"timer": timer.get_average_times()}
                client.request("send-stats", stats)
    finally:
        # Without this, the TrainerClient's background broadcast-listener thread
        # (started by recv_network_callback) is left dangling on Ctrl+C / any
        # exit -- same category of issue eval_act.py's env.close() fixes for the
        # SpaceMouse's background process (see main()'s finally block below).
        client.stop()


##############################################################################


def learner(rng, agent, replay_buffer, demo_buffer, wandb_logger=None):
    """
    The learner loop, which runs when "--learner" is set to True.
    """
    start_step = (
        int(os.path.basename(checkpoints.latest_checkpoint(os.path.abspath(FLAGS.checkpoint_path)))[11:])
        + 1
        if FLAGS.checkpoint_path and os.path.exists(FLAGS.checkpoint_path)
        else 0
    )
    step = start_step

    def stats_callback(type: str, payload: dict) -> dict:
        """Callback for when server receives stats request."""
        assert type == "send-stats", f"Invalid request type: {type}"
        if wandb_logger is not None:
            wandb_logger.log(payload, step=step)
        return {}  # not expecting a response

    # Create server
    server = TrainerServer(make_trainer_config(), request_callback=stats_callback)
    server.register_data_store("actor_env", replay_buffer)
    server.register_data_store("actor_env_intvn", demo_buffer)
    server.start(threaded=True)

    # Loop to wait until replay_buffer is filled
    pbar = tqdm.tqdm(
        total=config.training_starts,
        initial=len(replay_buffer),
        desc="Filling up replay buffer",
        position=0,
        leave=True,
    )
    while len(replay_buffer) < config.training_starts:
        pbar.update(len(replay_buffer) - pbar.n)  # Update progress bar
        time.sleep(1)
    pbar.update(len(replay_buffer) - pbar.n)  # Update progress bar
    pbar.close()

    # send the initial network to the actor
    server.publish_network(agent.state.params)
    print_green("sent initial network to actor")

    # 50/50 sampling from RLPD, half from demo and half from online experience
    replay_iterator = replay_buffer.get_iterator(
        sample_args={
            "batch_size": config.batch_size // 2,
            "pack_obs_and_next_obs": True,
        },
        device=sharding,
    )
    demo_iterator = demo_buffer.get_iterator(
        sample_args={
            "batch_size": config.batch_size // 2,
            "pack_obs_and_next_obs": True,
        },
        device=sharding,
    )

    # wait till the replay buffer is filled with enough data
    timer = Timer()
    
    if isinstance(agent, SACAgent):
        train_critic_networks_to_update = frozenset({"critic"})
        train_networks_to_update = frozenset({"critic", "actor", "temperature"})
    else:
        train_critic_networks_to_update = frozenset({"critic", "grasp_critic"})
        train_networks_to_update = frozenset({"critic", "grasp_critic", "actor", "temperature"})

    pbar = tqdm.tqdm(
        range(start_step, config.max_steps), dynamic_ncols=True, desc="learner"
    )
    for step in pbar:
        # run n-1 critic updates and 1 critic + actor update.
        # This makes training on GPU faster by reducing the large batch transfer time from CPU to GPU
        for critic_step in range(config.cta_ratio - 1):
            with timer.context("sample_replay_buffer"):
                batch = next(replay_iterator)
                demo_batch = next(demo_iterator)
                batch = concat_batches(batch, demo_batch, axis=0)

            with timer.context("train_critics"):
                agent, critics_info = agent.update(
                    batch,
                    networks_to_update=train_critic_networks_to_update,
                )

        with timer.context("train"):
            batch = next(replay_iterator)
            demo_batch = next(demo_iterator)
            batch = concat_batches(batch, demo_batch, axis=0)
            agent, update_info = agent.update(
                batch,
                networks_to_update=train_networks_to_update,
            )
        # publish the updated network
        if step > 0 and step % (config.steps_per_update) == 0:
            agent = jax.block_until_ready(agent)
            server.publish_network(agent.state.params)

        if step % config.log_period == 0:
            pbar.set_postfix(
                critic_loss=f"{update_info['critic']['critic_loss']:.6f}",
                actor_loss=f"{update_info['actor']['actor_loss']:.3f}",
            )
            if wandb_logger:
                wandb_logger.log(update_info, step=step)
                wandb_logger.log({"timer": timer.get_average_times()}, step=step)

        if (
            step > 0
            and config.checkpoint_period
            and step % config.checkpoint_period == 0
        ):
            checkpoints.save_checkpoint(
                os.path.abspath(FLAGS.checkpoint_path), agent.state, step=step, keep=100
            )


##############################################################################


def main(_):
    global config
    config = CONFIG_MAPPING[FLAGS.exp_name]()

    assert config.batch_size % num_devices == 0
    # seed
    rng = jax.random.PRNGKey(FLAGS.seed)
    rng, sampling_rng = jax.random.split(rng)

    assert FLAGS.exp_name in CONFIG_MAPPING, "Experiment folder not found."
    env = config.get_environment(
        fake_env=FLAGS.learner,
        save_video=FLAGS.save_video,
        classifier=True,
        human_classifier=FLAGS.human_classifier,
    )
    env = RecordEpisodeStatistics(env)

    rng, sampling_rng = jax.random.split(rng)
    
    if config.setup_mode == 'single-arm-fixed-gripper' or config.setup_mode == 'dual-arm-fixed-gripper':   
        agent: SACAgent = make_sac_pixel_agent(
            seed=FLAGS.seed,
            sample_obs=env.observation_space.sample(),
            sample_action=env.action_space.sample(),
            image_keys=config.image_keys,
            encoder_type=config.encoder_type,
            discount=config.discount,
        )
        include_grasp_penalty = False
    elif config.setup_mode == 'single-arm-learned-gripper':
        agent: SACAgentHybridSingleArm = make_sac_pixel_agent_hybrid_single_arm(
            seed=FLAGS.seed,
            sample_obs=env.observation_space.sample(),
            sample_action=env.action_space.sample(),
            image_keys=config.image_keys,
            encoder_type=config.encoder_type,
            discount=config.discount,
        )
        include_grasp_penalty = True
    elif config.setup_mode == 'dual-arm-learned-gripper':
        agent: SACAgentHybridDualArm = make_sac_pixel_agent_hybrid_dual_arm(
            seed=FLAGS.seed,
            sample_obs=env.observation_space.sample(),
            sample_action=env.action_space.sample(),
            image_keys=config.image_keys,
            encoder_type=config.encoder_type,
            discount=config.discount,
        )
        include_grasp_penalty = True
    else:
        raise NotImplementedError(f"Unknown setup mode: {config.setup_mode}")

    # replicate agent across devices
    # need the jnp.array to avoid a bug where device_put doesn't recognize primitives
    agent = jax.device_put(
        jax.tree.map(jnp.array, agent), sharding
    )

    if FLAGS.checkpoint_path is not None and os.path.exists(FLAGS.checkpoint_path):
        input("Checkpoint path already exists. Press Enter to resume training.")
        ckpt = checkpoints.restore_checkpoint(
            os.path.abspath(FLAGS.checkpoint_path),
            agent.state,
        )
        agent = agent.replace(state=ckpt)
        ckpt_number = os.path.basename(
            checkpoints.latest_checkpoint(os.path.abspath(FLAGS.checkpoint_path))
        )[11:]
        print_green(f"Loaded previous checkpoint at step {ckpt_number}.")

    def create_replay_buffer_and_wandb_logger():
        replay_buffer = MemoryEfficientReplayBufferDataStore(
            env.observation_space,
            env.action_space,
            capacity=config.replay_buffer_capacity,
            image_keys=config.image_keys,
            include_grasp_penalty=include_grasp_penalty,
        )
        # set up wandb and logging
        wandb_logger = make_wandb_logger(
            project="hil-serl",
            description=FLAGS.exp_name,
            debug=FLAGS.debug,
        )
        return replay_buffer, wandb_logger

    try:
        if FLAGS.learner:
            sampling_rng = jax.device_put(sampling_rng, device=sharding)
            replay_buffer, wandb_logger = create_replay_buffer_and_wandb_logger()
            demo_buffer = MemoryEfficientReplayBufferDataStore(
                env.observation_space,
                env.action_space,
                capacity=config.replay_buffer_capacity,
                image_keys=config.image_keys,
                include_grasp_penalty=include_grasp_penalty,
            )

            assert FLAGS.demo_path is not None
            for path in FLAGS.demo_path:
                with open(path, "rb") as f:
                    transitions = pkl.load(f)
                    for transition in transitions:
                        if 'infos' in transition and 'grasp_penalty' in transition['infos']:
                            transition['grasp_penalty'] = transition['infos']['grasp_penalty']
                        demo_buffer.insert(transition)
            print_green(f"demo buffer size: {len(demo_buffer)}")
            print_green(f"online buffer size: {len(replay_buffer)}")

            if FLAGS.checkpoint_path is not None and os.path.exists(
                os.path.join(FLAGS.checkpoint_path, "buffer")
            ):
                for file in glob.glob(os.path.join(FLAGS.checkpoint_path, "buffer/*.pkl")):
                    with open(file, "rb") as f:
                        transitions = pkl.load(f)
                        for transition in transitions:
                            replay_buffer.insert(transition)
                print_green(
                    f"Loaded previous buffer data. Replay buffer size: {len(replay_buffer)}"
                )

            if (
                not FLAGS.demo_buffer_offline_only
                and FLAGS.checkpoint_path is not None
                and os.path.exists(os.path.join(FLAGS.checkpoint_path, "demo_buffer"))
            ):
                for file in glob.glob(
                    os.path.join(FLAGS.checkpoint_path, "demo_buffer/*.pkl")
                ):
                    with open(file, "rb") as f:
                        transitions = pkl.load(f)
                        for transition in transitions:
                            demo_buffer.insert(transition)
                print_green(
                    f"Loaded previous demo buffer data. Demo buffer size: {len(demo_buffer)}"
                )

            # learner loop
            print_green("starting learner loop")
            learner(
                sampling_rng,
                agent,
                replay_buffer,
                demo_buffer=demo_buffer,
                wandb_logger=wandb_logger,
            )

        elif FLAGS.actor:
            sampling_rng = jax.device_put(sampling_rng, sharding)
            data_store = QueuedDataStore(50000)  # the queue size on the actor
            intvn_data_store = QueuedDataStore(50000)

            # actor loop
            print_green("starting actor loop")
            actor(
                agent,
                data_store,
                intvn_data_store,
                env,
                sampling_rng,
            )

        else:
            raise NotImplementedError("Must be either a learner or an actor")
    finally:
        # Without this, the actor's SpaceMouse background process (started by
        # SpacemouseIntervention -> SpaceMouseExpert, a multiprocessing.Manager
        # + daemon Process) is left dangling on Ctrl+C / any exit instead of
        # being torn down here -- same fix eval_act.py already needed for the
        # identical symptom. No-op on the learner side (fake_env=True means
        # SpacemouseIntervention was never wrapped in, and FlexivEnv.close()
        # itself early-returns when fake_env).
        env.close()


if __name__ == "__main__":
    app.run(main)
