#!/usr/bin/env python3
"""Real-hardware rollout for a trained LeRobot ACT policy.

Mirrors train_bc.py's --eval_n_trajs flow closely (same env/wrapper chain,
same wait-for-SHIFT-to-start / MIN_TCP_Z safety / success-prompt UX), just
driven by ACT's select_action() instead of BCAgent.sample_actions(). ACT's
output lands directly in the same normalized action space FlexivEnv expects,
since it was trained on the same recorded (already-normalized) actions.

Usage:
    python eval_act.py --exp_name=flexiv_task --eval_n_trajs=10 \
        --act_checkpoint=act_checkpoints/go_to_red_dot/checkpoint_24990

Blind A/B mode: pass --act_checkpoint_2 to compare two checkpoints. Trial
order is shuffled and which checkpoint is running is never printed before or
during a trial -- only revealed (with updated per-checkpoint success rates)
once that trial's result is already in, so setup (e.g. how you preload the
peg into the gripper) can't be subconsciously biased by knowing which policy
is about to act. --eval_n_trajs is the count *per checkpoint* in this mode.

    python eval_act.py --exp_name=flexiv_task --eval_n_trajs=10 \
        --act_checkpoint=act_checkpoints/peg_no_chamfer_100/checkpoint_100000 \
        --act_checkpoint_2=act_checkpoints/peg_no_chamfer_25/checkpoint_100000
"""

import copy
import datetime
import os
import pickle as pkl
import random
import time
from collections import defaultdict

import numpy as np
from absl import app, flags
from gymnasium.wrappers import RecordEpisodeStatistics

from experiments.mappings import CONFIG_MAPPING

FLAGS = flags.FLAGS
flags.DEFINE_string("exp_name", None, "Name of experiment corresponding to folder.")
flags.DEFINE_string("act_checkpoint", None, "Path to a trained LeRobot ACT checkpoint dir.")
flags.DEFINE_string(
    "act_checkpoint_2", None,
    "Optional second checkpoint dir. When set, runs a blind A/B comparison "
    "against --act_checkpoint instead of a single-policy eval -- see module "
    "docstring."
)
flags.DEFINE_integer(
    "eval_n_trajs", 10,
    "Number of episodes to roll out. Per-checkpoint count when "
    "--act_checkpoint_2 is set (so total trials = 2x this)."
)
flags.DEFINE_boolean(
    "save_interventions", False,
    "Record SpaceMouse interventions made during rollout as new episodes in a "
    "separate demo_data/*.pkl -- each contiguous stretch of intervened steps "
    "becomes its own episode (dones=True on its last step), same schema "
    "record_demos.py produces, so it's usable with train_bc.py/train_bc_act.py/"
    "visualize_demos.py/eval_bc_mse.py directly."
)
flags.DEFINE_string(
    "intervention_output_name", None,
    "Base name for the saved intervention .pkl (./demo_data/<name>.pkl, '.pkl' "
    "appended automatically). Defaults to '<exp_name>_interventions_<timestamp>.pkl' "
    "if unset. Ignored unless --save_interventions is set."
)


def load_act_policy(checkpoint_path: str):
    """Loads a LeRobot ACT checkpoint plus its bundled pre/post-processor
    pipelines (normalization stats baked in at train time)."""
    from lerobot.policies.act.modeling_act import ACTPolicy
    from lerobot.policies import make_pre_post_processors

    policy = ACTPolicy.from_pretrained(checkpoint_path)
    policy.eval()
    preprocessor, postprocessor = make_pre_post_processors(policy.config, pretrained_path=checkpoint_path)

    # e.g. "observation.image.wrist" -> "wrist" -- FlexivEnv's obs dict keys
    # cameras by their short HIL-SERL name (REALSENSE_SERIALS keys).
    image_key_map = {key: key.rsplit(".", 1)[-1] for key in policy.config.image_features}
    return policy, preprocessor, postprocessor, image_key_map


def build_act_obs(image_key_map: dict, obs: dict):
    """Converts FlexivEnv's obs dict (post-ChunkingWrapper, so state/images
    carry a leading obs_horizon=1 dim) into LeRobot's expected input dict."""
    import torch

    state = np.asarray(obs["state"]).flatten().astype(np.float32)
    lerobot_obs = {"observation.state": torch.from_numpy(state)}
    for lerobot_key, cam_name in image_key_map.items():
        img = np.asarray(obs[cam_name])
        if img.ndim == 4:
            img = img[0]  # drop ChunkingWrapper's leading dim
        img_chw01 = (img.astype(np.float32) / 255.0).transpose(2, 0, 1)  # HWC RGB uint8 -> CHW float [0,1]
        # Pre-batch ourselves: LeRobot's automatic batch-dim step only
        # recognizes the exact key "observation.image" or "observation.images.*"
        # (plural) -- this checkpoint's per-camera singular-"image" keys match
        # neither, so it silently leaves them unbatched otherwise.
        lerobot_obs[lerobot_key] = torch.from_numpy(img_chw01).unsqueeze(0)
    return lerobot_obs


def _save_intervention_episodes(episodes: list, output_path: str) -> None:
    """Flattens a list of episodes (each a list of transitions) into one flat
    list and overwrites output_path -- same overwrite-in-place-per-checkpoint
    pattern record_demos.py uses, so a crash mid-eval loses at most the
    in-progress intervention, not everything saved so far."""
    flat = [t for episode in episodes for t in episode]
    with open(output_path, "wb") as f:
        pkl.dump(flat, f)


def eval_act(env, trials, save_interventions=False, intervention_output_path=None):
    """Runs the given list of trials in order.

    Each entry in `trials` is a dict with keys policy/preprocessor/postprocessor/
    image_key_map (which checkpoint drives this episode) and label. label is
    None for a normal single-policy eval; for a blind A/B comparison it's the
    checkpoint path, and is only ever printed *after* that trial's result is
    known -- never before or during -- so which policy is about to run can't
    bias physical setup (e.g. peg preloading).
    """
    import torch

    n_trajs = len(trials)
    success_counter = 0
    time_list = []
    step_time_list = []
    zero_action = np.zeros(env.action_space.sample().shape)

    # Per-label running tallies, only used/printed when trials carry labels.
    label_success = defaultdict(int)
    label_total = defaultdict(int)

    # Each contiguous run of intervened steps becomes its own saved "episode"
    # (dones=True stamped on its last transition) -- saved_episodes accumulates
    # finalized runs; current_run is the one still being built. Shared across
    # every trial regardless of which checkpoint is currently active, since a
    # SpaceMouse intervention overrides whichever policy is running the same way.
    saved_episodes = []
    current_run = []

    def finalize_current_run():
        nonlocal current_run
        if not current_run:
            return
        current_run[-1]["dones"] = True
        current_run[-1]["masks"] = 0.0
        saved_episodes.append(current_run)
        current_run = []
        if save_interventions:
            _save_intervention_episodes(saved_episodes, intervention_output_path)
            print(f"[save_interventions] saved {len(saved_episodes)} intervention episode(s) to {intervention_output_path}")

    for episode, trial in enumerate(trials):
        policy = trial["policy"]
        preprocessor = trial["preprocessor"]
        postprocessor = trial["postprocessor"]
        image_key_map = trial["image_key_map"]
        label = trial["label"]

        obs, _ = env.reset()
        policy.reset()  # clears ACT's internal action-chunk queue -- must happen every episode

        # Wait for a SHIFT press before letting the policy act -- same pattern
        # train_bc.py's eval() uses. zero_action still passes through
        # FlexivSpacemouseIntervention, so touching the SpaceMouse during this
        # wait repositions the arm same as during collection.
        print(f"[eval {episode}/{n_trajs}] Hold SHIFT to let the policy start.")
        while True:
            obs, _, wait_done, _, info = env.step(zero_action)
            if info.get("start_recording"):
                break
            if wait_done:
                obs, _ = env.reset()
                policy.reset()

        done = False
        start_time = time.time()
        while not done:
            lerobot_obs = preprocessor(build_act_obs(image_key_map, obs))
            with torch.no_grad():
                action = policy.select_action(lerobot_obs)
            action = postprocessor(action).squeeze(0).numpy()

            step_pre = time.time()
            next_obs, reward, done, truncated, info = env.step(action)
            step_post = time.time()
            step_time_list.append(step_post-step_pre)

            if save_interventions:
                if "intervene_action" in info:
                    transition = dict(
                        observations=obs,
                        actions=info["intervene_action"],
                        next_observations=next_obs,
                        rewards=reward,
                        masks=1.0 - done,
                        dones=done,
                        infos=info,
                    )
                    current_run.append(copy.deepcopy(transition))
                else:
                    finalize_current_run()  # no-op if nothing's buffered
                if done:
                    finalize_current_run()

            obs = next_obs
            if done:
                if reward:
                    dt = time.time() - start_time
                    time_list.append(dt)
                    print(dt)
                success_counter += reward
                print(reward)
                print(f"{success_counter}/{episode + 1}")

                if label is not None:
                    label_total[label] += 1
                    label_success[label] += reward
                    print(f"[REVEAL] this trial ran checkpoint: {label}")
                    for lbl in sorted(label_total):
                        tot = label_total[lbl]
                        suc = label_success[lbl]
                        print(f"  {lbl}: {suc}/{tot} ({suc / tot:.1%})")

    print(f"success rate: {success_counter / n_trajs}")
    print(f"average time: {np.mean(time_list)}")
    print(f"average step time: {np.mean(step_time_list)}")
    if label_total:
        print("final per-checkpoint results:")
        for lbl in sorted(label_total):
            tot = label_total[lbl]
            suc = label_success[lbl]
            print(f"  {lbl}: {suc}/{tot} ({suc / tot:.1%})")

    return


def main(_):
    assert FLAGS.exp_name in CONFIG_MAPPING, "Experiment folder not found."
    assert FLAGS.act_checkpoint, "--act_checkpoint is required."
    config = CONFIG_MAPPING[FLAGS.exp_name]()
    env = config.get_environment(fake_env=False, save_video=False, classifier=True)
    env = RecordEpisodeStatistics(env)

    if FLAGS.act_checkpoint_2:
        assert FLAGS.act_checkpoint_2 != FLAGS.act_checkpoint, (
            "--act_checkpoint_2 must differ from --act_checkpoint for a blind A/B comparison."
        )
        policy_a, preprocessor_a, postprocessor_a, image_key_map_a = load_act_policy(FLAGS.act_checkpoint)
        policy_b, preprocessor_b, postprocessor_b, image_key_map_b = load_act_policy(FLAGS.act_checkpoint_2)
        trials = (
            [dict(policy=policy_a, preprocessor=preprocessor_a, postprocessor=postprocessor_a,
                  image_key_map=image_key_map_a, label=FLAGS.act_checkpoint)] * FLAGS.eval_n_trajs
            + [dict(policy=policy_b, preprocessor=preprocessor_b, postprocessor=postprocessor_b,
                     image_key_map=image_key_map_b, label=FLAGS.act_checkpoint_2)] * FLAGS.eval_n_trajs
        )
        random.shuffle(trials)
        print(
            f"[blind A/B] {len(trials)} trials total ({FLAGS.eval_n_trajs} each), order shuffled. "
            "Which checkpoint is running is revealed only after each trial completes."
        )
    else:
        policy, preprocessor, postprocessor, image_key_map = load_act_policy(FLAGS.act_checkpoint)
        trials = [dict(policy=policy, preprocessor=preprocessor, postprocessor=postprocessor,
                        image_key_map=image_key_map, label=None)] * FLAGS.eval_n_trajs

    intervention_output_path = None
    if FLAGS.save_interventions:
        if not os.path.exists("./demo_data"):
            os.makedirs("./demo_data")
        if FLAGS.intervention_output_name:
            name = FLAGS.intervention_output_name if FLAGS.intervention_output_name.endswith(".pkl") else f"{FLAGS.intervention_output_name}.pkl"
        else:
            uuid = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            name = f"{FLAGS.exp_name}_interventions_{uuid}.pkl"
        intervention_output_path = f"./demo_data/{name}"
        # Overwritten in place after every finalized intervention episode --
        # if this name already exists, that would silently destroy whatever's
        # already in it the moment the first intervention episode completes.
        assert not os.path.exists(intervention_output_path), (
            f"{intervention_output_path} already exists -- pick a different "
            "--intervention_output_name, or delete/rename it yourself first."
        )
        print(f"[save_interventions] will save to {intervention_output_path}")

    print("starting ACT actor loop")
    try:
        eval_act(
            env, trials,
            save_interventions=FLAGS.save_interventions,
            intervention_output_path=intervention_output_path,
        )
    finally:
        # Without this, the SpaceMouse's background process (still writing to
        # its multiprocessing.Manager dict) only gets cleaned up by interpreter
        # shutdown, which races against the Manager itself tearing down
        # (BrokenPipeError/EOFError after the run has already finished/printed
        # its results) -- env.close() reaches SpacemouseIntervention.close(),
        # which terminates that process cleanly first.
        env.close()
    return


if __name__ == "__main__":
    app.run(main)
