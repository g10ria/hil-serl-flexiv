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
"""

import time

import numpy as np
from absl import app, flags
from gymnasium.wrappers import RecordEpisodeStatistics

from experiments.mappings import CONFIG_MAPPING

FLAGS = flags.FLAGS
flags.DEFINE_string("exp_name", None, "Name of experiment corresponding to folder.")
flags.DEFINE_string("act_checkpoint", None, "Path to a trained LeRobot ACT checkpoint dir.")
flags.DEFINE_integer("eval_n_trajs", 10, "Number of episodes to roll out.")


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


def eval_act(env, policy, preprocessor, postprocessor, image_key_map, n_trajs):
    import torch

    success_counter = 0
    time_list = []
    step_time_list = []
    zero_action = np.zeros(env.action_space.sample().shape)

    for episode in range(n_trajs):
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
            obs = next_obs
            if done:
                if reward:
                    dt = time.time() - start_time
                    time_list.append(dt)
                    print(dt)
                success_counter += reward
                print(reward)
                print(f"{success_counter}/{episode + 1}")

    print(f"success rate: {success_counter / n_trajs}")
    print(f"average time: {np.mean(time_list)}")
    print(f"average step time: {np.mean(step_time_list)}")
    
    return


def main(_):
    assert FLAGS.exp_name in CONFIG_MAPPING, "Experiment folder not found."
    assert FLAGS.act_checkpoint, "--act_checkpoint is required."
    config = CONFIG_MAPPING[FLAGS.exp_name]()
    env = config.get_environment(fake_env=False, save_video=False, classifier=True)
    env = RecordEpisodeStatistics(env)

    policy, preprocessor, postprocessor, image_key_map = load_act_policy(FLAGS.act_checkpoint)

    print("starting ACT actor loop")
    eval_act(env, policy, preprocessor, postprocessor, image_key_map, FLAGS.eval_n_trajs)
    return


if __name__ == "__main__":
    app.run(main)
