import os
import sys
import numpy as np
from absl import app, flags

import jax

from experiments.mappings import CONFIG_MAPPING
import serl_launcher.utils.train_utils  # noqa: F401 -- side effect: jax pickle compat shim for legacy resnet10_params.pkl
from serl_launcher.networks.reward_classifier import load_classifier_func

FLAGS = flags.FLAGS
flags.DEFINE_string("exp_name", None, "Name of experiment corresponding to folder.")
flags.DEFINE_float("threshold", 0.75, "Sigmoid threshold to flag a step as a predicted success.")


def main(_):
    assert FLAGS.exp_name in CONFIG_MAPPING, "Experiment folder not found."
    config = CONFIG_MAPPING[FLAGS.exp_name]()
    # classifier=False: we want to see raw predictions ourselves, not have the
    # env terminate/threshold on them via MultiCameraBinaryRewardClassifierWrapper
    env = config.get_environment(fake_env=False, save_video=False, classifier=False)

    obs, _ = env.reset()
    sample = jax.tree.map(lambda x: np.asarray(x)[None], obs)

    # checkpoint_path is relative -- run this script from inside
    # experiments/<exp_name>/, same convention as train_reward_classifier.py
    classifier_func = load_classifier_func(
        key=jax.random.PRNGKey(0),
        sample=sample,
        image_keys=config.classifier_keys,
        checkpoint_path=os.path.abspath("classifier_ckpt/"),
    )

    print("Teleop the robot with the SpaceMouse. Watch P(success) below. Ctrl+C to quit.")
    try:
        while True:
            actions = np.zeros(env.action_space.sample().shape)
            obs, rew, done, truncated, info = env.step(actions)

            batched_obs = jax.tree.map(lambda x: np.asarray(x)[None], obs)
            logit = classifier_func(batched_obs)
            prob = float(jax.nn.sigmoid(logit).squeeze())

            flagged = prob > FLAGS.threshold
            sys.stdout.write(
                f"\rP(success) = {prob:.3f} {'<-- SUCCESS' if flagged else '           '}"
            )
            sys.stdout.flush()
            if flagged:
                print()  # keep flagged moments in scrollback instead of overwriting them

            if done or truncated:
                print("\nEpisode ended, resetting...")
                obs, _ = env.reset()
    except KeyboardInterrupt:
        print("\nExiting.")
    finally:
        env.close()


if __name__ == "__main__":
    app.run(main)
