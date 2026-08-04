import os
from tqdm import tqdm
import numpy as np
import copy
import pickle as pkl
import datetime
from absl import app, flags
import time

from experiments.mappings import CONFIG_MAPPING

FLAGS = flags.FLAGS
flags.DEFINE_string("exp_name", None, "Name of experiment corresponding to folder.")
flags.DEFINE_integer("successes_needed", 20, "Number of successful demos to collect.")
flags.DEFINE_string(
    "output_name", None,
    "Base name for the output .pkl (saved as ./demo_data/<output_name>.pkl -- '.pkl' "
    "appended automatically if you don't include it). Defaults to "
    "'<exp_name>_<timestamp>.pkl' if unset."
)

def save_transitions(transitions, file_name):
    with open(file_name, "wb") as f:
        pkl.dump(transitions, f)


def main(_):
    assert FLAGS.exp_name in CONFIG_MAPPING, 'Experiment folder not found.'
    config = CONFIG_MAPPING[FLAGS.exp_name]()
    env = config.get_environment(fake_env=False, save_video=False, classifier=True)

    obs, info = env.reset()
    print("Reset done")
    transitions = []
    success_count = 0
    success_needed = FLAGS.successes_needed
    pbar = tqdm(total=success_needed)
    zero_action = np.zeros(env.action_space.sample().shape)

    if not os.path.exists("./demo_data"):
        os.makedirs("./demo_data")
    # One stable filename for the whole session -- each successful episode
    # overwrites it in place (not a fresh file per checkpoint), so a crash
    # mid-session loses at most the in-progress episode instead of the whole
    # run, without leaving a pile of separately-timestamped partial files
    # behind. Episode count isn't in the name anymore since a changing count
    # would mean each write goes to a different path -- it's still trivially
    # recoverable from the file itself (len(transitions), or visualize_demos.py).
    if FLAGS.output_name:
        name = FLAGS.output_name if FLAGS.output_name.endswith(".pkl") else f"{FLAGS.output_name}.pkl"
        file_name = f"./demo_data/{name}"
        # save_transitions overwrites file_name in place after every episode --
        # if this name already exists, that would silently destroy whatever's
        # already in it the moment the first episode completes.
        assert not os.path.exists(file_name), (
            f"{file_name} already exists -- pick a different --output_name, or delete/rename "
            "it yourself first if you really mean to overwrite it."
        )
    else:
        uuid = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        file_name = f"./demo_data/{FLAGS.exp_name}_{uuid}.pkl"

    while success_count < success_needed:
        # Free-movement phase: teleoperate with the SpaceMouse to reposition the
        # arm for the next demo -- nothing is recorded yet. Mirrors
        # collect_demos/record_demos.py's own "reposition, then clutch to
        # start" flow. Hold SHIFT to start recording.
        print(
            f"\n[{success_count}/{success_needed}] Move arm into position with "
            "SpaceMouse. Hold SHIFT to start recording."
        )
        while True:
            obs, rew, done, truncated, info = env.step(zero_action)
            if info.get("start_recording"):
                break
            if done:
                # Only reachable via MAX_EPISODE_LENGTH or ESC while still just
                # repositioning (no SHIFT yet). HumanClassifierWrapper prompts
                # "Success? (1/0)" on ANY done regardless of phase -- answer
                # either way, it's discarded here; then keep waiting.
                obs, info = env.reset()

        print("[STARTED]")
        trajectory = []
        returns = 0
        quit_requested = False

        while True:
            actions = zero_action
            next_obs, rew, done, truncated, info = env.step(actions)
            returns += rew
            if "intervene_action" in info:
                actions = info["intervene_action"]
            transition = copy.deepcopy(
                dict(
                    observations=obs,
                    actions=actions,
                    next_observations=next_obs,
                    rewards=rew,
                    masks=1.0 - done,
                    dones=done,
                    infos=info,
                )
            )
            trajectory.append(transition)

            quit_requested = info.get("quit_session", False)
            obs = next_obs
            if done:
                break

        if info["succeed"]:
            for transition in trajectory:
                transitions.append(copy.deepcopy(transition))
            success_count += 1
            pbar.update(1)
            save_transitions(transitions, file_name)
            print(f"[checkpoint] saved {success_count} demo(s) so far to {file_name}")

        if quit_requested:
            print(f"Quit requested -- stopping early with {success_count} demo(s) recorded.")
            break

        obs, info = env.reset()

    save_transitions(transitions, file_name)
    print(f"Done -- saved {success_count} demos to {file_name}")
    env.close()

if __name__ == "__main__":
    app.run(main)