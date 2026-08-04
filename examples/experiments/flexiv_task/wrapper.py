"""Flexiv Rizon gym.Env for HIL-SERL, plus a SpaceMouse teleop/intervention wrapper.

Backed entirely by the already-working Flexiv control code in `robotscripts/`
(`flexiv_robot.py`, `collect_demos/flexiv_robot_demo_collector.py`,
`utils/realsense_d405.py`, `utils/spacemouse.py`) rather than reimplementing
robot control here. See ../../../../.. (robotscripts root) for that code.

Not a FrankaEnv subclass: FrankaEnv's `fake_env=True` path still fires one
HTTP call before returning, so it isn't actually hardware-free -- this class
is, which is what lets `train_bc.py`'s pure-BC training path build an agent
from `observation_space`/`action_space` alone with zero hardware connected.
"""

from __future__ import annotations

import sys
import threading
import time
from pathlib import Path

import cv2
import gymnasium as gym
import numpy as np
from pynput import keyboard

_ROBOTSCRIPTS_ROOT = str(Path(__file__).resolve().parents[4])
if _ROBOTSCRIPTS_ROOT not in sys.path:
    sys.path.insert(0, _ROBOTSCRIPTS_ROOT)

# make serl_robot_infra_flexiv packages importable
_SERL_ROBOT_INFRA_FLEXIV = str(Path(__file__).resolve().parents[3] / "serl_robot_infra_flexiv")
if _SERL_ROBOT_INFRA_FLEXIV not in sys.path:
    sys.path.insert(0, _SERL_ROBOT_INFRA_FLEXIV)

from spacemouse_expert import SpaceMouseExpert
from flexiv_api import FlexivRobot
from utils.realsense_d405 import ThreadedRealsenseImageGenerator
from utils.robotiq_gripper import RobotiqGripper

IMAGE_SIZE = 128

# Params for the resetting phase
RESET_POS_TOLERANCE_M = 0.01
RESET_TIMEOUT_S = 10.0


class FlexivEnv(gym.Env):
    """Cartesian-velocity-controlled Flexiv arm, following FrankaEnv's obs/action shape
    conventions closely enough that RelativeFrame/Quat2EulerWrapper/etc. work unmodified.
    """

    def __init__(self, hz: int = 10, fake_env: bool = False, config=None):
        self.config = config
        self.hz = hz
        self.dt = 1.0 / hz
        self.fake_env = fake_env
        self._camera_names = list(config.REALSENSE_SERIALS.keys())
        self.last_step_time = None
        self.last_gripper_act = time.time()
        self.gripper_sleep = 0.6
        self.gripper_open = True

        self.action_space = gym.spaces.Box(-1, 1, shape=(6,), dtype=np.float32)
        self.observation_space = gym.spaces.Dict(
            {
                "state": gym.spaces.Dict(
                    {
                        # xyz + quat (xyzw) -- reindexed from Flexiv's native wxyz on every read
                        "tcp_pose": gym.spaces.Box(-np.inf, np.inf, shape=(7,), dtype=np.float32),
                        "tcp_vel": gym.spaces.Box(-np.inf, np.inf, shape=(6,), dtype=np.float32),
                        "tcp_force": gym.spaces.Box(-np.inf, np.inf, shape=(3,), dtype=np.float32),
                        "tcp_torque": gym.spaces.Box(-np.inf, np.inf, shape=(3,), dtype=np.float32),
                    }
                ),
                "images": gym.spaces.Dict(
                    {
                        name: gym.spaces.Box(0, 255, shape=(IMAGE_SIZE, IMAGE_SIZE, 3), dtype=np.uint8)
                        for name in self._camera_names
                    }
                ),
            }
        )

        self._step_count = 0
        self._terminate = False
        self._quit_session = False
        self._start_recording = False
        self._recording = False  # MAX_EPISODE_LENGTH only bounds the episode once this is True

        if fake_env:
            return

        gripper = RobotiqGripper()
        gripper.activate()
        self.gripper = gripper

        self.robot = FlexivRobot(config.ROBOT_SERIAL, gripper_com_port=None, compliant_z=True)
        self.cameras = ThreadedRealsenseImageGenerator(
            [config.REALSENSE_SERIALS[name] for name in self._camera_names]
        )

        self._listener = keyboard.Listener(on_press=self._on_key_press)
        self._listener.start()

    def _on_key_press(self, key) -> None:
        if key == keyboard.Key.esc:
            self._terminate = True
        elif key == keyboard.Key.shift:
            # Toggle, same as CLUTCH in collect_demos/record_demos.py: while
            # free-moving (not yet recording), starts recording; while
            # already recording, ends the episode early (same effect as
            # hitting MAX_EPISODE_LENGTH or ESC). One-shot: consumed and
            # cleared by the next step() call, same pattern as _quit_session.
            self._start_recording = True
        elif getattr(key, "char", None) == "q":
            # End this episode AND stop collecting after it -- record_demos.py
            # checks info["quit_session"] and breaks its outer loop, saving
            # whatever's been recorded so far instead of requiring
            # --successes_needed to be hit.
            self._terminate = True
            self._quit_session = True

    def _get_obs(self):
        '''
        Returns obs_dict, q
        obs_dict is a dict of the observation representation at this step
        (as specified in the config)
        q is the current joint positions - used for visualizing this demo in sim,
        not for training, so it's not included in obs_dict and instead set via
        the 'info' dict
        '''
        states = self.robot.get_states()
        tcp_pose_in_world_frame = states.tcp_pose
        tcp_vel_in_world_frame = states.tcp_vel
        tcp_wrench_in_tcp_frame = states.ext_wrench_in_tcp
        q = states.q # joint positions in rad

        # print(tcp_wrench_in_tcp_frame)

        tcp_pose_xyzw_world_frame = np.concatenate(
            [
                tcp_pose_in_world_frame[:3],
                [tcp_pose_in_world_frame[4], tcp_pose_in_world_frame[5], tcp_pose_in_world_frame[6], tcp_pose_in_world_frame[3]],
            ]
        ).astype(np.float32)

        raw_images = self.cameras.get_latest_images()
        images = {
            name: cv2.resize(img["rgb"], (IMAGE_SIZE, IMAGE_SIZE))
            for name, img in zip(self._camera_names, raw_images)
        }
        # print(
        #     f"[_get_obs] robot_state={t1 - t0:.4f}s camera_read={t2 - t1:.4f}s "
        #     f"resize={t3 - t2:.4f}s total={t3 - t0:.4f}s"
        # )

        obs = {
            "state": {
                "tcp_pose": tcp_pose_xyzw_world_frame,
                "tcp_vel": np.asarray(tcp_vel_in_world_frame, dtype=np.float32),
                "tcp_force": np.asarray(tcp_wrench_in_tcp_frame[:3], dtype=np.float32),
                "tcp_torque": np.asarray(tcp_wrench_in_tcp_frame[3:], dtype=np.float32),
            },
            "images": images,
        }
        return obs, np.asarray(q, dtype=np.float32)

    def _send_gripper_command(self, pos: float, mode="binary"):
        """Internal function to send gripper command to the robot."""
        if mode == "binary":
            if (pos <= -0.5) and (self.gripper_open) and (time.time() - self.last_gripper_act > self.gripper_sleep):  # close gripper
                print("closing gripper")
                threading.Thread(target=self.gripper.grasp, daemon=True).start()
                self.last_gripper_act = time.time()
                self.gripper_open = False
            elif (pos >= 0.5) and (not self.gripper_open) and (time.time() - self.last_gripper_act > self.gripper_sleep):  # open gripper
                print("opening gripper")
                threading.Thread(target=self.gripper.open, daemon=True).start()
                self.last_gripper_act = time.time()
                self.gripper_open = True
            else: 
                return
        elif mode == "continuous":
            raise NotImplementedError("Continuous gripper control is optional")

    def step(self, action: np.ndarray):
        if self.last_step_time is None:
            self.last_step_time = time.time()

        start_time = time.time()

        this_step_time = time.time()
        action = np.clip(np.asarray(action, dtype=np.float32), -1, 1)
        self.robot.action(action, this_step_time-self.last_step_time) # FlexivRobot handles interfacing with the robot
        self.last_step_time = this_step_time

        gripper_action = action[6] * self.config.ACTION_SCALE[2]
        self._send_gripper_command(gripper_action)

        obs, q = self._get_obs()

        start_recording = self._start_recording
        self._start_recording = False # one-shot: consume it so it doesn't re-fire next step
        if start_recording:
            if not self._recording:
                # when [recording] is hit, begins the episode (either rolling out a policy or recording a demo)
                self._recording = True
                self._step_count = 0
            else:
                self._terminate = True # episode terminates, same effect as hitting MAX_EPISODE_LENGTH or ESC

        # also terminate if tcp goes below the table
        if obs["state"]["tcp_pose"][2] <= self.config.MIN_TCP_Z:
            print(
                f"[FlexivEnv] tcp z={obs['state']['tcp_pose'][2]:.4f} <= "
                f"MIN_TCP_Z={self.config.MIN_TCP_Z} -- terminating episode (table safety)."
            )
            self._terminate = True

        # also terminate if we hit the step count (and we were recording)
        if self._recording and self._step_count >= self.config.MAX_EPISODE_LENGTH:
            self._terminate = True

        self._step_count += 1

        if self._terminate:
            print(f"\t\tpolicy ran for {self._step_count} steps")
            self.robot.send_freeze() # freeze the robot and end the episode

        sleep_time = max(0.0, self.dt - (time.time() - start_time))
        time.sleep(sleep_time)

        return obs, 0.0, self._terminate, False, {
            "succeed": False,
            "quit_session": self._quit_session,
            "start_recording": start_recording,
            "q": q,
        }

    def reset(self, *, seed=None, options=None):
        if self.fake_env:
            raise RuntimeError("Cannot reset a fake_env=True FlexivEnv -- no hardware was connected")

        threading.Thread(target=self.gripper.open, daemon=True).start()
        self.last_gripper_act = time.time()
        self.gripper_open = True
        time.sleep(1.0)
        # move to config.RESET_ORIGIN + some random xyz noise
        print("Moving to reset pose...")
        self._move_to_reset_pose()
        self._step_count = 0
        self._terminate = False
        self._recording = False
        obs, q = self._get_obs()
        return obs, {"q": q}

    def _move_to_reset_pose(self):
        """Blocks until the arm's TCP position is within RESET_POS_TOLERANCE_M
        of config.RESET_ORIGIN (+ noise), repeatedly re-sending the target via
        send_pose -- NRT_CARTESIAN_MOTION_FORCE expects a continuously
        refreshed target rather than a true fire-and-forget move command, same
        reason step() re-sends every tick. Capped to RESET_MOVE_MAX_LIN_VEL/
        RESET_MOVE_MAX_ANG_VEL (well under teleop's ACTION_SCALE) since this
        move runs unsupervised, no human on the SpaceMouse to react."""
        target_pose = np.asarray(self.config.RESET_ORIGIN, dtype=np.float64).copy()
        noise = np.random.uniform(-1.0, 1.0, size=3) * np.asarray(self.config.RESET_POS_NOISE, dtype=np.float64)
        target_pose[:3] += noise

        start = time.time()
        while time.time() - start < RESET_TIMEOUT_S:
            self.robot.send_pose(target_pose)
            current_pos = np.asarray(self.robot.get_current_pose()[:3])
            if np.linalg.norm(current_pos - target_pose[:3]) < RESET_POS_TOLERANCE_M:
                break
            time.sleep(self.dt)
        else:
            print(
                f"[FlexivEnv] warning: didn't reach reset pose within {RESET_TIMEOUT_S}s "
                f"(off by {np.linalg.norm(current_pos - target_pose[:3]):.3f}m) -- continuing anyway."
            )

    def close(self):
        if self.fake_env:
            return
        self.cameras.close()
        self.robot.close()

class SpacemouseIntervention(gym.ActionWrapper):
    def __init__(self, env, action_indices=None):
        super().__init__(env)
        self.gripper_enabled = True
        if self.action_space.shape == (6,):
            self.gripper_enabled = False

        self.expert = SpaceMouseExpert()
        self.left, self.right = False, False
        self.action_indices = action_indices

    def action(self, action: np.ndarray) -> np.ndarray:
        """
        Input:
        - action: policy action
        Output:
        - action: spacemouse action if nonzero; else, policy action
        """
        expert_a, buttons = self.expert.get_action()
        # Only ever want LEFT/RIGHT off one physical SpaceMouse here (this
        # wrapper is single-device only) -- take the first 2 regardless of
        # whether the driver reports 2 buttons (one device) or 4 (a wireless
        # receiver's dongle sometimes enumerates as 2 "devices" for the same
        # physical mouse, per spacemouse_expert.py's len(state)==2 branch).
        self.left, self.right = buttons[0], buttons[1]
        intervened = False
        
        if np.linalg.norm(expert_a) > 0.001:
            intervened = True

        if self.left:  # close gripper
            gripper_action = np.random.uniform(-1, -0.9, size=(1,))
            intervened = True
        elif self.right:  # open gripper
            gripper_action = np.random.uniform(0.9, 1, size=(1,))
            intervened = True
        else:
            gripper_action = np.zeros((1,))
        expert_a = np.concatenate((expert_a, gripper_action), axis=0)
        action = np.concatenate((action, gripper_action), axis=0) # NOTE: for actions with gripper, don't concat here

        if self.action_indices is not None:
            filtered_expert_a = np.zeros_like(expert_a)
            filtered_expert_a[self.action_indices] = expert_a[self.action_indices]
            expert_a = filtered_expert_a

        if intervened:
            return expert_a, True
        
        return action, False

    def step(self, action):
        new_action, replaced = self.action(action)

        obs, rew, done, truncated, info = self.env.step(new_action)
        if replaced:
            action_dim = self.action_space.shape[-1] # truncate action to the action dimension
            new_action_truncated = new_action[:action_dim]
            info["intervene_action"] = new_action_truncated
        info["left"] = self.left
        info["right"] = self.right
        return obs, rew, done, truncated, info

    def close(self):
        self.expert.close() # close the Spacemouse's background daemon process
        return self.env.close()