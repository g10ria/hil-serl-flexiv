"""Flexiv Rizon task config, mirroring ram_insertion/usb_pickup_insertion's pattern.

Fill in EnvConfig.REALSENSE_SERIALS (and tune ACTION_SCALE) before use.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List, Tuple

_ROBOTSCRIPTS_ROOT = str(Path(__file__).resolve().parents[4])
if _ROBOTSCRIPTS_ROOT not in sys.path:
    sys.path.insert(0, _ROBOTSCRIPTS_ROOT)

from utils.consts import LEFT_ARM_SERIAL

from franka_env.envs.relative_env import RelativeFrame
from franka_env.envs.wrappers import HumanClassifierWrapper, Quat2EulerWrapper
from serl_launcher.wrappers.chunking import ChunkingWrapper
from serl_launcher.wrappers.serl_obs_wrappers import SERLObsWrapper

from experiments.config import DefaultTrainingConfig
from experiments.flexiv_task.wrapper import FlexivEnv, SpacemouseIntervention


class EnvConfig():
    ROBOT_SERIAL: str = LEFT_ARM_SERIAL
    REALSENSE_SERIALS: Dict[str, str] = {
        "wrist": "230322277032",  # RealSense serial number(s)
    }
    ACTION_SCALE: Tuple[float, float, float] = (0.05, 1.0, 1)
    MAX_EPISODE_LENGTH: int = 600
    HZ: int = 30

    RESET_ORIGIN: List[float] = [0.5916, -0.0251, 0.2566, 0.0284, 0.7191, 0.6936, -0.0317]
    # Max +/- uniform noise (meters), added independently to x, y, z each
    RESET_POS_NOISE: Tuple[float, float, float] = (0.05, 0.05, 0.05)

    MIN_TCP_Z: float = -0.01 # table safety cutoff in world-frame z axis


class TrainConfig(DefaultTrainingConfig):
    image_keys: List[str] = list(EnvConfig.REALSENSE_SERIALS.keys())
    proprio_keys: List[str] = ["tcp_pose", "tcp_vel", "tcp_force", "tcp_torque"] # no gripper for now
    encoder_type: str = "resnet-pretrained"
    setup_mode: str = "single-arm-fixed-gripper"

    def get_environment(self, fake_env=False, save_video=False, classifier=False):
        env = FlexivEnv(hz=EnvConfig.HZ, fake_env=fake_env, config=EnvConfig())
        if not fake_env:
            env = SpacemouseIntervention(env)
        env = RelativeFrame(env)
        env = Quat2EulerWrapper(env)
        env = SERLObsWrapper(env, proprio_keys=self.proprio_keys)
        env = ChunkingWrapper(env, obs_horizon=1, act_exec_horizon=None)
        if classifier:
            env = HumanClassifierWrapper(env)
        return env
