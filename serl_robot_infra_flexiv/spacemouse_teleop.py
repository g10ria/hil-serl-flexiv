#!/usr/bin/env python

import argparse
import sys
import threading
import time
from pathlib import Path

# spacemouse_teleop.py -> serl_robot_infra_flexiv -> hil-serl -> robotscripts
_ROBOTSCRIPTS_ROOT = str(Path(__file__).resolve().parents[2])
if _ROBOTSCRIPTS_ROOT not in sys.path:
    sys.path.insert(0, _ROBOTSCRIPTS_ROOT)

# Inserted after (so it ends up first in sys.path, taking precedence) --
# flexiv_robot/flexiv_robot_demo_collector exist in both this folder and
# the root; the sibling copies here are the ones this script should use.
_THIS_DIR = str(Path(__file__).resolve().parent)
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)

from utils.robotiq_gripper import RobotiqGripper, GripperFault


'''
this script allows for teleop of the robot arm(s) using a spacemouse
it directly passes the velocity of the spacemouse through to the arm
'''

from spacemouse_expert import SpaceMouseExpert
import consts
from flexiv_api import FlexivRobot

ENABLED_ARMS = [consts.LEFT_ARM_SERIAL]  # put the serial numbers of arms to enable here
DT = 1.0 / 30.0  # put hz of the controls here

ACTION_TAKEN_EPS = 1e-3  # treat smaller-than-this raw deflection as "not moved"


def _run_gripper_call(gripper, fn, label: str) -> None:
    """Runs a RobotiqGripper call (grasp/open) on a background thread and
    actually surfaces failures, same as FlexivEnv._run_gripper_call in
    flexiv_task/wrapper.py -- without this, a GripperFault raised inside a
    bare threading.Thread just vanishes into Python's default thread-exception
    hook, so nothing indicates whether the gripper physically responded."""
    try:
        fn()
    except GripperFault as e:
        print(f"[gripper] {label} FAILED: {e!r} -- reactivating and retrying...")
        try:
            gripper.activate()
        except Exception as activate_err:
            print(f"[gripper] reactivation failed: {activate_err!r} -- gripper needs a physical check (power/cable/e-stop).")
            return
        try:
            fn()
            print(f"[gripper] {label} succeeded after reactivation.")
        except Exception as retry_err:
            print(f"[gripper] {label} FAILED again after reactivation: {retry_err!r} -- gripper needs a physical check.")
    except Exception as e:
        print(f"[gripper] {label} FAILED: {e!r} -- gripper may need re-activation (restart this script).")

# SpaceMouseExpert starts a multiprocessing.Process internally -- on Windows,
# multiprocessing uses "spawn" (not "fork"), which re-imports this whole
# module in the child process. Without this __main__ guard, that re-import
# hits the module-level SpaceMouseExpert()/arms construction again, starting
# a second process before the first one finishes bootstrapping (the
# RuntimeError this fixes). utils/spacemouse.py's threading-based driver
# never needed this since threads don't re-import the module.
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--no-gripper", action="store_true",
        help="Skip connecting to and activating the gripper, and ignore the spacemouse "
        "button entirely -- for teleop sessions with no gripper physically attached "
        "(RobotiqGripper() opens a serial connection in its constructor, so just "
        "skipping activate() isn't enough to avoid touching the hardware)."
    )
    args = parser.parse_args()

    if len(ENABLED_ARMS) == 0:
        raise ValueError("no arms are specified to be on")

    arms = [FlexivRobot(serial, compliant_z=True) for serial in ENABLED_ARMS]
    current_poses = [arm.get_current_pose() for arm in arms]
    mouse = SpaceMouseExpert()  # defaults to pyspacemouse_windows; kicks off a background process that samples the spacemouse
    gripper = None
    if not args.no_gripper:
        gripper = RobotiqGripper()
        gripper.activate()

    mouse_prev_input = 0
    gripper_closed = False

    prev_time = time.time()

    try:
        while True:
            raw_action, buttons = mouse.get_action()  # raw ~[-1,1] deflection + button list

            now = time.time()
            for i in range(len(arms)):
                arms[i].action(raw_action[:6], now-prev_time)
            prev_time = now

            if gripper is not None:
                if mouse_prev_input == 0 and buttons[0] == 1:
                    if gripper_closed:
                        print("opening gripper")
                        threading.Thread(target=_run_gripper_call, args=(gripper, gripper.open, "opening gripper"), daemon=True).start()
                    else:
                        print("closing gripper")
                        threading.Thread(target=_run_gripper_call, args=(gripper, gripper.grasp, "closing gripper"), daemon=True).start()
                    gripper_closed = not gripper_closed
                    mouse_prev_input = 1
                elif buttons[0] == 0:
                    mouse_prev_input = 0

            time.sleep(DT)
    finally:
        for a in arms:
            a.stop()
