"""Copy of hil-serl/serl_robot_infra/franka_env/spacemouse/spacemouse_expert.py,
adapted for this flat package: defaults to the sibling pyspacemouse_windows.py
(the only driver copied in here) instead of franka_env.spacemouse's plain
pyspacemouse.py, which isn't part of this package.
"""

import sys
from pathlib import Path

# So sibling modules in this same folder (pyspacemouse_windows) resolve via a
# plain top-level import regardless of the caller's own sys.path/cwd --
# same defensive pattern used throughout robotscripts/hil-serl's flexiv_task
# integration (e.g. wrapper.py's _ROBOTSCRIPTS_ROOT insertion).
_THIS_DIR = str(Path(__file__).resolve().parent)
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)

import multiprocessing
import numpy as np
import pyspacemouse_windows
from typing import Tuple


class SpaceMouseExpert:
    """
    This class provides an interface to the SpaceMouse.
    It continuously reads the SpaceMouse state and provides
    a "get_action" method to get the latest action and button state.
    """

    def __init__(self, pyspacemouse_module=pyspacemouse_windows):
        # pyspacemouse_module: which driver module to read from -- defaults
        # to this package's own pyspacemouse_windows.py, but callers can pass
        # a drop-in alternate. Just needs to expose the same
        # open()/read_all() module-level functions.
        self.pyspacemouse_module = pyspacemouse_module

        # Manager to handle shared state between processes
        self.manager = multiprocessing.Manager()
        self.latest_data = self.manager.dict()
        self.latest_data["action"] = [0.0] * 6  # Using lists for compatibility
        self.latest_data["buttons"] = [0, 0, 0, 0]

        # Start a process to continuously read the SpaceMouse state. Target is
        # a plain staticmethod, not a bound method -- on Windows,
        # multiprocessing uses "spawn", which pickles the target; a bound
        # method here would require pickling `self`, including self.manager,
        # whose internal state (locks, weakref finalizers) isn't picklable.
        # Only individual Manager proxies like latest_data are meant to cross
        # process boundaries, not the Manager itself. The driver module is
        # passed by NAME (a plain string), not the module object itself --
        # raw module objects aren't picklable at all (TypeError: cannot
        # pickle 'module' object), regardless of platform.
        self.process = multiprocessing.Process(
            target=SpaceMouseExpert._read_spacemouse, args=(self.latest_data, pyspacemouse_module.__name__)
        )
        self.process.daemon = True
        self.process.start()

    @staticmethod
    def _read_spacemouse(latest_data, pyspacemouse_module_name):
        # Re-import by name in THIS process rather than receiving the module
        # object directly (see the pickling note above), which also
        # conveniently solves a second issue: on Windows "spawn", child
        # processes get a fresh module import with no shared state, so
        # calling open() in the parent process wouldn't leave any trace of
        # the opened device handle in this child's copy of the module.
        # (No extra sys.path setup needed here: this staticmethod is only
        # reachable by the child re-importing this module fresh, which reruns
        # the file's own top-of-module sys.path insertion first.)
        import importlib

        pyspacemouse_module = importlib.import_module(pyspacemouse_module_name)
        pyspacemouse_module.open()
        while True:
            state = pyspacemouse_module.read_all()
            action = [0.0] * 6
            buttons = [0, 0, 0, 0]

            if len(state) == 2:
                action = [
                    -state[0].y, state[0].x, state[0].z,
                    -state[0].roll, -state[0].pitch, -state[0].yaw,
                    -state[1].y, state[1].x, state[1].z,
                    -state[1].roll, -state[1].pitch, -state[1].yaw
                ]
                buttons = state[0].buttons + state[1].buttons
            elif len(state) == 1:
                action = [
                    -state[0].y, state[0].x, state[0].z,
                    -state[0].roll, -state[0].pitch, -state[0].yaw
                ]
                buttons = state[0].buttons

            # Update the shared state
            latest_data["action"] = action
            latest_data["buttons"] = buttons

    def get_action(self) -> Tuple[np.ndarray, list]:
        """Returns the latest action and button state of the SpaceMouse."""
        action = self.latest_data["action"]
        buttons = self.latest_data["buttons"]
        return np.array(action), buttons

    def close(self):
        self.process.terminate()
