# HELPER - wraps the Flexiv RDK with demo-collection functionality
# (get_observation/action) built directly in.
#
# Flattened merge of robotscripts/flexiv_robot.py +
# robotscripts/collect_demos/flexiv_robot_demo_collector.py (formerly base class
# + subclass, now one class) migrated into hil-serl/serl_robot_infra_flexiv (the
# Flexiv-side analog of hil-serl/serl_robot_infra, which is Franka-only).

from __future__ import annotations

import sys
from pathlib import Path

# flexiv_api.py -> serl_robot_infra_flexiv -> hil-serl -> robotscripts
_ROBOTSCRIPTS_ROOT = str(Path(__file__).resolve().parents[2])
if _ROBOTSCRIPTS_ROOT not in sys.path:
    sys.path.insert(0, _ROBOTSCRIPTS_ROOT)

import signal
import time
import spdlog
from scipy.spatial.transform import Rotation
import flexivrdk as rdk

import numpy as np

from utils.robotiq_gripper import RobotiqGripper
from consts import LEFT_ARM_SERIAL, RIGHT_ARM_SERIAL, ACTION_SCALE, KZ, BZ, MAX_LIN_VEL, MAX_ANG_VEL


def initialize_flexiv_robots(serialNumbers = [LEFT_ARM_SERIAL],
                      defaultModes = [rdk.Mode.NRT_CARTESIAN_MOTION_FORCE]):
    robots = []
    for i in range(len(serialNumbers)):
        robots.append(FlexivRobot(serialNumbers[i], defaultModes[i]))
    return robots

class FlexivRobot:
    def __init__(self,
                 serialNumber,
                 defaultMode = rdk.Mode.NRT_CARTESIAN_MOTION_FORCE,
                 gripper_com_port = None,
                 should_enable = True,
                 home_robot = False,
                 compliant_z = False):
        logger = spdlog.ConsoleLogger(f"[{serialNumber} logger]")
        self.logger = logger
        self.compliant_z = compliant_z

        try:
            robot = rdk.Robot(serialNumber)
            self.robot = robot
            self._mode = defaultMode

            # Clear fault on the connected robot if any
            if robot.fault():
                logger.warn("Fault occurred on the connected robot, trying to clear ...")
                if not robot.ClearFault():
                    logger.error("Fault cannot be cleared, exiting ...")
                    return 1
                logger.info("Fault on the connected robot is cleared")

            # Plug in the gripper if there is one
            if gripper_com_port is not None:
                gripper = RobotiqGripper(serial_port=gripper_com_port)
                gripper.activate()
                self.gripper = gripper
                logger.info(f"Robot now has gripper")
            else: logger.info(f"Robot has no gripper")

            # Everything below only applies to enabled robots
            if should_enable:
                self.enabled = True
                logger.info("Enabling robot ...")
                robot.Enable()
                while not robot.operational():
                    time.sleep(1)
                logger.info("Robot is now operational")

                # Zero the force/torque sensor
                robot.SwitchMode(rdk.Mode.NRT_PRIMITIVE_EXECUTION)
                robot.ExecutePrimitive("ZeroFTSensor", {})
                while not robot.primitive_states().get("terminated", 0):
                    time.sleep(0.2)
                print("F/T sensor zeroed.")

                # Home if homing was true
                if home_robot:
                    robot.SwitchMode(rdk.Mode.NRT_PLAN_EXECUTION)
                    robot.ExecutePlan("PLAN-Home")
                    while robot.busy():
                        time.sleep(0.5)

                if compliant_z:
                    robot.SwitchMode(self._mode.NRT_CARTESIAN_MOTION_FORCE)
                    nominal_k_x = robot.info().K_x_nom # nominal: [10000.0, 10000.0, 10000.0, 1500.0, 1500.0, 1500.0]
                    nominal_k_x[2] = KZ
                    robot.SetCartesianImpedance(nominal_k_x, [0.7, 0.7, 0.8, 0.7, 0.7, 0.7]) # note: z_x damping ratio valid range is [0.3, 0.8]
                    print("set cartesian impedance")
                    # robot.SetForceControlAxis([False, False, True, False, False, False])
                    # robot.SetForceControlFrame(rdk.CoordType.TCP)
                    # robot.SetMaxContactWrench([10.0, 10.0, float("inf"), 3.0, 3.0, 3.0])
                else:
                    robot.SwitchMode(defaultMode)
                self._register_shutdown_handler()

        except Exception as e:
            logger.error(str(e))

        self.current_pose = self.get_current_pose()

    @property
    def r(self):
        return self.robot

    def get_logger(self):
        return self.logger

    # Calling reset() homes the robot and puts it in the provided default mode
    def reset(self) -> None:
        self.robot.SwitchMode(self._mode.NRT_PLAN_EXECUTION)
        self.robot.ExecutePlan("PLAN-Home")
        while self.robot.busy():
            time.sleep(0.2)
        self.robot.SwitchMode(self._mode.NRT_CARTESIAN_MOTION_FORCE)

    def get_states(self):
        return self.robot.states()

    # smoothly moves to the given joint positions using Flexiv's own joint
    # motion generator (NRT_JOINT_POSITION), then switches back to the
    # default cartesian mode. Useful for going to a known starting
    # configuration before starting cartesian velocity control.
    def move_to_joint_positions(self, positions, max_vel=None, max_acc=None, tolerance=0.01, timeout=10.0):
        n = len(positions)
        target_vels = [0.0] * n  # come to rest once we arrive
        max_vel = max_vel if max_vel is not None else [0.1] * n # default max vel/acc
        max_acc = max_acc if max_acc is not None else [0.3] * n

        self.robot.SwitchMode(self._mode.NRT_JOINT_POSITION)
        self.robot.SendJointPosition(list(positions), target_vels, max_vel, max_acc)

        start = time.time()
        while time.time() - start < timeout:
            current_q = np.array(self.robot.states().q)
            if np.max(np.abs(current_q - np.array(positions))) < tolerance:
                break
            time.sleep(0.05)

        self.robot.SwitchMode(self._mode.NRT_CARTESIAN_MOTION_FORCE)

    '''
    HELPERS START --------------------
    '''

    def stop(self):
        if self.robot: self.robot.Stop()

    def get_current_pose(self):
        return list(self.robot.states().tcp_pose) # [x,y,z,qw,qx,qy,qz]

    def local_z_axis_world(self, tcp_pose) -> np.ndarray: # gets the TCP's local z axis in world-frame coordinates
        qw, qx, qy, qz = tcp_pose[3:7]
        rot_mat = Rotation.from_quat([qx, qy, qz, qw]).as_matrix()  # -> scipy [x,y,z,w]
        return rot_mat[:, 2]

    # def disable_impedance(self):
    #     self.robot.SwitchMode(self._mode.NRT_CARTESIAN_MOTION_FORCE)
    #     self.robot.SetForceControlAxis([False, False, False, False, False, False])

    # def enable_fake_impedance_mode(self):
    #     self.robot.SetForceControlAxis([False, False, False, False, False, False])
    #     self.robot.SetForceControlFrame(rdk.CoordType.TCP)
    #     self.robot.SetMaxContactWrench([10.0, 10.0, float("inf"), 3.0, 3.0, 3.0])

    '''
    HELPERS END --------------------
    '''

    '''
    Sends a target pose to the robot with a max velocity and max angular velocity
    Assumes impedance is disabled
    Intended for far-away destination poses with a blocking loop outside of this function
    '''
    def send_pose(self, target_pose):
        pos = list(target_pose[:3])
        quat = np.array(target_pose[3:7], dtype=np.float64)
        quat = quat / np.linalg.norm(quat)
        target_pose = pos + list(quat)
        target_velocity = [0.0] * 6
        self.current_pose = self.get_current_pose()

        self.robot.SendCartesianMotionForce(
            pose=target_pose,
            wrench=[0.0]*6,
            velocity=[0.0]*6,
            max_linear_vel=MAX_LIN_VEL,
            max_angular_vel=MAX_ANG_VEL,
        )

    '''
    Sends a "freeze" command to the robot (target pose = current post, wrench = 0, vel = 0)
    '''
    def send_freeze(self):
        self.robot.SendCartesianMotionForce(
            self.get_current_pose(),
            [0.0] * 6, # wrench
            [0.0] * 6, # vel
            MAX_LIN_VEL,
            MAX_ANG_VEL
        )

    def action(self, action: np.ndarray, dt):
        self.send_posedelta(action, dt)

    '''
    assumes action is of the form [x,y,z,rx,ry,rz] where all of these values
    are pose deltas in the range [-1, 1]
    treats these basically like velocities and rescales by dt so that the
    internal pose tracking works
    '''
    def send_posedelta(self, action: np.ndarray, dt = 1/30.0):
        MAX_DT_VALUE = 1/10.0 # no overshoots from dt buildup
        dt = min(MAX_DT_VALUE, dt)

        # Explicit [3:6], not the open-ended [3:] -- action may be 7-wide
        # (gripper appended as a 7th element by callers like
        # FlexivEnv.step()/SpacemouseIntervention), and an open-ended slice
        # would silently pull the gripper value in as a 4th rotation
        # component instead of raising a clear shape error.
        target_lin_vel = action[:3] * MAX_LIN_VEL
        target_ang_vel = action[3:6] * MAX_ANG_VEL
        target_vels = list(target_lin_vel) + list(target_ang_vel)

        pos_delta = target_lin_vel * dt
        rot_delta = target_ang_vel * dt

        curr_pose = self.current_pose  # [x,y,z,qw,qx,qy,qz]
        curr_pos = np.array(curr_pose[:3])
        curr_quat_wxyz = curr_pose[3:]
        # apply the position delta
        new_pos = curr_pos + pos_delta
        # apply the rotation delta
        curr_quat_xyzw = [curr_quat_wxyz[1], curr_quat_wxyz[2], curr_quat_wxyz[3], curr_quat_wxyz[0]]
        delta_rot = Rotation.from_rotvec(rot_delta)
        new_rot_xyzw = (delta_rot * Rotation.from_quat(curr_quat_xyzw)).as_quat()  # scipy [x,y,z,w]
        new_quat_wxyz = [new_rot_xyzw[3], new_rot_xyzw[0], new_rot_xyzw[1], new_rot_xyzw[2]]  # back to [w,x,y,z]

        # concatenate the target pose
        target_pose = list(new_pos) + new_quat_wxyz
        self.current_pose = target_pose  # store the COMMANDED target, not a hardware readback

        # with np.printoptions(precision=3, suppress=True, floatmode='fixed'):
        #     print(np.array(target_pose), np.array(target_vels))

        # wrench = self.get_impedance_wrench(target_pose) if self.compliant_z else np.zeros(6)
        wrench = np.zeros(6)

        self.robot.SendCartesianMotionForce(
            pose=target_pose,
            wrench=wrench,
            velocity=target_vels,
            max_linear_vel=MAX_LIN_VEL,
            max_angular_vel=MAX_ANG_VEL,
        )

    # '''
    # Retrieves the impedance wrench - used for manually doing "impedance" in the tcp frame
    # using force control
    # '''
    # def get_impedance_wrench(self, target_pose):
    #     tcp_pose = np.array(self.get_current_pose()) # tcp pose in world frame
    #     z_axis_world = self.local_z_axis_world(tcp_pose) # z axis of the tcp in world frame

    #     disp_local_z = float(np.dot(tcp_pose[:3] - target_pose[:3], z_axis_world)) # current pose - resting pose
    #     vel_local_z = float(np.dot(self.r.states().tcp_vel[:3], z_axis_world)) # measured tcp velocity along local z

    #     z_axis_force = KZ * disp_local_z + BZ * vel_local_z
    #     return [0.0, 0.0, z_axis_force, 0.0, 0.0, 0.0]

    # # assumes that the target pose is quite close to the current pose
    # def send_velocity_command_with_z_compliance(self, target_pose, target_velocity, max_linear_vel=None, max_angular_vel=None):
    #     target_wrench = self.get_impedance_wrench(target_pose)
    #     self.robot.SendCartesianMotionForce(
    #         target_pose,
    #         target_wrench,
    #         target_velocity,   # feed-forward velocity
    #         max_linear_vel=max_linear_vel if max_linear_vel is not None else MAX_LIN_VEL,
    #         max_angular_vel=max_angular_vel if max_angular_vel is not None else MAX_ANG_VEL,
    #     )

    # def send_velocity_command(self, target_pose, target_velocity, max_linear_vel=None, max_angular_vel=None):
    #     self.robot.SendCartesianMotionForce(
    #         target_pose,
    #         [0.0] * 6, # wrench
    #         target_velocity,   # feed-forward velocity
    #         max_linear_vel=max_linear_vel if max_linear_vel is not None else MAX_LIN_VEL,
    #         max_angular_vel=max_angular_vel if max_angular_vel is not None else MAX_ANG_VEL,
    #     )

    # def send_velocity(self, current_pose, lin_vel, ang_vel, dt):
    #     # TODO: do i even need this function still
    #     """
    #     current_pose: [x, y, z, qw, qx, qy, qz] - current/last commanded TCP pose
    #     lin_vel: [vx, vy, vz] in m/s
    #     ang_vel: [wx, wy, wz] in rad/s
    #     dt: time since last call, seconds

    #     Returns the new target_pose (feed this back in as current_pose next call).
    #     """

    #     pos = np.array(current_pose[:3])
    #     quat = np.array(current_pose[3:])  # [qw, qx, qy, qz]

    #     # Integrate position
    #     new_pos = pos + np.array(lin_vel) * dt

    #     # Integrate orientation via small-angle rotation
    #     rotvec = np.array(ang_vel) * dt
    #     delta_rot = Rotation.from_rotvec(rotvec)
    #     # convert
    #     current_rot = Rotation.from_quat([quat[1], quat[2], quat[3], quat[0]])  # scipy wants [x,y,z,w]
    #     new_rot = delta_rot * current_rot

    #     # current_rot = Rotation.to
    #     new_quat_xyzw = new_rot.as_quat()
    #     new_quat = [new_quat_xyzw[3], new_quat_xyzw[0], new_quat_xyzw[1], new_quat_xyzw[2]]  # back to [w,x,y,z]

    #     target_pose = list(new_pos) + new_quat
    #     target_velocity = list(lin_vel) + list(ang_vel)

    #     if self.compliant_z:
    #         self.send_velocity_command_with_z_compliance(target_pose=target_pose, target_velocity=target_velocity)
    #     else:
    #         self.send_velocity_command(target_pose=target_pose, target_velocity=target_velocity)

    #     return target_pose

    def close(self) -> None:
        self.robot.Stop()

    def _register_shutdown_handler(self) -> None:
        def _handler(sig, frame):
            print("\nShutting down robot...")
            self.close()
            sys.exit(0)

        signal.signal(signal.SIGINT, _handler)


# running this just prints the current state
if __name__ == "__main__":
    robot = FlexivRobot(serial_number="Rizon 4s-063533")
    robot.reset()
    print(robot.get_observation())
    robot.close()
