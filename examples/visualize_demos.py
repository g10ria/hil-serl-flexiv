#!/usr/bin/env python3
"""Play back a recorded flexiv_task demo pickle (hil-serl/examples/demo_data/*.pkl)
to sanity-check that the camera(s), proprioception, and actions were actually
recorded correctly -- no dependency on gymnasium/jax/the wrapper chain, this
just reads the saved transitions directly.

Also renders the arm itself (MuJoCo, offscreen) from each step's recorded
joint positions (`infos["q"]`), next to the camera feed -- demos recorded
before that field was added to FlexivEnv just show a placeholder there.

Usage:
    python visualize_demos.py                                    # most recent file in ./demo_data
    python visualize_demos.py --path demo_data/flexiv_task_5_demos_....pkl
    python visualize_demos.py --episode 2 --start_step 40 --scale 6
    python visualize_demos.py --noshow_robot                     # camera feed only
    python visualize_demos.py --exp_name flexiv_task --policy_checkpoint bc_checkpoints/flexiv_task_test
        # also overlays the policy's predicted action distribution (mean +/- std)
        # at each step, so you can step around and see whether it fluctuates.

Controls:
    SPACE       play / pause
    d / a       step forward / backward by 1 (pauses)
    D / A       step forward / backward by 10 (pauses)
    ESC         quit
"""

import glob
import os
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")  # offscreen rendering only, no GUI backend needed
import matplotlib.pyplot as plt
import mujoco
import numpy as np
import pickle as pkl
from absl import app, flags

FLAGS = flags.FLAGS
flags.DEFINE_string(
    "path", None, "Path to a demo_data/*.pkl file. Defaults to the most recently written file in ./demo_data/."
)
flags.DEFINE_integer("episode", None, "Only load this episode index (default: all episodes in the file).")
flags.DEFINE_integer("fps", 10, "Auto-play frame rate.")
flags.DEFINE_list("image_keys", None, "Which observation image keys to show (default: all non-'state' keys).")
flags.DEFINE_integer("scale", 5, "Display scale factor for each camera image (recorded frames are small, e.g. 128x128).")
flags.DEFINE_integer("start_step", 0, "Start paused at this step index (0-based, across all loaded episodes concatenated).")
flags.DEFINE_boolean("show_robot", True, "Render the arm (MuJoCo, from recorded joint positions) alongside the camera feed.")
flags.DEFINE_string(
    "mjcf_path", None, "Path to the MuJoCo model xml. Defaults to <robotscripts root>/assets/Rizon4s.xml."
)
flags.DEFINE_float("cam_distance", 1.5, "Robot-view camera distance from the lookat point.")
flags.DEFINE_float("cam_azimuth", -90, "Robot-view camera azimuth, degrees.")
flags.DEFINE_float("cam_elevation", -10, "Robot-view camera elevation, degrees (negative looks down).")
flags.DEFINE_list("cam_lookat", ["0.1", "0", "0.6"], "Robot-view camera lookat point, as 'x,y,z'.")
flags.DEFINE_boolean("show_force_torque", True, "Show a force/torque-vs-step chart below the camera/robot views.")
flags.DEFINE_string(
    "exp_name", None,
    "Experiment name -- required if --policy_checkpoint is set (builds the same "
    "observation/action space the checkpoint was trained with), and also used to look "
    "up ACTION_SCALE so displayed actions show real m/s / rad/s instead of the "
    "normalized [-1,1]-ish values actually recorded/predicted. Without it, actions "
    "display in those raw normalized units."
)
flags.DEFINE_string(
    "policy_checkpoint", None,
    "Path to a trained BC checkpoint dir. If set, overlays the policy's predicted "
    "action distribution (mean +/- std) at the current step -- step around to see "
    "whether it behaves consistently or fluctuates wildly frame to frame."
)
flags.DEFINE_integer("policy_seed", 42, "Seed for BCAgent init -- doesn't affect the deterministic mean shown.")
flags.DEFINE_string(
    "act_checkpoint", None,
    "Path to a trained LeRobot ACT checkpoint dir (containing config.json, "
    "model.safetensors, policy_preprocessor.json, policy_postprocessor.json). If set, "
    "overlays ACT's predicted action (first step of its predicted chunk) alongside the "
    "BC policy's, if any."
)

_ROBOTSCRIPTS_ROOT = Path(__file__).resolve().parents[2]  # visualize_demos.py -> examples -> hil-serl -> robotscripts
DEFAULT_MJCF_PATH = _ROBOTSCRIPTS_ROOT / "assets" / "Rizon4s.xml"

# flexiv_task's proprio_keys = ["tcp_pose", "tcp_vel", "tcp_force", "tcp_torque"]
# (config.py) -- BUT gymnasium.spaces.Dict silently re-sorts its keys
# ALPHABETICALLY at construction time (SERLObsWrapper builds
# gym.spaces.Dict({key: ... for key in proprio_keys}), and Dict.__init__ sorts
# regardless of insertion order), so the actual flattened layout is NOT
# declaration order -- it's tcp_force(3) + tcp_pose(6, xyz+euler) +
# tcp_torque(3) + tcp_vel(6) = 18. Verified directly against gymnasium's
# flatten() with labeled dummy values, not assumed. If you change
# proprio_keys, recompute these (alphabetical order of whatever keys remain).
PROPRIO_FORCE_SLICE = slice(0, 3)
PROPRIO_POSE_SLICE = slice(3, 9)
PROPRIO_TORQUE_SLICE = slice(9, 12)
PROPRIO_VEL_SLICE = slice(12, 18)


def find_latest_demo_file() -> str:
    files = glob.glob(os.path.join("demo_data", "*.pkl"))
    if not files:
        raise FileNotFoundError(
            "No .pkl files found in ./demo_data/ -- pass --path explicitly, "
            "or run this from the directory containing demo_data/ (hil-serl/examples)."
        )
    return max(files, key=os.path.getmtime)


def split_into_episodes(transitions: list) -> list:
    """record_demos.py saves one flat list of transitions with all accepted
    episodes concatenated together -- split back into per-episode lists using
    each transition's `dones` flag."""
    episodes, current = [], []
    for t in transitions:
        current.append(t)
        if t["dones"]:
            episodes.append(current)
            current = []
    if current:  # tolerate a trailing partial episode (shouldn't normally happen)
        episodes.append(current)
    return episodes


def extract_force_torque(episode: list):
    """(N,3) force and (N,3) torque arrays across one episode, pulled straight
    out of the recorded proprio vector (see PROPRIO_*_SLICE above) -- these are
    genuinely part of the trained policy's input, unlike `q`, so they're
    already sitting in observations["state"] rather than needing an info-dict
    side channel."""
    proprio = np.stack([np.asarray(t["observations"]["state"]).flatten() for t in episode])
    if proprio.shape[1] < 18:
        print(f"[warn] proprio has {proprio.shape[1]} dims, expected >=18 for the assumed "
              "tcp_pose(6)+tcp_vel(6)+tcp_force(3)+tcp_torque(3) layout -- force/torque chart may be wrong.")
    return proprio[:, PROPRIO_FORCE_SLICE], proprio[:, PROPRIO_TORQUE_SLICE]


def render_force_torque_base(forces: np.ndarray, torques: np.ndarray, width: int, height: int, dpi: int = 100):
    """Renders the static force/torque traces (axes, gridlines, legend -- everything
    that doesn't change between steps) ONCE per episode. The per-frame "current step"
    marker is drawn separately with cv2.line on top of this, since re-invoking
    matplotlib's full draw pipeline (figure/axes/legend layout) every single frame was
    far too slow for smooth scrubbing -- it starved the OpenCV window's paint loop, so
    playback looked frozen while a key was held and only caught up on release.

    Returns (base_image_bgr, step_to_px, (y0_top, y1_top), (y0_bottom, y1_bottom)):
    step_to_px maps a step index to an x pixel column, and the two y-ranges are the
    vertical pixel extents of the force/torque subplots respectively, so the caller can
    draw a marker line spanning just each subplot (not the gap/labels between them).
    """
    fig, axes = plt.subplots(2, 1, figsize=(width / dpi, height / dpi), dpi=dpi, sharex=True)
    steps = np.arange(len(forces))
    for i, label in enumerate(["x", "y", "z"]):
        axes[0].plot(steps, forces[:, i], label=label, linewidth=1)
        axes[1].plot(steps, torques[:, i], label=label, linewidth=1)
    for ax, ylabel in [(axes[0], "force (N)"), (axes[1], "torque (Nm)")]:
        ax.set_ylabel(ylabel, fontsize=8)
        ax.legend(loc="upper right", fontsize=6, ncol=3)
        ax.grid(alpha=0.3)
        ax.tick_params(labelsize=7)
    axes[1].set_xlabel("step", fontsize=8)
    fig.tight_layout()
    fig.canvas.draw()

    buf = np.asarray(fig.canvas.buffer_rgba())
    img = np.ascontiguousarray(cv2.cvtColor(buf, cv2.COLOR_RGBA2BGR))

    fig_h_px = fig.canvas.get_width_height()[1]
    x0_data, x1_data = axes[0].get_xlim()
    bbox0 = axes[0].get_window_extent()
    bbox1 = axes[1].get_window_extent()
    px_left, px_right = bbox0.x0, bbox0.x1
    y_range_top = (int(fig_h_px - bbox0.y1), int(fig_h_px - bbox0.y0))
    y_range_bottom = (int(fig_h_px - bbox1.y1), int(fig_h_px - bbox1.y0))
    plt.close(fig)

    def step_to_px(step):
        frac = (step - x0_data) / (x1_data - x0_data) if x1_data != x0_data else 0.0
        return int(round(px_left + frac * (px_right - px_left)))

    return img, step_to_px, y_range_top, y_range_bottom


_force_torque_base_cache = {}


def get_force_torque_base(ep_i: int, forces: np.ndarray, torques: np.ndarray, width: int, height: int):
    key = (ep_i, width, height)
    if key not in _force_torque_base_cache:
        _force_torque_base_cache[key] = render_force_torque_base(forces, torques, width, height)
    return _force_torque_base_cache[key]


def load_policy_agent(exp_name: str, checkpoint_path: str, seed: int):
    """Builds the same BC agent architecture train_bc.py trains and restores a
    checkpoint into it. Imports are local/lazy -- jax/flax and the whole
    experiments config chain are only needed when --policy_checkpoint is
    actually used, so plain demo review doesn't require any of it installed."""
    import jax
    from flax.training import checkpoints
    from experiments.mappings import CONFIG_MAPPING
    from serl_launcher.utils.launcher import make_bc_agent

    config = CONFIG_MAPPING[exp_name]()
    env = config.get_environment(fake_env=True, save_video=False, classifier=True)
    bc_agent = make_bc_agent(
        seed=seed,
        sample_obs=env.observation_space.sample(),
        sample_action=env.action_space.sample(),
        image_keys=config.image_keys,
        encoder_type=config.encoder_type,
    )
    ckpt_dir = os.path.abspath(checkpoint_path)
    # restore_checkpoint silently no-ops (returns the untouched target) if
    # nothing's found at this path -- that would mean displaying predictions
    # from a randomly initialized head with no error at all, so check first.
    if checkpoints.latest_checkpoint(ckpt_dir) is None:
        raise FileNotFoundError(f"No checkpoint found at {ckpt_dir}")
    ckpt = checkpoints.restore_checkpoint(ckpt_dir, bc_agent.state)
    return bc_agent.replace(state=ckpt)


def load_action_scale(exp_name: str):
    """Returns (action_scale, action_dim).
    action_scale: 6-vector (lin,lin,lin,ang,ang,ang) that converts a normalized
    [-1,1]-ish action into the real physical units (m/s linear, rad/s
    angular) actually sent to the robot -- same ACTION_SCALE
    FlexivEnv.step() multiplies by.
    action_dim: the exp_name's actual env.action_space width -- used to
    truncate recorded/predicted actions that are wider than the current
    config (e.g. a leftover gripper dim from a since-reverted action space),
    same trimming eval_bc_mse.py does before stacking/comparing actions.
    fake_env=True, so no hardware needed."""
    from experiments.mappings import CONFIG_MAPPING

    config = CONFIG_MAPPING[exp_name]()
    env = config.get_environment(fake_env=True, save_video=False, classifier=False)
    lin, ang, grip = env.unwrapped.config.ACTION_SCALE
    return np.array([lin, lin, lin, ang, ang, ang]), env.action_space.shape[-1]


def predict_policy_action(bc_agent, obs):
    """Returns (mean, std) of the policy's predicted action distribution for
    one observation -- mean doubles as the deterministic action (dist.mode()
    equals the mean for the MultivariateNormalDiag this agent uses)."""
    import jax

    batched_obs = jax.tree.map(lambda x: np.asarray(x)[None], obs)  # add a batch dim
    dist = bc_agent.forward_policy(batched_obs, temperature=1.0)
    mean = np.asarray(jax.device_get(dist.mean()))[0]
    std = np.asarray(jax.device_get(dist.stddev()))[0]
    return mean, std


def load_act_policy(checkpoint_path: str):
    """Loads a LeRobot ACT checkpoint plus its bundled pre/post-processor
    pipelines (normalization stats baked in at train time -- no dataset_stats
    need to be supplied separately). Imports are local/lazy -- torch/lerobot
    are only needed when --act_checkpoint is actually used."""
    from lerobot.policies.act.modeling_act import ACTPolicy
    from lerobot.policies import make_pre_post_processors

    policy = ACTPolicy.from_pretrained(checkpoint_path)
    policy.eval()
    preprocessor, postprocessor = make_pre_post_processors(policy.config, pretrained_path=checkpoint_path)

    # e.g. "observation.image.wrist" -> "wrist" -- our own obs dict keys
    # cameras by their short HIL-SERL name (REALSENSE_SERIALS keys), so this
    # maps each of the checkpoint's expected image features back to that.
    image_key_map = {}
    for lerobot_key in policy.config.image_features:
        cam_name = lerobot_key.rsplit(".", 1)[-1]
        image_key_map[lerobot_key] = cam_name

    return policy, preprocessor, postprocessor, image_key_map


def predict_act_action(policy, preprocessor, postprocessor, image_key_map, obs):
    """Returns ACT's predicted action (first step of its predicted chunk) for
    one observation, as a plain numpy array in the same normalized units
    everything else in this file uses -- ACT was trained directly on the
    HIL-SERL-normalized recorded actions, so its (unnormalized-by-LeRobot)
    output lands in that same space already, no extra conversion needed."""
    import torch

    state = np.asarray(obs["state"]).flatten().astype(np.float32)
    lerobot_obs = {"observation.state": torch.from_numpy(state)}
    for lerobot_key, cam_name in image_key_map.items():
        img = to_display_frame(obs[cam_name])  # (H,W,3) BGR uint8, same helper the camera panel uses
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_chw01 = (img.astype(np.float32) / 255.0).transpose(2, 0, 1)
        # Pre-batch ourselves: LeRobot's automatic batch-dim step only
        # recognizes the exact key "observation.image" or "observation.images.*"
        # (plural) -- this checkpoint's per-camera singular-"image" keys don't
        # match either pattern, so it silently leaves them unbatched otherwise.
        lerobot_obs[lerobot_key] = torch.from_numpy(img_chw01).unsqueeze(0)

    lerobot_obs = preprocessor(lerobot_obs)
    with torch.no_grad():
        chunk = policy.predict_action_chunk(lerobot_obs)
    action = postprocessor(chunk[:, 0, :])
    return action.squeeze(0).numpy()


def build_frame_records(episodes: list) -> list:
    """Flatten (episode, step) transitions into one list, so forward/backward
    navigation is a single index and can cross episode boundaries for free."""
    records = []
    for ep_i, episode in enumerate(episodes):
        for step_i, transition in enumerate(episode):
            records.append((ep_i, step_i, len(episode), transition))
    return records


def to_display_frame(img) -> np.ndarray:
    # Observation images are (1, H, W, 3) uint8 RGB -- the leading dim is
    # ChunkingWrapper's obs_horizon=1 stacking.
    img = np.asarray(img)
    if img.ndim == 4:
        img = img[0]
    return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)


class RobotRenderer:
    """Offscreen-renders the Rizon4s from recorded joint positions, for compositing
    next to the camera feed. mj_forward computes kinematics only -- this poses the
    model to match what actually happened, it doesn't simulate anything."""

    def __init__(
        self,
        mjcf_path: str,
        size: int = 480,
        distance: float = 1.5,
        azimuth: float = -90,
        elevation: float = -10,
        lookat=(0.1, 0.0, 0.6),
        tcp_site: str = "tcp_site",
    ):
        self.model = mujoco.MjModel.from_xml_path(str(mjcf_path))
        # Rizon4s.xml doesn't define its own lights; the default headlight is
        # fairly dim and leaves the arm looking flat/dark grey on black.
        self.model.vis.headlight.ambient[:] = [0.4, 0.4, 0.4]
        self.model.vis.headlight.diffuse[:] = [0.8, 0.8, 0.8]
        self.data = mujoco.MjData(self.model)
        self.renderer = mujoco.Renderer(self.model, height=size, width=size)
        self.size = size

        # Used to convert recorded/predicted actions (in the TCP's own body
        # frame -- see RelativeFrame) into world frame for display, via
        # forward kinematics from the recorded joint positions `q`. Rizon4s.xml
        # defines this site right at the Flexiv TCP convention's origin.
        self.tcp_site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, tcp_site)
        if self.tcp_site_id < 0:
            print(f"[RobotRenderer] warning: no site named {tcp_site!r} in the model -- "
                  "world-frame action display will be unavailable.")

        # Defaults tuned against actual link positions (checked via forward
        # kinematics at a representative pose: base at the origin, reaching up
        # to roughly z=1.2m) so the whole arm fits in frame rather than being
        # cropped -- override via --cam_distance/--cam_azimuth/--cam_elevation/
        # --cam_lookat if a different view suits your task better.
        self.cam = mujoco.MjvCamera()
        mujoco.mjv_defaultCamera(self.cam)
        self.cam.distance = distance
        self.cam.azimuth = azimuth
        self.cam.elevation = elevation
        self.cam.lookat[:] = lookat

    def _forward(self, q: np.ndarray) -> None:
        n = min(len(q), self.model.nq)
        self.data.qpos[:n] = q[:n]
        mujoco.mj_forward(self.model, self.data)

    def render(self, q: np.ndarray) -> np.ndarray:
        self._forward(q)
        self.renderer.update_scene(self.data, camera=self.cam)
        return cv2.cvtColor(self.renderer.render(), cv2.COLOR_RGB2BGR)

    def tcp_rotation_world(self, q: np.ndarray) -> "np.ndarray | None":
        """3x3 rotation matrix of the TCP in world frame, via forward
        kinematics from q -- None if the model has no tcp_site."""
        if self.tcp_site_id < 0:
            return None
        self._forward(q)
        return self.data.site_xmat[self.tcp_site_id].reshape(3, 3).copy()

    def blank(self, message: str) -> np.ndarray:
        frame = np.zeros((self.size, self.size, 3), dtype=np.uint8)
        cv2.putText(
            frame, message, (10, self.size // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (128, 128, 128), 1, cv2.LINE_AA
        )
        return frame


SIDEBAR_BG = (32, 28, 26)  # dark gray, BGR
SIDEBAR_DIVIDER_COLOR = (70, 65, 60)
SIDEBAR_FONT = cv2.FONT_HERSHEY_SIMPLEX
SIDEBAR_FONT_SCALE = 0.62
SIDEBAR_HEADER_FONT_SCALE = 0.85
SIDEBAR_LINE_HEIGHT = 30
SIDEBAR_TOP_MARGIN = 34
SIDEBAR_BOTTOM_MARGIN = 14
SIDEBAR_PADDING = 16
DIM_LABELS = ["x", "y", "z", "rx", "ry", "rz"]
DIVIDER = ("---", None)  # sentinel: draws a thin horizontal rule instead of text


def sidebar_size(lines: list) -> tuple:
    """Content-driven (width, height) for render_sidebar's lines -- so the
    caller can size/pad everything else to match instead of the sidebar
    silently clipping whenever more lines get added later."""
    width = 2 * SIDEBAR_PADDING
    for text, _ in lines:
        if text and text != DIVIDER[0]:
            (w, _), _ = cv2.getTextSize(text, SIDEBAR_FONT, SIDEBAR_FONT_SCALE, 1)
            width = max(width, w + 2 * SIDEBAR_PADDING)
    height = SIDEBAR_TOP_MARGIN + len(lines) * SIDEBAR_LINE_HEIGHT + SIDEBAR_BOTTOM_MARGIN
    return width, height


def render_sidebar(lines: list, width: int, height: int) -> np.ndarray:
    """lines: list of (text, color_bgr) tuples, top to bottom. Empty text is
    just vertical spacing; DIVIDER draws a thin rule instead. The first
    non-empty line is treated as a header and drawn larger/bold."""
    sidebar = np.full((height, width, 3), SIDEBAR_BG, dtype=np.uint8)
    y = SIDEBAR_TOP_MARGIN
    header_drawn = False
    for text, color in lines:
        if text == DIVIDER[0]:
            cv2.line(sidebar, (SIDEBAR_PADDING, y - SIDEBAR_LINE_HEIGHT // 2),
                      (width - SIDEBAR_PADDING, y - SIDEBAR_LINE_HEIGHT // 2), SIDEBAR_DIVIDER_COLOR, 1)
        elif text:
            is_header = not header_drawn
            header_drawn = True
            font_scale = SIDEBAR_HEADER_FONT_SCALE if is_header else SIDEBAR_FONT_SCALE
            thickness = 2 if is_header else 1
            cv2.putText(sidebar, text, (SIDEBAR_PADDING, y), SIDEBAR_FONT, font_scale, color, thickness, cv2.LINE_AA)
        y += SIDEBAR_LINE_HEIGHT
    return sidebar


def to_world_frame(action_body: np.ndarray, R: np.ndarray) -> np.ndarray:
    """Converts a body(TCP)-frame action to world frame -- same block-diagonal
    rotation RelativeFrame.transform_action applies (linear and angular parts
    rotated separately by the same R), just done here for display rather than
    for actually driving the robot."""
    action_body = np.asarray(action_body, dtype=np.float64)
    out = action_body.copy()
    out[:3] = R @ action_body[:3]
    if len(out) > 3:
        out[3:6] = R @ action_body[3:6]
    return out


def render(record, image_keys: list, scale: int, robot_renderer: "RobotRenderer | None", force_torque_series=None, policy_agent=None, action_scale=None, act_bundle=None, action_dim=None) -> np.ndarray:
    ep_i, step_i, ep_len, transition = record
    obs = transition["observations"]
    action = np.asarray(transition["actions"])
    if action_dim is not None:
        action = action[:action_dim]
    proprio = np.asarray(obs["state"]).flatten()
    q = transition.get("infos", {}).get("q")

    frames = []
    for k in image_keys:
        f = to_display_frame(obs[k])
        if scale != 1:
            f = cv2.resize(f, (f.shape[1] * scale, f.shape[0] * scale), interpolation=cv2.INTER_LINEAR)
        frames.append(f)
    camera_panel = np.concatenate(frames, axis=1) if len(frames) > 1 else frames[0]
    camera_panel = np.ascontiguousarray(camera_panel)

    dim_labels = DIM_LABELS[:len(action)]

    # Recorded/predicted actions are in the TCP's own body frame (see
    # RelativeFrame) -- recover the world-frame equivalent via forward
    # kinematics from q (RobotRenderer.tcp_rotation_world), same rotation
    # RelativeFrame.transform_action applies when actually driving the robot.
    R_world = robot_renderer.tcp_rotation_world(np.asarray(q)) if (robot_renderer is not None and q is not None) else None
    action_world = to_world_frame(action, R_world) if R_world is not None else None

    # action_scale (if given) converts the normalized [-1,1]-ish recorded/
    # predicted values into the real physical units (m/s linear, rad/s
    # angular) FlexivEnv.step() actually sends to the robot -- applying it
    # after the TCP<->world rotation above is equivalent to applying it
    # before, since ACTION_SCALE is a single scalar across each 3-axis block
    # (linear, angular) and scalar multiplication commutes with rotation.
    def scaled(v):
        if v is None:
            return None
        return v * action_scale if action_scale is not None else v

    unit_suffix = " [m/s | rad/s]" if action_scale is not None else " [normalized]"

    lines = [
        (f"ep {ep_i}  step {step_i}/{ep_len - 1}", (0, 255, 0)),
        (f"reward: {transition['rewards']}", (200, 200, 200)),
        ("" if q is not None else "[no q recorded -- world frame unavailable]", (0, 0, 255)),
        DIVIDER,
        (f"action (tcp frame){unit_suffix}:", (0, 255, 255)),
        (f"  {np.round(scaled(action), 3)}", (0, 255, 255)),
        (f"action (world frame){unit_suffix}:", (0, 255, 255)),
        (f"  {np.round(scaled(action_world), 3)}" if action_world is not None else "  n/a", (0, 255, 255)),
        DIVIDER,
        ("tcp_pose (xyz+euler):", (0, 255, 255)),
        (f"  {np.round(proprio[PROPRIO_POSE_SLICE], 3)}", (0, 255, 255)),
        ("tcp_force (N):", (0, 255, 255)),
        (f"  {np.round(proprio[PROPRIO_FORCE_SLICE], 3)}", (0, 255, 255)),
        ("tcp_torque (Nm):", (0, 255, 255)),
        (f"  {np.round(proprio[PROPRIO_TORQUE_SLICE], 3)}", (0, 255, 255)),
    ]

    policy_console_line = ""
    if policy_agent is not None:
        policy_mean, policy_std = predict_policy_action(policy_agent, obs)
        policy_mean_world = to_world_frame(policy_mean, R_world) if R_world is not None else None
        se = (policy_mean - action) ** 2  # kept in normalized units -- matches the actual training loss
        lines += [
            DIVIDER,
            (f"policy_mean (tcp frame){unit_suffix}:", (255, 0, 255)),
            (f"  {np.round(scaled(policy_mean), 3)}", (255, 0, 255)),
            (f"policy_mean (world frame){unit_suffix}:", (255, 0, 255)),
            (f"  {np.round(scaled(policy_mean_world), 3)}" if policy_mean_world is not None else "  n/a", (255, 0, 255)),
            (f"policy_std (tcp frame){unit_suffix}:", (255, 0, 255)),
            (f"  {np.round(scaled(policy_std), 3)}", (255, 0, 255)),
            DIVIDER,
            ("MSE (policy_mean vs action, normalized units):", (255, 255, 0)),
        ]
        lines += [(f"  {label:>3s}: {e:.4f}", (255, 255, 0)) for label, e in zip(dim_labels, se)]
        lines.append((f"  sum: {se.sum():.4f}  mean: {se.mean():.4f}", (255, 255, 0)))

        physical_gap_line = ""
        if action_scale is not None:
            # Exact physical gap for this one step -- unlike MSE, this is
            # meaningful on its own: the actual m/s / rad/s difference between
            # what the policy would have commanded and what really happened.
            # Linear and angular reported separately since their units differ
            # and can't be meaningfully combined into one norm.
            gap = scaled(policy_mean) - scaled(action)
            gap_lin = np.linalg.norm(gap[:3])
            gap_ang = np.linalg.norm(gap[3:6]) if len(gap) > 3 else None
            lines += [
                DIVIDER,
                ("Physical gap (policy vs actual):", (255, 255, 0)),
                (f"  |linear|: {gap_lin:.4f} m/s", (255, 255, 0)),
            ]
            if gap_ang is not None:
                lines.append((f"  |angular|: {gap_ang:.4f} rad/s", (255, 255, 0)))
            physical_gap_line = f" gap_lin={gap_lin:.4f}m/s gap_ang={gap_ang:.4f}rad/s"

        policy_console_line = (
            f"  policy_mean={np.round(scaled(policy_mean), 3)} policy_std={np.round(scaled(policy_std), 3)} "
            f"mse_sum={se.sum():.4f}{physical_gap_line}"
        )

    act_console_line = ""
    if act_bundle is not None:
        act_policy, act_preprocessor, act_postprocessor, act_image_key_map = act_bundle
        act_action = predict_act_action(act_policy, act_preprocessor, act_postprocessor, act_image_key_map, obs)
        act_action_world = to_world_frame(act_action, R_world) if R_world is not None else None
        act_se = (act_action - action) ** 2  # normalized units, same convention as the BC MSE above
        lines += [
            DIVIDER,
            (f"ACT action (tcp frame){unit_suffix}:", (0, 165, 255)),
            (f"  {np.round(scaled(act_action), 3)}", (0, 165, 255)),
            (f"ACT action (world frame){unit_suffix}:", (0, 165, 255)),
            (f"  {np.round(scaled(act_action_world), 3)}" if act_action_world is not None else "  n/a", (0, 165, 255)),
            DIVIDER,
            ("MSE (ACT vs action, normalized units):", (0, 165, 255)),
        ]
        lines += [(f"  {label:>3s}: {e:.4f}", (0, 165, 255)) for label, e in zip(dim_labels, act_se)]
        lines.append((f"  sum: {act_se.sum():.4f}  mean: {act_se.mean():.4f}", (0, 165, 255)))

        act_physical_gap_line = ""
        if action_scale is not None:
            act_gap = scaled(act_action) - scaled(action)
            act_gap_lin = np.linalg.norm(act_gap[:3])
            act_gap_ang = np.linalg.norm(act_gap[3:6]) if len(act_gap) > 3 else None
            lines += [
                DIVIDER,
                ("Physical gap (ACT vs actual):", (0, 165, 255)),
                (f"  |linear|: {act_gap_lin:.4f} m/s", (0, 165, 255)),
            ]
            if act_gap_ang is not None:
                lines.append((f"  |angular|: {act_gap_ang:.4f} rad/s", (0, 165, 255)))
            act_physical_gap_line = f" gap_lin={act_gap_lin:.4f}m/s gap_ang={act_gap_ang:.4f}rad/s"

        act_console_line = (
            f"  act_action={np.round(scaled(act_action), 3)} mse_sum={act_se.sum():.4f}{act_physical_gap_line}"
        )

    world_str = f" action_world={np.round(scaled(action_world), 3)}" if action_world is not None else ""
    print(
        f"  ep {ep_i} step {step_i:3d}/{ep_len - 1}: action={np.round(scaled(action), 3)}{world_str}  "
        f"tcp_pose={np.round(proprio[PROPRIO_POSE_SLICE], 3)}  "
        f"tcp_force={np.round(proprio[PROPRIO_FORCE_SLICE], 3)}  reward={transition['rewards']}"
        + ("" if q is not None else "  [no q recorded]")
        + policy_console_line
        + act_console_line
    )

    # The sidebar's content (esp. with a policy loaded) can need more vertical
    # room than the camera image provides -- size the whole row to whichever
    # is taller, padding the camera panel rather than letting the sidebar
    # silently clip.
    sidebar_w, sidebar_h = sidebar_size(lines)
    panel_h = max(camera_panel.shape[0], sidebar_h)
    if camera_panel.shape[0] != panel_h:
        pad = np.zeros((panel_h - camera_panel.shape[0], camera_panel.shape[1], 3), dtype=np.uint8)
        camera_panel = np.concatenate([camera_panel, pad], axis=0)

    if robot_renderer is not None:
        robot_panel = robot_renderer.render(np.asarray(q)) if q is not None else robot_renderer.blank("no joint data (q) in this demo")
        if robot_panel.shape[0] != panel_h:
            new_w = int(robot_panel.shape[1] * panel_h / robot_panel.shape[0])
            robot_panel = cv2.resize(robot_panel, (new_w, panel_h))
        top_row = np.concatenate([camera_panel, robot_panel], axis=1)
    else:
        top_row = camera_panel

    sidebar = render_sidebar(lines, sidebar_w, panel_h)
    top_row = np.concatenate([top_row, sidebar], axis=1)

    if force_torque_series is None:
        return top_row

    forces, torques = force_torque_series
    base_img, step_to_px, y_range_top, y_range_bottom = get_force_torque_base(
        ep_i, forces, torques, top_row.shape[1], 220
    )
    chart_panel = base_img.copy()
    x_px = step_to_px(step_i)
    cv2.line(chart_panel, (x_px, y_range_top[0]), (x_px, y_range_top[1]), (0, 0, 0), 1, cv2.LINE_AA)
    cv2.line(chart_panel, (x_px, y_range_bottom[0]), (x_px, y_range_bottom[1]), (0, 0, 0), 1, cv2.LINE_AA)
    return np.concatenate([top_row, chart_panel], axis=0)


def main(_):
    path = FLAGS.path or find_latest_demo_file()
    with open(path, "rb") as f:
        transitions = pkl.load(f)
    print(f"Loaded {len(transitions)} transitions from {path}")

    episodes = split_into_episodes(transitions)
    print(f"Found {len(episodes)} episode(s) (record_demos.py only ever saves successful ones)")

    if FLAGS.episode is not None:
        episodes = [episodes[FLAGS.episode]]

    records = build_frame_records(episodes)

    sample_obs = records[0][3]["observations"]
    image_keys = FLAGS.image_keys or [k for k in sample_obs.keys() if k != "state"]
    print(f"Image keys: {image_keys}  |  state shape: {np.asarray(sample_obs['state']).shape}")

    force_torque_series = None
    if FLAGS.show_force_torque:
        force_torque_series = [extract_force_torque(ep) for ep in episodes]

    policy_agent = None
    if FLAGS.policy_checkpoint:
        assert FLAGS.exp_name, "--exp_name is required when --policy_checkpoint is set."
        print(f"Loading policy from {FLAGS.policy_checkpoint} (exp_name={FLAGS.exp_name})...")
        policy_agent = load_policy_agent(FLAGS.exp_name, FLAGS.policy_checkpoint, FLAGS.policy_seed)
        print("Policy loaded.")

    action_scale = None
    action_dim = None
    if FLAGS.exp_name:
        action_scale, action_dim = load_action_scale(FLAGS.exp_name)
        print(f"ACTION_SCALE loaded: displaying actions in real units (scale={action_scale}).")

    act_bundle = None
    if FLAGS.act_checkpoint:
        print(f"Loading ACT policy from {FLAGS.act_checkpoint}...")
        act_bundle = load_act_policy(FLAGS.act_checkpoint)
        print("ACT policy loaded.")

    robot_renderer = None
    if FLAGS.show_robot:
        mjcf_path = FLAGS.mjcf_path or DEFAULT_MJCF_PATH
        try:
            robot_renderer = RobotRenderer(
                mjcf_path,
                distance=FLAGS.cam_distance,
                azimuth=FLAGS.cam_azimuth,
                elevation=FLAGS.cam_elevation,
                lookat=[float(v) for v in FLAGS.cam_lookat],
            )
            print(f"Robot renderer loaded from {mjcf_path}")
        except Exception as e:
            print(f"Could not load robot model from {mjcf_path} ({e}) -- continuing without robot view.")

    print("\nControls: SPACE play/pause | d/a step +1/-1 | D/A step +10/-10 | ESC quit\n")

    delay_ms = max(1, int(1000 / FLAGS.fps))
    idx = max(0, min(FLAGS.start_step, len(records) - 1))
    paused = FLAGS.start_step > 0
    last_idx = -1  # force initial render

    window = "demo playback"
    cv2.namedWindow(window, cv2.WINDOW_AUTOSIZE)

    while True:
        if idx != last_idx:
            ep_i = records[idx][0]
            ft_series = force_torque_series[ep_i] if force_torque_series is not None else None
            frame = render(records[idx], image_keys, FLAGS.scale, robot_renderer, ft_series, policy_agent, action_scale, act_bundle, action_dim)
            cv2.imshow(window, frame)
            last_idx = idx

        key = cv2.waitKey(0 if paused else delay_ms) & 0xFF

        if key == 27:  # ESC
            break
        elif key == ord(" "):
            paused = not paused
        elif key == ord("d"):
            paused = True
            idx = min(idx + 1, len(records) - 1)
        elif key == ord("a"):
            paused = True
            idx = max(idx - 1, 0)
        elif key == ord("D"):
            paused = True
            idx = min(idx + 10, len(records) - 1)
        elif key == ord("A"):
            paused = True
            idx = max(idx - 10, 0)

        if not paused:
            idx = min(idx + 1, len(records) - 1)
            if idx == len(records) - 1:
                paused = True  # stop at the end instead of silently looping/exiting

    cv2.destroyAllWindows()


if __name__ == "__main__":
    app.run(main)
