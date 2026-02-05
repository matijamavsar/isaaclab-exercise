# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
from __future__ import annotations

import os
import time
from datetime import datetime

import cv2
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from isaaclab.assets import Articulation, ArticulationCfg
from isaaclab.assets import ParticleClothObjectCfg, RigidObjectCfg
from isaaclab.controllers import DifferentialIKController, DifferentialIKControllerCfg
import isaaclab.sim as sim_utils
from isaaclab.envs import DirectRLEnv, DirectRLEnvCfg
from isaaclab.managers import SceneEntityCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import PhysxCfg, RenderCfg, SimulationCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.utils.math import subtract_frame_transforms, quat_slerp
from isaacsim.core.utils.prims import get_prim_at_path
from omni.physx.scripts import physicsUtils, particleUtils
from pxr import Gf, Sdf, UsdGeom, UsdPhysics
import omni.usd
import omni.kit.commands
from isaaclab_assets import FRANKA_PANDA_HIGH_PD_CFG

from torchvision.utils import save_image  # kept because original code used it in logs
from scipy.spatial import ConvexHull      # kept; convex hull code gated behind ifs

# Local utilities
from .dmp_integrator import BatchDMPIntegrator
from .min_jerk_traj import generate_minimum_jerk

# ROS 2
import rclpy
from rclpy.executors import SingleThreadedExecutor
from sensor_msgs.msg import JointState
from std_msgs.msg import Header
from trajectory_msgs.msg import JointTrajectory
from robotblockset.ros2.franka_ros2 import franka_ros2

from curobo.types.base import TensorDeviceType
from curobo.types.math import Pose
from curobo.types.robot import RobotConfig
from curobo.util_file import get_robot_configs_path, join_path, load_yaml
from curobo.wrap.reacher.ik_solver import IKSolver, IKSolverConfig

import curobo
curobo.util.logger.setup_logger('error', 'curobo')

# -----------------------------
# Utilities / Filters
# -----------------------------

def butter2_biquad_coeffs(fc: float, dt: float, device=None, dtype=torch.float32):
    fs = 1.0 / dt
    K = torch.tan(torch.tensor(torch.pi, device=device) * fc / fs)
    sqrt2 = torch.sqrt(torch.tensor(2.0, device=device))
    norm = 1.0 / (1.0 + sqrt2 * K + K * K)
    b0 = (K * K) * norm
    b1 = 2.0 * (K * K) * norm
    b2 = (K * K) * norm
    a1 = 2.0 * (K * K - 1.0) * norm
    a2 = (1.0 - sqrt2 * K + K * K) * norm
    to = lambda v: torch.as_tensor(v, device=device, dtype=dtype)
    return to(b0), to(b1), to(b2), to(a1), to(a2)


class BiquadLP2Batch:
    """2-pole Butterworth low-pass, batched. Input shape [B, DOF]."""
    def __init__(self, batch_size: int, dof: int, fc: float, dt: float,
                 device=None, dtype=torch.float32):
        self.B, self.D = batch_size, dof
        self.b0, self.b1, self.b2, self.a1, self.a2 = butter2_biquad_coeffs(fc, dt, device, dtype)
        self.device, self.dtype = device, dtype
        self.x1 = torch.zeros(self.B, self.D, device=device, dtype=dtype)
        self.x2 = torch.zeros(self.B, self.D, device=device, dtype=dtype)
        self.y1 = torch.zeros(self.B, self.D, device=device, dtype=dtype)
        self.y2 = torch.zeros(self.B, self.D, device=device, dtype=dtype)

    @torch.no_grad()
    def reset(self, x0: torch.Tensor):
        self.x1.copy_(x0); self.x2.copy_(x0); self.y1.copy_(x0); self.y2.copy_(x0)

    @torch.no_grad()
    def step(self, x: torch.Tensor) -> torch.Tensor:
        y = (self.b0 * x + self.b1 * self.x1 + self.b2 * self.x2
             - self.a1 * self.y1 - self.a2 * self.y2)
        self.x2.copy_(self.x1); self.x1.copy_(x)
        self.y2.copy_(self.y1); self.y1.copy_(y)
        return y


class Reward:
    """Container for one reward term."""
    def __init__(self, use: bool, scale: float):
        self.use = use
        self.scale = scale
        self.value: None


# -----------------------------
# Config
# -----------------------------

@configclass
class FrankaDMPClothPlaceEnvCfg(DirectRLEnvCfg):
    # env
    episode_length_s = 8
    max_episode_length = episode_length_s * 120
    num_updates_per_episode = 10
    decimation = max_episode_length // num_updates_per_episode
    action_space = 55
    observation_space = 100
    state_space = 0
    enable_camera_recording = True
    plot_trajectories = False
    disable_init_motion = False
    use_weighted_atan_rewards = False
    use_weighted_exp_rewards = False
    write_joint_state = True
    seed = 42

    torch.manual_seed(seed)

    # simulation
    sim: SimulationCfg = SimulationCfg(
        dt=1 / 120,
        render_interval=8,  # ~15 FPS
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
        physx=PhysxCfg(
            gpu_max_particle_contacts=2**22,
            gpu_max_soft_body_contacts=2**23,
        ),
        render=RenderCfg(rendering_mode="performance"),
    )

    # scene
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=1, env_spacing=2.0, replicate_physics=False)

    # cloth + handles
    cloth = ParticleClothObjectCfg(
        prim_path="/World/envs/env_.*/Cloth",
        init_state=ParticleClothObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 0.356), rot=(0.5, 0.5, 0.5, 0.5)),
        spawn=sim_utils.UsdFileCfg(
            usd_path="/workspace/isaaclab/source/isaaclab_tasks/isaaclab_tasks/sim/particle_cloth_new_noHandles.usd",
            scale=(1.0, 1.0, 1.0),
        ),
    )

    if enable_camera_recording:
        from isaaclab.sensors import CameraCfg
        camera = CameraCfg(
            prim_path="/World/Camera",
            offset=CameraCfg.OffsetCfg(
                pos=(-5.0, -5.0, 3.0),
                rot=(0.9020, -0.0828, 0.2000, 0.3736),
                convention="world",
            ),
            data_types=["rgb"],
            spawn=sim_utils.PinholeCameraCfg(
                focal_length=24.0, focus_distance=400.0, horizontal_aperture=20.955, clipping_range=(0.1, 40.0)
            ),
            width=1080,
            height=720,
        )
        ee_camera = CameraCfg(
            # mount under the Franka hand so it inherits the link pose
            prim_path="/World/envs/env_.*/Robot/panda_leftfinger/ee_camera",
            offset=CameraCfg.OffsetCfg(
                # small forward offset in the hand frame; tweak as needed
                pos=(0.0, 0.0, 0.05),
                # looking along the hand's -Z/+X depends on your asset; start with identity
                rot=(0.0, 0.0, 1.0, 0.0),
                convention="parent",
            ),
            data_types=["depth"],
            spawn=sim_utils.PinholeCameraCfg(
                focal_length=16.0, focus_distance=100.0,
                horizontal_aperture=12.0, clipping_range=(0.05, 20.0)
            ),
            width=128,    # smaller to keep obs light-weight
            height=128,
        )

    cloth_plain = ParticleClothObjectCfg(
        prim_path="/World/envs/env_.*/Cloth",
        init_state=ParticleClothObjectCfg.InitialStateCfg(pos=(0, 0, 0), rot=(1, 0, 0, 0)),
        spawn=None,
    )

    # Robot (single Franka)
    robot = FRANKA_PANDA_HIGH_PD_CFG.replace(
        prim_path="/World/envs/env_.*/Robot",
        init_state=ArticulationCfg.InitialStateCfg(
            joint_pos={
                "panda_joint1": -0.0311,
                "panda_joint2": -0.5669,
                "panda_joint3": 0.0147,
                "panda_joint4": -1.5801,
                "panda_joint5": 0.0091,
                "panda_joint6": 1.0138,
                "panda_joint7": 0.7715,
                "panda_finger_joint.*": 0.005,
            },
            pos=(0.0, -0.3, 0.0),
            rot=(0.7071, 0, 0, 0.7071),
        ),
    )

    # ground plane
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
        ),
        debug_vis=False,
    )

    action_scale = 15
    dof_velocity_scale = 0.1
    filter_kernel_size = 7
    use_dynamic_rewards = False

    rewards = {
        "spread_reward": Reward(True, 1.0),
        "height_reward": Reward(True, 1.0),
        "corner_x_reward": Reward(False, 10.0),
        "direction_reward": Reward(False, 15.0),
        "endspeed_reward": Reward(False, 1.0),
        "action_penalty": Reward(False, 1e-2),
    }


# -----------------------------
# Env
# -----------------------------

def make_circle_path_torch(center=(0, 0, 3), radius=10, num_points=720, device="cpu"):
    cx, cy, cz = center
    angles = torch.linspace(torch.pi, torch.pi + 2 * torch.pi, num_points, device=device, requires_grad=False)
    xs = cx + radius * torch.cos(angles)
    ys = cy + radius * torch.sin(angles)
    zs = torch.full_like(xs, cz)
    return torch.stack([xs, ys, zs], dim=1)


class FrankaDMPClothPlaceEnv(DirectRLEnv):
    """Single-robot refactor of the original dual-arm environment."""

    cfg: FrankaDMPClothPlaceEnvCfg

    # -------------------------
    # Init
    # -------------------------
    def __init__(self, cfg: FrankaDMPClothPlaceEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
        self.seed(self.cfg.seed)

        # Video recording
        now = datetime.now().strftime("%Y%m%d_%H%M%S")
        if self.cfg.enable_camera_recording:
            self.video_folder = os.path.join("./logs/videos", now)
            os.makedirs(self.video_folder, exist_ok=True)
        self.video_writer = None
        self.render_count = 0

        self.action_steps = torch.zeros(self.num_envs, device=self.device, dtype=torch.int32)
        self.iteration_step = torch.tensor(0, device=self.device)
        self.dt = self.cfg.sim.dt * self.cfg.decimation

        # Camera motion helpers
        self.init_camera_pose = torch.tensor(
            [-10.0, -10.0, 0.3, 0.9238795, 0.0, 0.0, 0.3826834], device=self.device
        )
        self.new_camera_pose = self.init_camera_pose.clone()
        self.camera_path = torch.linspace(self.init_camera_pose[0], self.init_camera_pose[0] + 10, 720).to(self.device)
        self.circle_camera_path = make_circle_path_torch().to(self.device)
        self.action_penalties = torch.zeros((self.cfg.max_episode_length, self.num_envs), device=self.device)

        # Filters / Buffers
        self.lp = None  # low-pass filter initialized after robot spawn

        tensor_args = TensorDeviceType()

        config_file = load_yaml(join_path(get_robot_configs_path(), "franka.yml"))
        urdf_file = config_file["robot_cfg"]["kinematics"][
            "urdf_path"
        ]  # Send global path starting with "/"
        base_link = config_file["robot_cfg"]["kinematics"]["base_link"]
        ee_link = config_file["robot_cfg"]["kinematics"]["ee_link"]
        robot_cfg = RobotConfig.from_basic(urdf_file, base_link, ee_link, tensor_args)

        ik_config = IKSolverConfig.load_from_robot_config(
            robot_cfg,
            None,
            rotation_threshold=0.05,
            position_threshold=0.005,
            num_seeds=20,
            self_collision_check=False,
            self_collision_opt=False,
            tensor_args=tensor_args,
            use_cuda_graph=True,
        )
        self.ik_solver = IKSolver(ik_config)

        # DMPs
        self.dmp_initialized = False
        self.dmp_integrator = BatchDMPIntegrator(N_basis=25, dof=2, device=self.device)

        # Reward helpers
        self._rewards = self.cfg.rewards
        self.z_abs_pos = torch.zeros((self.cfg.max_episode_length, self.num_envs), device=self.device)
        self.action_penalty_buf = torch.zeros((self.cfg.max_episode_length, self.num_envs), device=self.device)

        # Min-jerk probe (kept for parity; currently unused)
        abs_points = [[0.0, 0.0, 0.81], [-0.3, 0.0, 0.81], [0.2, 0.0, 0.81], [0.0, 0.0, 0.81]]
        durations_relative = torch.tensor([1, 2, 1], device=self.device)
        durations = self.cfg.episode_length_s * durations_relative / durations_relative.sum()
        self._probing_traj = torch.tensor(
            generate_minimum_jerk(waypoints=abs_points, durations=durations.cpu().tolist(), num_points=self.cfg.decimation),
            device=self.device,
        )
        self._probing_traj = torch.cat(
            (self._probing_traj[0].unsqueeze(0), self._probing_traj, self._probing_traj[-1].unsqueeze(0)), dim=0
        )

        # ROS2 (single robot)
        rclpy.init(args=None)
        self._ros_node = rclpy.create_node("isaaclab_joint_io")
        self._ros_exec = SingleThreadedExecutor()
        self._ros_exec.add_node(self._ros_node)

        self._ros_joint_names = []
        self._ros_desired_q = None
        self._ros_desired_t = [None for _ in range(self.num_envs)]
        self.use_ros_cmd = True
        self._ros_state_pub = {}
        self._ros_cmd_sub = {}
        self.robot_rbs = None  # robotblockset kinematics helper

        # Joint limits/speeds
        self.dof_lower_limits = self._robot.data.soft_joint_pos_limits[0, :, 0].to(self.device)
        self.dof_upper_limits = self._robot.data.soft_joint_pos_limits[0, :, 1].to(self.device)
        self.dof_speed_scales = torch.ones_like(self.dof_lower_limits)
        for fname in ["panda_finger_joint1", "panda_finger_joint2"]:
            j_idx = self._robot.find_joints(fname)[0]
            self.dof_speed_scales[j_idx] = 0.1

        # Filters/buffers
        self.lp = BiquadLP2Batch(self.scene.cfg.num_envs, self._robot.num_joints, fc=1.0, dt=self.physics_dt, device=self.device)
        self.robot_dof_targets = torch.zeros((self.num_envs, self._robot.num_joints), device=self.device)
        self.gripper_actions = torch.zeros((self.num_envs, 2), device=self.device)
        self.ee_jacobi_idx = 7

        # ROS pubs/subs for single robot
        names = getattr(self._robot.data, "joint_names", None)
        self._ros_joint_names = [str(n) for n in names] if names and len(names) else [
            "panda_joint1","panda_joint2","panda_joint3","panda_joint4","panda_joint5","panda_joint6","panda_joint7",
            "panda_finger_joint1","panda_finger_joint2"
        ]
        self._ros_desired_q = torch.zeros(self.num_envs, self._robot.num_joints, device=self.device)

        ns = "robot1"
        for env_i in range(self.num_envs):
            base_ns = f"/env_{env_i}/{ns}"
            state_topic = f"{base_ns}/joint_states"
            cmd_topic = f"{base_ns}/joint_trajectory_command"
            self._ros_state_pub[env_i] = self._ros_node.create_publisher(JointState, state_topic, 10)

            def _make_cb(ei):
                def _cb(msg: JointTrajectory):
                    if not msg.points:
                        return
                    pt = msg.points[0]
                    q_target = self._robot.data.joint_pos[ei].clone()
                    if msg.joint_names and len(pt.positions) >= 1:
                        name_to_idx = {n: i for i, n in enumerate(msg.joint_names)}
                        for j, n in enumerate(self._ros_joint_names[: self._robot.num_joints]):
                            src = name_to_idx.get(n)
                            if src is not None and src < len(pt.positions):
                                q_target[j] = float(pt.positions[src])
                    else:
                        n_copy = min(self._robot.num_joints, len(pt.positions))
                        if n_copy == 0:
                            return
                        q_target[:n_copy] = torch.tensor(pt.positions[:n_copy], device=self.device)
                    self._ros_desired_q[ei] = q_target
                    self._ros_desired_t[ei] = self._ros_node.get_clock().now()
                    if self.use_ros_cmd:
                        env_ids = torch.tensor([ei], device=self.device)
                        self._robot.write_joint_position_to_sim(q_target, env_ids=env_ids)
                        self._robot.set_joint_position_target(q_target, env_ids=env_ids)
                return _cb

            self._ros_cmd_sub[env_i] = self._ros_node.create_subscription(JointTrajectory, cmd_topic, _make_cb(env_i), 10)

        # robotblockset IK helper
        self.robot_rbs = franka_ros2(ns="env_0/robot1")

    def _ros_spin_once(self):
        self._ros_exec.spin_once(timeout_sec=0.0)

    def _js_msg(self, names, positions, velocities=None):
        msg = JointState()
        msg.header = Header()
        msg.header.stamp = self._ros_node.get_clock().now().to_msg()
        msg.name = names
        msg.position = [float(x) for x in positions]
        if velocities is not None:
            msg.velocity = [float(x) for x in velocities]
        return msg

    def _setup_camera_writer(self, step):
        block_index = step // 20000
        filename = f"{self.video_folder}/record_block_{block_index}.mp4"
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        img = self._camera.data.output["rgb"][0]
        H, W, _ = img.shape
        self.video_writer = cv2.VideoWriter(filename, fourcc, 15, (W, H))

    def _close_camera_writer(self):
        if self.video_writer is not None:
            self.video_writer.release()
            self.video_writer = None

    def _capture_and_write_frame(self):
        self._camera.update(self.physics_dt)
        img_rgb = self._camera.data.output["rgb"][0]
        img_bgr = cv2.cvtColor(img_rgb.cpu().numpy(), cv2.COLOR_RGB2BGR)
        self.video_writer.write(img_bgr)

    # -------------------------
    # Scene / Setup
    # -------------------------

    def _setup_scene(self):
        # Robot
        self._robot = Articulation(self.cfg.robot)
        self.scene.articulations["robot"] = self._robot

        # Terrain
        self.cfg.terrain.num_envs = self.scene.cfg.num_envs
        self.cfg.terrain.env_spacing = self.scene.cfg.env_spacing
        self._terrain = self.cfg.terrain.class_type(self.cfg.terrain)

        # Cloth + handles
        self._cloth = self.cfg.cloth.class_type(self.cfg.cloth)
        self._cloth_plain = self.cfg.cloth_plain.class_type(self.cfg.cloth_plain)

        # Clone envs
        self.scene.clone_environments(copy_from_source=True)

        # Randomize cloth length (scale Y) per env
        self.cloth_lengths = torch.zeros(self.scene.cfg.num_envs, device=self.device)
        for env_idx in range(self.scene.cfg.num_envs):
            cloth_prim_path = f"/World/envs/env_{env_idx}/Cloth"
            cloth_prim = get_prim_at_path(cloth_prim_path)
            random_scale = torch.tensor([1.0, torch.rand(1).item() * (1.0 - 0.6) + 0.6, 1.0])
            cloth_prim.GetAttribute("xformOp:scale").Set(Gf.Vec3f(*random_scale.tolist()))
            cloth_prim.GetAttribute("xformOp:translate").Set(
                Gf.Vec3f(0.0, 0.0, 0.356 + ((1 - random_scale[1]) * 0.356).item())
            )
            self.cloth_lengths[env_idx] = random_scale[1].item() * 0.7

        # Camera
        if self.cfg.enable_camera_recording:
            self._camera = self.cfg.camera.class_type(self.cfg.camera)
            self._ee_camera = self.cfg.ee_camera.class_type(self.cfg.ee_camera)

        # Light
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

        # Hide world environment (optional)
        stage = omni.usd.get_context().get_stage()
        env_prim = stage.GetPrimAtPath("/World/ground/Environment")
        if env_prim:
            UsdGeom.Imageable(env_prim).GetVisibilityAttr().Set(UsdGeom.Tokens.invisible)

        # Control helpers
        self.robot_entity_cfg = SceneEntityCfg("robot", joint_names=["panda_joint.*"], body_names=["panda_hand"])
        self._ik_controller = DifferentialIKController(
            DifferentialIKControllerCfg(command_type="pose", use_relative_mode=False, ik_method="dls"),
            num_envs=self.num_envs,
            device=self.device,
        )

    def _get_ee_pose_w(self):
        """End-effector pose (world) of single robot hand link."""
        hand_id = self.robot_entity_cfg.body_ids[0]
        ee_pose_w = self._robot.data.body_state_w[:, hand_id, 0:7]
        return ee_pose_w

    def _get_absolute_pose(self):
        """For single robot, 'absolute' == EE pose."""
        return self._get_ee_pose_w()

    def _get_corners(self):
        cloth_positions = self._cloth.root_physx_view.get_positions().reshape(self.num_envs, -1, 3)
        side_len = int(torch.sqrt(torch.tensor(cloth_positions.shape[1])).to(self.device))
        corners = cloth_positions[:, [-side_len, -1, 0, side_len - 1], :]
        return corners

    # -------------------------
    # RL hooks
    # -------------------------

    def _pre_physics_step(self, actions: torch.Tensor):
        # Sanitize/reshape DMP parameters (still 2 DOF for x/z; tau in last entry)
        dmp_parameters = actions.clone()
        dmp_parameters[:, 2] = dmp_parameters[:, 2].clamp(-0.7, 0.7)  # unused in single-arm, kept for compat
        dmp_parameters[:, 3] = dmp_parameters[:, 3].clamp(0.0, 1.0)  # amplitude, etc.
        # dmp_parameters = dmp_parameters.clamp(-1.5, 1.5)

        current_absolute_pose = self._get_absolute_pose()
        current_absolute_pose[:, 0:3] -= self.scene.env_origins
        dmp_parameters[:, 0] = current_absolute_pose[:, 0]  # x start
        dmp_parameters[:, 1] = current_absolute_pose[:, 2]  # z start

        self.actions = dmp_parameters
        dmp_tau = torch.zeros(self.num_envs, device=self.device) + self.cfg.episode_length_s
        self.actions[:, -1] = dmp_tau

        hard_reset = (self.episode_length_buf == 0).all()
        reset_dmp_indices = torch.arange(0, dmp_parameters.shape[0])

        if len(reset_dmp_indices) > 0:
            self.dmp_integrator.reset_indices(
                reset_dmp_indices,
                dmp_parameters,
                dmp_tau,
                dt=self.physics_dt,
                variant=2,
                soft_reset=not (hard_reset),
            )
            self.dmp_initialized = True

        self.action_steps *= 0

    def _apply_action(self):
        # Init LP on first step of episode
        if self.action_steps[0] == 0:
            q0 = self._robot.data.joint_pos[:, : self._robot.num_joints]
            self.lp.reset(q0)

        # Step DMP
        t, y, dy, ddy = self.dmp_integrator.step()
        self.dmp_integrator.x[t >= self.actions[:, -1]] = 0.1353

        # Build target absolute pose (world): keep y fixed to mid-line (0 in env frame)
        current_abs = self._get_absolute_pose()
        target_abs = torch.zeros_like(current_abs)
        target_abs[:, 0] = y[:, 0]  # x from DMP
        target_abs[:, 2] = y[:, 1]  # z from DMP
        target_abs[:, 0:3] += self.scene.env_origins
        target_abs[:, 3:] = torch.tensor([1, 0, 0, 0], device=self.device)  # gripper orientation

        # IK in robot base frame
        root_pose_w = self._robot.data.root_state_w[:, 0:7]
        target_pos_b, target_quat_b = subtract_frame_transforms(
            root_pose_w[:, 0:3], root_pose_w[:, 3:7], target_abs[:, 0:3], target_abs[:, 3:7]
        )
        target_quat_b *= 0.0
        target_quat_b[:, 1] = 1.0

        # Solve IK via robotblockset
        # try:
        #     current_q = self._robot.data.joint_pos[:, self.robot_entity_cfg.joint_ids]
        #     target_q = self.robot_rbs.IKin(
        #         torch.cat([target_pos_b, target_quat_b], dim=-1).squeeze().cpu(),
        #         current_q.squeeze().cpu(), pos_err=0.01, ori_err=0.05
        #     )[0]
        #     if not torch.isnan(torch.tensor(target_q, device=self.device)).any():
        #         self.robot_dof_targets[:, 0:7] = torch.tensor(target_q, device=self.device)
        # except Exception:
        #     pass

        # Solve IK via Isaac Lab methods
        # self.inverse_kinematics(y)

        # Solve IK via Curobo
        self.inverse_kinematics_curobo(y)

        # Optional smoothing (commented out to keep response snappy)
        q_smooth = self.lp.step(self.robot_dof_targets)
        self.robot_dof_targets = q_smooth

        self._robot.write_joint_position_to_sim(self.robot_dof_targets)
        self._robot.set_joint_position_target(self.robot_dof_targets)

        # Camera recording window
        if self.cfg.enable_camera_recording:
            in_window = (self.iteration_step % 20000) < 720
            if in_window and self.video_writer is None:
                self.render_count = 0
                self._setup_camera_writer(self.iteration_step)
            if in_window and self.video_writer is not None:
                if self.render_count % self.cfg.sim.render_interval == 0:
                    self._capture_and_write_frame()
                self.render_count += 1
            if (not in_window) and self.video_writer is not None:
                self.render_count = 0
                self._close_camera_writer()

        # Log z for reward invalidation
        self.z_abs_pos[self.action_steps] = current_abs[:, 2].clone()

        self.iteration_step += 1
        self.action_steps += 1

    def inverse_kinematics_curobo(self, y):
        current_absolute_pose = self._get_absolute_pose()
        target_absolute_pose = current_absolute_pose.clone() * 0.0
        target_absolute_pose[:,0] = y[:,0]
        target_absolute_pose[:,2] = y[:,1]
        target_absolute_pose[:,0:3] += self.scene.env_origins
        
        target_absolute_pose[:, 3:] = torch.tensor([0, 1, 0, 0], device=self.device)
       
        root_pose_w = self._robot.data.root_state_w[:, 0:7]
        rarm_pos_b, _ = subtract_frame_transforms(
            root_pose_w[:, 0:3], root_pose_w[:, 3:7], 
            target_absolute_pose[:, 0:3], target_absolute_pose[:, 3:7]
        )

        rarm_goal_quat_b = torch.zeros((self.num_envs, 4), device=self.device)
        rarm_goal_quat_b[:,1] = 1.0

        with torch.inference_mode(False):
            current_joints = self._robot.data.joint_pos[
                :, self.robot_entity_cfg.joint_ids].clone()
            goal = Pose(rarm_pos_b.clone(), rarm_goal_quat_b.clone())
            result = self.ik_solver.solve_batch(goal,
                retract_config=current_joints,
                seed_config=current_joints.unsqueeze(0).repeat(self.num_envs,1,1))
            q_solution = result.solution.squeeze(1)

        self.robot_dof_targets[:,-2:] = self.gripper_actions
        new_targets = q_solution
        new_targets = F.pad(input=new_targets, pad=(0,2), mode='constant', value=0)
        new_targets = torch.clamp(new_targets, self.dof_lower_limits, 
                                    self.dof_upper_limits)
        self.robot_dof_targets = new_targets

    def inverse_kinematics(self, y):
        current_absolute_pose = self._get_absolute_pose()
        target_absolute_pose = current_absolute_pose.clone() * 0.0
        target_absolute_pose[:,0] = y[:,0]
        target_absolute_pose[:,2] = y[:,1]
        target_absolute_pose[:,0:3] += self.scene.env_origins
        
        target_absolute_pose[:, 3:] = torch.tensor([0, 1, 0, 0], device=self.device)

        root_pose_w = self._robot.data.root_state_w[:, 0:7]
        rarm_pos_b, _ = subtract_frame_transforms(
            root_pose_w[:, 0:3], root_pose_w[:, 3:7], 
            target_absolute_pose[:, 0:3], target_absolute_pose[:, 3:7]
        )

        rarm_goal_quat_b = torch.zeros((self.num_envs, 4), device=self.device)
        rarm_goal_quat_b[:,1] = 1.0
        rarm_pose_b = torch.cat((rarm_pos_b, rarm_goal_quat_b), dim=-1)

        self._ik_controller.set_command(rarm_pose_b)

        self.robot_dof_targets[:,-2:] = self.gripper_actions
        # ✅ Compute new target positions for each robot separately
        robot = self._robot
        robot_entity_cfg = self.robot_entity_cfg

        ee_pose_w = robot.data.body_state_w[
            :, robot_entity_cfg.body_ids[0], 0:7]

        # Obtain robot's Jacobian matrix
        jacobian_w = robot.root_physx_view.get_jacobians()[
            :, self.ee_jacobi_idx, :, 
            robot_entity_cfg.joint_ids]
        base_rot = robot.data.root_quat_w
        jacobian = self._ik_controller.get_jacobian_in_root_frame(jacobian_w, base_rot)

        # Get root pose and joint positions
        root_pose_w = robot.data.root_state_w[:, 0:7]
        current_joint_pos = robot.data.joint_pos[
            :, robot_entity_cfg.joint_ids]

        ee_pos_b, ee_quat_b = subtract_frame_transforms(
            root_pose_w[:, 0:3], root_pose_w[:, 3:7], 
            ee_pose_w[:, 0:3], ee_pose_w[:, 3:7]
        )

        # Compute new joint positions using IK
        self.robot_dof_targets[:, 0:7] = self._ik_controller.compute(
            ee_pos_b, ee_quat_b, jacobian, current_joint_pos)

        padded_joint_pos = F.pad(input=current_joint_pos, pad=(0,2), mode='constant', value=0)
        self.joint_change = self.robot_dof_targets - padded_joint_pos

        # self.scaled_joint_change = self.robot_dof_speed_scales[robot_key] * self.dt * (
        #     self.robot_dof_targets[robot_key] - padded_joint_pos) * self.cfg.action_scale

        self.action_penalties[self.action_steps] = self.joint_change.abs().mean(dim=-1)

        if self.cfg.write_joint_state:
            ### Limit change of joints if writing directly
            self.clamped_joint_change = self.joint_change # .clamp(-1e-1, 1e-1)
            new_targets = padded_joint_pos + self.clamped_joint_change
        else:
            ### Or just send it to target positions
            new_targets = self.robot_dof_targets

        # ✅ Clamp values within each robot's DOF limits
        new_targets = torch.clamp(new_targets, self.dof_lower_limits, self.dof_upper_limits)

        self.robot_dof_targets = new_targets
        return current_absolute_pose

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        terminated = 0
        truncated = self.episode_length_buf >= self.max_episode_length
        if truncated.any():
            truncated[:] = True
        return terminated, truncated

    def compute_rewards(self):
        corners = self._get_corners()
        cloth_positions = self._cloth_plain.root_physx_view.get_positions().reshape(self.num_envs, -1, 3)
        mean_cloth_height = cloth_positions[:, :, 2].mean(dim=-1)

        rewards = {}
        # Spread reward (avg pairwise distance between corners)
        pairwise_sum = torch.zeros(self.num_envs, device=self.device)
        count = 0.0
        for i in range(4):
            for j in range(i + 1, 4):
                pairwise_sum += torch.norm(corners[:, i] - corners[:, j], dim=-1)
                count += 1.0

        # Corner X positioning/direction (relative to env origin x)
        free_x = corners[:, [2, 3], 0] - self.scene.env_origins[:, 0].unsqueeze(1)
        grasped_x = corners[:, [0, 1], 0] - self.scene.env_origins[:, 0].unsqueeze(1)

        # Endspeed (single robot)
        endspeed = self._robot.data.joint_vel.abs().sum(-1)

        rewards["height_reward"] = (1.0 / (0.1 + mean_cloth_height)) * self._rewards["height_reward"].scale
        rewards["height_reward"][mean_cloth_height > 0.4] *= 0.0
        rewards["spread_reward"] = (pairwise_sum / count) * self._rewards["spread_reward"].scale
        rewards["corner_x_reward"] = (free_x.mean(-1)) * self._rewards["corner_x_reward"].scale
        rewards["direction_reward"] = (free_x.mean(-1) - grasped_x.mean(-1)) * self._rewards["direction_reward"].scale
        rewards["endspeed_reward"] = (1 / (0.1 + endspeed)) * self._rewards["endspeed_reward"].scale

        # Simple action penalty buffer (kept structure)
        rewards["action_penalty"] = (self.action_penalty_buf.mean(0)) * self._rewards["action_penalty"].scale
        return rewards

    def _get_rewards(self) -> torch.Tensor:
        reward_dict = self.compute_rewards()
        for name, value in reward_dict.items():
            self._rewards[name].value = value

        total_reward = torch.zeros(self.num_envs, device=self.device)
        for k, term in self._rewards.items():
            if not term.use:
                continue
            if "penalty" in k:
                total_reward -= term.value
            else:
                total_reward += term.value

        # Invalidate when tool too low
        invalid_env = (self.z_abs_pos < 0.15).any(0)
        total_reward[invalid_env] -= 20

        # Only give reward at the last update step
        if (self.episode_length_buf != self.cfg.num_updates_per_episode).any():
            total_reward[self.episode_length_buf != self.cfg.num_updates_per_episode] *= 0.0

        self.extras["log"] = {n: v.value.mean() for n, v in self._rewards.items()}
        return total_reward

    # -------------------------
    # Reset / Joints / Scale
    # -------------------------

    def _reset_idx(self, env_ids: torch.Tensor | None):
        super()._reset_idx(env_ids)

        self.action_steps[env_ids] = 0

        # Resolve entity cfg once
        try:
            self.robot_entity_cfg.resolve(self.scene)
        except Exception:
            pass

        # Cache default cloth/handles state
        if not hasattr(self, "default_states") or self.default_states is None:
            self.default_states = {}
            cloth_positions = self._cloth_plain.root_physx_view.get_positions()
            self.default_states["_cloth_plain"] = cloth_positions.clone()

        # Reset robot joints
        joint_pos = self._robot.data.default_joint_pos[env_ids].clone()
        joint_vel = self._robot.data.default_joint_vel[env_ids].clone()
        self._robot.set_joint_position_target(joint_pos, env_ids=env_ids)
        self._robot.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)

        # Reset cloth/handles
        zeros_vel = torch.zeros(
            (len(env_ids), self._cloth_plain.root_physx_view.max_particles_per_cloth * 3), device=self.device
        )
        self._cloth_plain.root_physx_view.set_velocities(zeros_vel, indices=env_ids)
        self._cloth_plain.root_physx_view.set_positions(self.default_states["_cloth_plain"][env_ids], indices=env_ids)
        self._cloth_plain.update(self.physics_dt)

        # Create fixed joint (robot finger ↔ RightCube) once
        # if not getattr(self, "joints_created", False):
        #     self.create_joint_single(env_ids)
        #     self.joints_created = True

    def create_joint_single(self, env_ids):
        """Attach the robot's left finger to RightCube with a fixed joint (single robot)."""
        for idx in env_ids:
            env_i = int(idx)
            cube_path = Sdf.Path(f"/World/envs/env_{env_i}/Cloth/RightCube")
            finger_path = Sdf.Path(f"/World/envs/env_{env_i}/Robot/panda_leftfinger")
            joint_path = finger_path.AppendElementString("fixedJoint")
            fixedJoint = UsdPhysics.FixedJoint.Define(self.scene.stage, joint_path)
            fixedJoint.CreateBody0Rel().SetTargets([finger_path])
            fixedJoint.CreateBody1Rel().SetTargets([cube_path])
            fixedJoint.CreateLocalPos0Attr().Set(Gf.Vec3f(0, 0, 0))
            fixedJoint.CreateLocalRot0Attr().Set(Gf.Quatf(1, 0, 0, 0))
            fixedJoint.CreateLocalPos1Attr().Set(Gf.Vec3f(0, 0, 0))
            fixedJoint.CreateLocalRot1Attr().Set(Gf.Quatf(1, 0, 0, 0))

    # -------------------------
    # Observations
    # -------------------------

    def _get_observations(self) -> torch.Tensor:
        # Normalize DOF positions
        dof_pos_scaled = (
            2.0
            * (self._robot.data.joint_pos - self.dof_lower_limits)
            / (self.dof_upper_limits - self.dof_lower_limits)
            - 1.0
        )
        obs_robot = torch.cat((dof_pos_scaled, self._robot.data.joint_vel * self.cfg.dof_velocity_scale), dim=-1)

        # Cloth corners (flattened)
        current_corner_obs = self._get_corners().reshape(self.num_envs, -1)

        # Base observation (as before)
        observations = torch.cat((obs_robot, current_corner_obs), dim=-1)

        # 🔸 Append depth image from EE camera
        if getattr(self, "_ee_camera", None) is not None:
            # advance and fetch depth
            self._ee_camera.update(self.physics_dt)
            # Try both keys depending on your build
            if "distance_to_image_plane" in self._ee_camera.data.output:
                depth = self._ee_camera.data.output["distance_to_image_plane"]  # [B,H,W,1] or [B,H,W]
            else:
                depth = self._ee_camera.data.output["depth"]                     # [B,H,W,1] or [B,H,W]

            # ensure shape [B,H,W]
            if depth.dim() == 4 and depth.size(-1) == 1:
                depth = depth.squeeze(-1)

            # sanitize numeric issues
            # (clip to sensor range; convert NaN/Inf to max)
            # min_d = depth.min()
            # max_d = depth.max()
            min_d, max_d = 0.05, 1.0
            depth = torch.clamp(depth, min=min_d, max=max_d)

            # optional normalization to [0,1]
            depth_norm = (depth - min_d) / (max_d - min_d)

            # flatten per env and concatenate
            depth_flat = depth_norm.reshape(self.num_envs, -1)
            observations = torch.cat((observations, depth_flat), dim=-1)

        return {"policy": observations}