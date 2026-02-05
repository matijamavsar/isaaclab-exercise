# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
from __future__ import annotations

import torch
import os

from isaacsim.core.utils.stage import get_current_stage
from isaacsim.core.utils.torch.transformations import tf_combine, tf_inverse, tf_vector
from pxr import UsdGeom, UsdShade, Sdf, Gf, PhysxSchema
import cv2

import time
from datetime import datetime

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, ArticulationCfg
from isaaclab.assets import DeformableObjectCfg, RigidObjectCfg
from isaaclab.assets import ParticleClothObjectCfg
from isaaclab.envs import DirectRLEnv, DirectRLEnvCfg
from isaaclab.managers import SceneEntityCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg, PhysxCfg, RenderCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab.utils.math import sample_uniform, subtract_frame_transforms, matrix_from_quat
from isaaclab.utils.math import quat_slerp, quat_mul, quat_inv
from isaaclab.sensors import ContactSensorCfg, CameraCfg
from pxr import UsdPhysics, UsdGeom, Gf, Sdf
import omni.usd
import omni.kit.commands
from omni.physx.scripts import physicsUtils, particleUtils
from isaaclab.controllers import DifferentialIKController, DifferentialIKControllerCfg
from isaacsim.core.utils.prims import get_prim_at_path
import torch.nn.functional as F
from isaaclab_assets import FRANKA_PANDA_HIGH_PD_CFG
from torchvision.utils import save_image
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull

from .dmp_integrator import BatchDMPIntegrator
from .min_jerk_traj import generate_minimum_jerk

""" Run this training using
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py --task DMP-Based-Coat-Fold-Init-Motion --num_envs 64 --max_iterations 16000 --headless --enable_cameras
"""

def butter2_biquad_coeffs(fc: float, dt: float, device=None, dtype=torch.float32):
    fs = 1.0 / dt
    K = torch.tan(torch.tensor(torch.pi, device=device) * fc / fs)
    sqrt2 = torch.sqrt(torch.tensor(2.0, device=device))
    norm = 1.0 / (1.0 + sqrt2*K + K*K)
    b0 =  (K*K) * norm
    b1 =  2.0*(K*K) * norm
    b2 =  (K*K) * norm
    a1 =  2.0*(K*K - 1.0) * norm
    a2 =  (1.0 - sqrt2*K + K*K) * norm
    # scalars as 0-d tensors (broadcastable)
    to = lambda v: torch.as_tensor(v, device=device, dtype=dtype)
    return to(b0), to(b1), to(b2), to(a1), to(a2)


class BiquadLP2Batch:
    """
    2-pole Butterworth low-pass, batched.
    Shapes:
      - input x: [B, DOF]
      - internal state: [B, DOF]
    """
    def __init__(self, batch_size: int, dof: int, fc: float, dt: float,
                 device=None, dtype=torch.float32):
        self.B, self.D = batch_size, dof
        self.b0, self.b1, self.b2, self.a1, self.a2 = butter2_biquad_coeffs(fc, dt, device, dtype)
        self.device, self.dtype = device, dtype

        # states: x[n-1], x[n-2], y[n-1], y[n-2]
        self.x1 = torch.zeros(self.B, self.D, device=device, dtype=dtype)
        self.x2 = torch.zeros(self.B, self.D, device=device, dtype=dtype)
        self.y1 = torch.zeros(self.B, self.D, device=device, dtype=dtype)
        self.y2 = torch.zeros(self.B, self.D, device=device, dtype=dtype)

    @torch.no_grad()
    def reset(self, x0: torch.Tensor):
        """x0: [B, DOF] initial value (e.g., current joint positions)."""
        self.x1.copy_(x0)
        self.x2.copy_(x0)
        self.y1.copy_(x0)
        self.y2.copy_(x0)

    @torch.no_grad()
    def step(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B, DOF] unfiltered targets -> returns y: [B, DOF]."""
        y = (self.b0*x + self.b1*self.x1 + self.b2*self.x2
             - self.a1*self.y1 - self.a2*self.y2)

        # shift state (avoid reallocs)
        self.x2.copy_(self.x1)
        self.x1.copy_(x)
        self.y2.copy_(self.y1)
        self.y1.copy_(y)
        return y


class OneEuroFilter:
        """One-Euro filter for real-time smoothing of actions."""
        def __init__(self, beta=0.02, min_cutoff=0.3, d_cutoff=1.0, dt=1/120):
            self.beta = beta
            self.min_cutoff = min_cutoff
            self.d_cutoff = d_cutoff
            self.dt = dt
            self.prev_x = None
            self.dx = 0

        def filter(self, x):
            """Apply the One-Euro filter to smooth changes."""
            alpha = self.compute_alpha(self.min_cutoff)
            x_hat = alpha * x + (1 - alpha) * (self.prev_x if self.prev_x is not None else x)
            self.dx = x_hat - (self.prev_x if self.prev_x is not None else x_hat)
            d_alpha = self.compute_alpha(self.d_cutoff)
            self.dx = d_alpha * self.dx + (1 - d_alpha) * self.dx
            self.prev_x = x_hat
            return x_hat

        def compute_alpha(self, cutoff):
            return 1.0 / (1.0 + (cutoff * self.dt))

class Reward:
    """Container for one reward term."""
    def __init__(self, use: bool, scale: float):
        self.use = use
        self.scale = scale
        self.value: None

@configclass
class FrankaDMPCoatFoldEnvCfg(DirectRLEnvCfg):
    # env
    episode_length_s = 5
    max_episode_length = episode_length_s*120
    num_updates_per_episode = 5
    decimation = max_episode_length // num_updates_per_episode
    action_space = 82
    observation_space = 100
    state_space = 0
    enable_camera_recording = True
    plot_trajectories = False
    disable_init_motion = False
    use_weighted_atan_rewards = False
    use_weighted_exp_rewards = False
    write_joint_state = True
    cloth_length_scale = 0.7
    box_x_center = 0.5
    box_side = 0.4
    seed = 42

    torch.manual_seed(seed)

    # simulation
    sim: SimulationCfg = SimulationCfg(
        dt=1 / 120,
        render_interval=8, # for 15 fps, we use 8 (120 / 8 = 15)
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
        physx=PhysxCfg(
            gpu_max_particle_contacts=2**22, # Default is 2**20
            gpu_max_soft_body_contacts=2**23,
        ),
        render=RenderCfg(
            rendering_mode='performance',
        )
    )

    # scene
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=1, env_spacing=2.5, replicate_physics=False)

    # coat_on_hanger = DeformableObjectCfg(
    #     prim_path="/World/envs/env_.*/cuboid",
    #     init_state=DeformableObjectCfg.InitialStateCfg(pos=(0.3, 0.0, 0.35), rot=(0.5, -0.5, 0.5, -0.5)),
    #     spawn=sim_utils.UsdFileCfg(
    #         usd_path="/workspace/isaaclab/source/isaaclab_tasks/isaaclab_tasks/sim/deformable_cloth_hanger_v2.usd",
    #         # scale=(1, 1, 1),
    #     ),
    # )

    if enable_camera_recording:
        camera = CameraCfg(
            prim_path="/World/Camera",
            offset=CameraCfg.OffsetCfg(pos=(-5.0, -5.0, 3.0), rot=( 0.9020, -0.0828,  0.2000,  0.3736), convention="world"),
            data_types=["rgb"],
            spawn=sim_utils.PinholeCameraCfg(
                focal_length=24.0, focus_distance=400.0, horizontal_aperture=20.955, clipping_range=(0.1, 40.0)
            ),
            width=1080,
            height=720,
        )

    ### Robots
    robot = FRANKA_PANDA_HIGH_PD_CFG.replace(
        prim_path="/World/envs/env_.*/Robot1",
        spawn=sim_utils.UsdFileCfg(
            usd_path="/workspace/isaaclab/source/isaaclab_tasks/isaaclab_tasks/sim/franka_with_coat_v6_veryLowRes_sideways.usd",
            activate_contact_sensors=False,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=False,
                max_depenetration_velocity=5.0,
            ),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=False, solver_position_iteration_count=12, solver_velocity_iteration_count=1
            ),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            joint_pos={
                "panda_joint1": -0.0311,
                "panda_joint2": -0.5669,
                "panda_joint3":  0.0147,
                "panda_joint4": -1.5801,
                "panda_joint5":  0.0091,
                "panda_joint6":  1.0138,
                "panda_joint7":  0.7715,
                "panda_finger_joint.*": 0.005,
            },
            pos=(0.0, -0.0, 0.0),
            rot=(1.0, 0.0, 0.0, 0.0),
        ),)

    cloth_plain = DeformableObjectCfg(
        prim_path="/World/envs/env_.*/Robot1/deformable_cloth_hanger_v3/deformable_coat_scripted/Cloth",
        init_state=DeformableObjectCfg.InitialStateCfg(pos=(0, 0, 0), rot=(1, 0, 0, 0)),
        spawn=None,
    )

    handle = RigidObjectCfg(
        prim_path="/World/envs/env_.*/Robot1/deformable_cloth_hanger_v3/Cube",
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0, 0, 0), rot=(1, 0, 0, 0)),
        spawn=None,
    )

    # ### Robots
    # robot = FRANKA_PANDA_HIGH_PD_CFG.replace(
    #     prim_path="/World/envs/env_.*/Robot1",
    #     init_state=ArticulationCfg.InitialStateCfg(
    #         joint_pos={
    #             "panda_joint1": -0.0311,
    #             "panda_joint2": -0.5669,
    #             "panda_joint3":  0.0147,
    #             "panda_joint4": -1.5801,
    #             "panda_joint5":  0.0091,
    #             "panda_joint6":  1.0138,
    #             "panda_joint7":  0.7715,
    #             "panda_finger_joint.*": 0.005,
    #         },
    #         pos=(0.0, -0.0, 0.0),
    #         rot=(1.0, 0.0, 0.0, 0.0),
    #     ),)

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

    action_scale = 15 # default is 7.5
    dof_velocity_scale = 0.1
    filter_kernel_size = 7
    use_dynamic_rewards = False

    rewards = {
        "hull_reward": 
            Reward(True, 5.0),
        "height_balance_reward":
            Reward(True, 1.0),
        "containment_reward":
            Reward(True, 1.0),
        "action_penalty": 
            Reward(True, 1e-3),
    }

def make_circle_path_torch(center=(0, 0, 3), radius=10, num_points=720, device='cpu'):
    cx, cy, cz = center
    angles = torch.linspace(torch.pi, torch.pi + 2*torch.pi, num_points, device=device, requires_grad=False)
    xs = cx + radius * torch.cos(angles)
    ys = cy + radius * torch.sin(angles)
    zs = torch.full_like(xs, cz)
    return torch.stack([xs, ys, zs], dim=1)

class FrankaDMPCoatFoldEnv(DirectRLEnv):
    # pre-physics step calls
    #   |-- _pre_physics_step(action)
    #   |-- _apply_action()
    # post-physics step calls
    #   |-- _get_dones()
    #   |-- _get_rewards()
    #   |-- _reset_idx(env_ids)
    #   |-- _get_observations()

    cfg: FrankaDMPCoatFoldEnvCfg

    def __init__(self, cfg: FrankaDMPCoatFoldEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        # Set the seed
        self.seed(self.cfg.seed)

        # Setup video recording
        now = datetime.now().strftime("%Y%m%d_%H%M%S")
        if self.cfg.enable_camera_recording:
            self.video_folder = os.path.join("./logs/videos", now)
            os.makedirs(self.video_folder, exist_ok=True)        
        self.video_writer = None
        self.render_count = 0
        self.action_steps = torch.zeros(self.num_envs, device=self.device).int()
        self.init_camera_pose = torch.tensor([-10.0, -10.0, 0.3, 0.9238795, 0.0,  0.0,  0.3826834], device=self.device)
        self.new_camera_pose = torch.tensor([-10.0, -10.0, 0.3, 0.9238795, 0.0,  0.0,  0.3826834], device=self.device)
        self.camera_path = torch.linspace(self.init_camera_pose[0], self.init_camera_pose[0]+10, 720).to(self.device)
        self.circle_camera_path = make_circle_path_torch().to(self.device)

        self.frame_log_counter = 0
        self.probing_done = False

        self.dt = self.cfg.sim.dt * self.cfg.decimation
        self.ee_jacobi_idx = 7
        self.iteration_step = torch.tensor(0, device=self.device)
        self.default_states = None
        self.joints_created = False
        self.dmp_initialized = False
        self.reset_count = 0

        self.robot_entity_cfg = {}
        self._rewards = self.cfg.rewards

        diff_ik_cfg = DifferentialIKControllerCfg(
            command_type="pose", use_relative_mode=False, ik_method="dls")
        self._ik_controller = {}

        self.sigmoid = torch.nn.Sigmoid()

        # Initialize robot dictionaries
        self.robots = {}  # Store robot objects
        self.prev_rot_around_x = torch.zeros((self.num_envs), device=self.device)
        self.prev_joints = {}  # Store previous joint velocities
        self.filtered_dof_targets = {}  # Store smoothed action buffers
        self.robot_dof_targets = {}  # Store DOF targets
        self.gripper_actions = torch.zeros((self.num_envs, 2), device=self.device)
        self.ee_distances = torch.zeros((self.cfg.max_episode_length, self.num_envs), device=self.device)
        self.abs_pos = torch.zeros((self.cfg.max_episode_length, self.num_envs, 3), device=self.device)
        self.action_penalties = {'robot_1': torch.zeros((self.cfg.max_episode_length, self.num_envs), device=self.device)}

        H  = self.cfg.decimation
        self.prev_corners_buf = torch.zeros((H, self.num_envs, 6), device=self.device)
        self.prev_abs_buf = torch.zeros((H, self.num_envs, 2), device=self.device)
        self.target_joints_buffer = torch.zeros((H, self.num_envs, 7), device=self.device)
        self.actual_joints_buffer = torch.zeros((H, self.num_envs, 7), device=self.device)

        self.robot_dof_lower_limits = {}  # Store lower joint limits
        self.robot_dof_upper_limits = {}  # Store upper joint limits
        self.robot_dof_speed_scales = {}  # Store joint speed scales

        # Generate minimum jerk trajectory to imitate
        abs_points = [
            [0.0, 0.0, 0.81],
            [0.0, 0.0, 0.61],
            [0.0, 0.0, 0.41],
            [0.0, 0.0, 0.21],
        ]
        durations_relative = torch.tensor([
            1,
            2,
            1,
        ], device=self.device)

        durations = self.cfg.episode_length_s * durations_relative/durations_relative.sum()

        self._probing_traj = generate_minimum_jerk(
            waypoints=abs_points, durations=durations.cpu().tolist(), 
            num_points=self.cfg.decimation)
        self._probing_traj = torch.tensor(self._probing_traj, device=self.device)
        self._probing_traj = torch.cat((self._probing_traj[0].unsqueeze(0), 
                                     self._probing_traj, self._probing_traj[-1].unsqueeze(0)),dim=0)

        # Define the number of robots dynamically
        self.num_robots = 1

        for i in range(1, self.num_robots + 1):
            robot_key = f"robot_{i}"

            # Instantiate robot and store in dictionary
            self.robots[robot_key] = getattr(self, "_" + robot_key)

            # Initialize buffers
            num_joints = self.robots[robot_key].num_joints

            self.prev_joints[robot_key] = torch.zeros((self.cfg.filter_kernel_size, 
                                                       self.num_envs, 7), device=self.device)
            self.filtered_dof_targets[robot_key] = torch.zeros((self.num_envs, num_joints), device=self.device)
            self.robot_dof_targets[robot_key] = torch.zeros((self.num_envs, num_joints), device=self.device)

            # Joint limits and speed scales
            self.robot_dof_lower_limits[robot_key] = self.robots[robot_key].data.soft_joint_pos_limits[0, :, 0].to(device=self.device)
            self.robot_dof_upper_limits[robot_key] = self.robots[robot_key].data.soft_joint_pos_limits[0, :, 1].to(device=self.device)
            self.robot_dof_speed_scales[robot_key] = torch.ones_like(self.robot_dof_lower_limits[robot_key])

            self.robot_entity_cfg[robot_key] = SceneEntityCfg(robot_key, joint_names=["panda_joint.*"], body_names=["panda_hand"])
            self._ik_controller[robot_key] = DifferentialIKController(diff_ik_cfg, num_envs=self.num_envs, device=self.device)

            # Adjust finger joint speed scale
            finger_joints = ["panda_finger_joint1", "panda_finger_joint2"]
            for finger_joint in finger_joints:
                joint_idx = self.robots[robot_key].find_joints(finger_joint)[0]
                self.robot_dof_speed_scales[robot_key][joint_idx] = 0.1

        # Initialize dictionaries for robot-specific data
        self.robot_local_grasp_pos = {}  # Grasp position for each robot
        self.robot_local_grasp_rot = {}  # Grasp rotation for each robot
        self.hand_link_idx = {}  # Hand link indices
        self.left_finger_link_idx = {}  # Left finger indices
        self.right_finger_link_idx = {}  # Right finger indices
        self.gripper_forward_axis = {}  # Forward axis for each robot
        self.gripper_up_axis = {}  # Up axis for each robot

        for i in range(1, self.num_robots + 1):
            robot_key = f"robot_{i}"

            # ✅ Store hand and finger link indices for each robot
            self.hand_link_idx[robot_key] = self.robots[robot_key].find_bodies("panda_link7")[0][0]
            self.left_finger_link_idx[robot_key] = self.robots[robot_key].find_bodies("panda_leftfinger")[0][0]
            self.right_finger_link_idx[robot_key] = self.robots[robot_key].find_bodies("panda_rightfinger")[0][0]

        self.dmp_integrator = BatchDMPIntegrator(N_basis=25, dof=3, device=self.device)
        self.corner_dmp_integrator = BatchDMPIntegrator(N_basis=25, dof=12, device=self.device)
        self.corner_traj = torch.zeros((self.num_envs, self.cfg.decimation, 6), device=self.device)
        self.corner_dmp_weights = torch.zeros((self.num_envs, 12, 25), device=self.device)
        self.corner_y0 = torch.zeros((self.num_envs, 12), device=self.device)
        self.corner_goal = torch.zeros((self.num_envs, 12), device=self.device)

        self.lp1 = {}
        self.lp1['robot_1'] = BiquadLP2Batch(self.scene.cfg.num_envs, 7, fc=0.5, dt=self.physics_dt, device=self.device)

    def plot_final_poses(self):
        if not hasattr(self, "absolute_pose_buffer_f"):
            self.absolute_pose_buffer_f = []
            self.rgb_buffer_f = []
            self.frame_log_counter_f = 0

        if self.action_steps[0] == 0:
            self.absolute_pose_buffer_f = []
            self.rgb_buffer_f = []
            self.frame_log_counter_f = 0

        log_step = 360  # We log the entire trajectory at 360 steps
        self.frame_log_counter_f += 1

        # Record absolute pose for each environment
        abs_pose = self._get_robot_ee_pose().clone()  # (num_envs, 7)
        abs_pose[:,:3] = abs_pose[:,:3] - self.scene.env_origins
        self.absolute_pose_buffer_f.append(abs_pose.cpu())  # Log poses for all environments

        if (self.frame_log_counter_f != log_step):  # Check if it’s time to log
            return

        # Record RGB snapshots at the end of the episode
        if hasattr(self, '_camera') and self._camera.data.output["rgb"] is not None:
            rgb_img = self._camera.data.output['rgb'] / 255
            self.rgb_buffer_f.append(rgb_img)

        # Make the panel figure (2 rows, 1 column)
        fig, axs = plt.subplots(2, 1, figsize=(6, 6.5))  # 2 rows, 1 column

        # Absolute pose subplot (top row for all environments)
        abs_pose_data = torch.stack(self.absolute_pose_buffer_f)[:,:,:3]  # Get the absolute poses for all environments
        num_envs = abs_pose_data.shape[1]  # Number of environments

        import numpy as np
        colors = plt.cm.viridis(np.linspace(0.2, 0.8, num_envs))
        linestyles = ['--', '-.', ':', '-']
        for i in range(num_envs):
            x = abs_pose_data[:, i, 0]  # x positions for environment i
            z = abs_pose_data[:, i, 2]  # z positions for environment i
            axs[0].plot(x[0], z[0], marker='o', linestyle='-', color=colors[i])  # Plot trajectory for each environment
            axs[0].plot(x, z, linestyle=linestyles[i], color=colors[i], label=f"Env {i+1}")  # Plot trajectory for each environment

        # axs[0].set_title("Absolute Pose Trajectories (All Environments)", fontsize=14)
        axs[0].set_xlim(-0.2, 0.8)
        axs[0].set_ylim(0.0, 1.0)
        axs[0].set_xlabel("x [m]")
        axs[0].set_ylabel("z [m]")
        axs[0].grid(True)
        axs[0].legend()

        # RGB snapshot subplot (bottom row for the final image of all environments)
        axs[1].imshow(self.rgb_buffer_f[-1][0].cpu())  # Last RGB image from buffer
        axs[1].axis('off')
        # axs[1].set_title("Final Image at the End of Episode", fontsize=14)

        # Save the plot
        os.makedirs('./logs/snapshots', exist_ok=True)
        fname = f"./logs/snapshots/final_{self.frame_log_counter_f:05d}.pdf"
        plt.tight_layout()
        plt.savefig(fname)
        plt.close()

        # Clear buffers for the next cycle
        self.absolute_pose_buffer_f.clear()
        self.rgb_buffer_f.clear()
        breakpoint()


    def log_trajectory_panel(self):
        if not hasattr(self, "absolute_pose_buffer"):
            self.absolute_pose_buffer = []
            self.rgb_buffer = []
            self.frame_log_counter = 0
            self.rgb_counter = 0
            
        if self.action_steps[0] == 0:
            self.absolute_pose_buffer = []
            self.rgb_buffer = []
            self.frame_log_counter = 0
            self.rgb_counter = 0

        num_subplots = 6
        indices = [60, 90, 120, 160, 220, 300]
        self.frame_log_counter += 1

        # Record absolute pose
        abs_pose = self._get_robot_ee_pose().clone()  # (num_envs, 7)
        abs_pose[:,:3] = abs_pose[:,:3] - self.scene.env_origins
        self.absolute_pose_buffer.append(abs_pose[0].cpu())  # Only log env 0

        if self.rgb_counter >= len(indices):
            return

        if (self.frame_log_counter != indices[self.rgb_counter]):#  or (self.episode_length_buf.sum() == 0):
            return

        # Record RGB snapshot
        if hasattr(self, '_camera') and self._camera.data.output["rgb"] is not None:
            rgb_img = self._camera.data.output['rgb'][0] / 255
            self.rgb_buffer.append(rgb_img)
            self.rgb_counter += 1

        # Wait until we have 20 samples
        if len(self.rgb_buffer) < num_subplots:
            return

        # Make the panel figure (20 in a row)
        fig, axs = plt.subplots(2, num_subplots, figsize=(3.5 * num_subplots, 2 * 2.5))  # 2 rows, 20 columns

        for i in range(0, num_subplots):
            # Pose subplot
            ### NOTE: We start from i+1 so it's not just a point
            x = [pose[0].item() for pose in self.absolute_pose_buffer[:indices[i]]]
            z = [pose[2].item() for pose in self.absolute_pose_buffer[:indices[i]]]
            if i==0:
                axs[0, i].plot(x[0], z[0], marker='o', linestyle='-', color='tab:green')
                axs[0, i].plot(x, z, linestyle='-', color='tab:purple')
            else:
                axs[0, i].plot(x[0], z[0], marker='o', linestyle='-', color='tab:green')
                axs[0, i].plot(x, z, linestyle='-', color='tab:purple')
            # axs[0, i].set_title("Absolute Pose Trajectory (Env 0)")
            axs[0, i].set_xlim(-0.2, 0.6)
            axs[0, i].set_ylim(0.0, 1.0)
            axs[0, i].set_xlabel("x [m]")
            if i==0:
                axs[0, i].set_ylabel("z [m]")
            axs[0, i].grid(True)
            # axs[0, i].axis('equal')

            # RGB snapshot
            axs[1, i].imshow(self.rgb_buffer[i].cpu())
            axs[1, i].axis('off')

        # Save plot
        os.makedirs('./logs/snapshots', exist_ok=True)
        fname = f"./logs/snapshots/panel_{self.frame_log_counter:05d}.pdf"
        plt.tight_layout()
        plt.savefig(fname)
        plt.close()

        # Clear buffers
        self.absolute_pose_buffer.clear()
        self.rgb_buffer.clear()
        breakpoint()

    def _spawn_box_walls_with_wrappers(self, x_center=0.5, side=0.5, thickness=0.01, height=0.30, color=(0.35, 0.35, 0.35)):
        """Spawn 4 thin kinematic walls (0.5 m square) at ~x=0.5 in every env using sim_utils."""
        import omni.usd
        from pxr import UsdGeom, Gf
        stage = omni.usd.get_context().get_stage()

        # 1) Ensure the parent Xform exists in EACH env (prevents the 'source prim' error)
        for env_idx in range(self.scene.cfg.num_envs):
            base = f"/World/envs/env_{env_idx}/Walls"
            if not stage.GetPrimAtPath(base):
                UsdGeom.Xform.Define(stage, base)

        # 2) Common sim-utils configs for all walls
        mat = sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply", restitution_combine_mode="multiply",
            static_friction=1.0, dynamic_friction=1.0, restitution=0.0
        )
        rigid = sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True, disable_gravity=True)
        collide = sim_utils.CollisionPropertiesCfg()
        vis = sim_utils.PreviewSurfaceCfg(diffuse_color=color)

        def make_wall(name: str, size_xyz, pos_xyz):
            from isaaclab.assets import RigidObjectCfg
            cfg = RigidObjectCfg(
                prim_path=f"/World/envs/env_.*/Walls/{name}",  # regex -> all envs
                init_state=RigidObjectCfg.InitialStateCfg(pos=pos_xyz, rot=(1, 0, 0, 0)),
                spawn=sim_utils.CuboidCfg(
                    size=size_xyz,
                    physics_material=mat,
                    rigid_props=rigid,
                    collision_props=collide,
                    visual_material=vis,
                ),
            )
            obj = cfg.class_type(cfg)  # wrapper with .data / write_* available
            setattr(self, f"_wall_{name.lower()}", obj)
            return obj

        # Geometry (env-relative positions)
        hz = height * 0.5
        sx, sy, sz = side, side, height
        t = thickness

        # Long along Y (N/S) and long along X (E/W)
        self._wall_north = make_wall("North", (sx, t,  sz), (x_center, +side*0.5, hz))
        self._wall_south = make_wall("South", (sx, t,  sz), (x_center, -side*0.5, hz))
        self._wall_east  = make_wall("East",  (t,  sy, sz), (x_center + side*0.5, 0.0, hz))
        # self._wall_west  = make_wall("West",  (t,  sy, sz), (x_center - side*0.5, 0.0, hz))

    def _spawn_box_walls(self, x_center=0.5, side=0.5, thickness=0.01, height=0.30, color=(0.35, 0.35, 0.35)):
        """
        Spawn 4 thin cube walls that form a square 'box' of size `side` at x ≈ `x_center`.
        Walls are static colliders.
        """
        stage = omni.usd.get_context().get_stage()

        for env_idx in range(self.scene.cfg.num_envs):
            env_origin = self.scene.env_origins[env_idx]
            ox = float(env_origin[0].item()); oy = float(env_origin[1].item()); oz = float(env_origin[2].item())

            base_path = f"/World/envs/env_{env_idx}/Walls"
            if not stage.GetPrimAtPath(base_path):
                UsdGeom.Xform.Define(stage, base_path)

            # Wall dimensions (using a unit cube scaled non-uniformly)
            # Two walls long along Y (north/south), two along X (east/west).
            zc = oz + height * 0.5  # center so they rest on ground (z=0)
            cx = ox + x_center
            cy = oy

            walls = [
                # name,            scale(x,y,z),                   position(x,y,z)
                ("North", (side, thickness, height),              (cx, cy + side * 0.5, zc)),
                ("South", (side, thickness, height),              (cx, cy - side * 0.5, zc)),
                ("East",  (thickness, side, height),              (cx + side * 0.5, cy, zc)),
                ("West",  (thickness, side, height),              (cx - side * 0.5, cy, zc)),
            ]

            for name, scale_xyz, pos_xyz in walls:
                prim_path = f"{base_path}/{name}"
                cube = UsdGeom.Cube.Define(stage, prim_path)
                cube.CreateSizeAttr(1.0)  # unit cube, we scale below

                # xform ops (scale, then translate)
                physicsUtils.setup_transform_as_scale_orient_translate(cube)
                physicsUtils.set_or_add_scale_op(cube, Gf.Vec3f(*scale_xyz))
                physicsUtils.set_or_add_translate_op(cube, Gf.Vec3f(*pos_xyz))

                # make it a static collider
                prim = stage.GetPrimAtPath(prim_path)
                UsdPhysics.CollisionAPI.Apply(prim)
                # optional: visual color
                cube.GetDisplayColorAttr().Set([Gf.Vec3f(*color)])

    def _create_cloth(self, stage, env_idx):
        cloth_mesh_path = Sdf.Path(f"/World/envs/env_{env_idx}/Cloth")
        particle_material_path = Sdf.Path("/World/particleMaterial")

        # ✅ Create a mesh that will be turned into cloth
        plane_resolution = 20
        plane_width = 70

        # ✅ Get the environment center from Isaac Lab
        env_center = self.scene.env_origins[env_idx]  # This gives (x, y, z) of the env center

        # ✅ Adjust the cloth spawn height relative to the environment center
        cloth_position = Gf.Vec3f(float(env_center[0]), 
                                  float(env_center[1]), 
                                  float(env_center[2]))

        success, tmp_path = omni.kit.commands.execute(
            "CreateMeshPrimWithDefaultXform",
            prim_type="Plane",
            u_patches=plane_resolution,
            v_patches=plane_resolution,
            u_verts_scale=1,
            v_verts_scale=1,
            half_scale=0.5 * plane_width,
        )
        if not success:
            return

        omni.kit.commands.execute("MovePrim", path_from=tmp_path, path_to=cloth_mesh_path)
        cloth_mesh = UsdGeom.Mesh.Define(stage, cloth_mesh_path)

        physicsUtils.setup_transform_as_scale_orient_translate(cloth_mesh)
        physicsUtils.set_or_add_translate_op(cloth_mesh, cloth_position/100)  # Set cloth in env center
        physicsUtils.set_or_add_orient_op(cloth_mesh, Gf.Quatf(1.0, Gf.Vec3f(0.0, 0.0, 0.0)))  # No rotation
        physicsUtils.set_or_add_scale_op(cloth_mesh, Gf.Vec3f(1.0))  # No scaling        
    
        # ✅ Bind the SAME Material for all cloth instances
        material_path = particle_material_path
        # material_path = Sdf.Path("/World/Materials/ClothMaterial")  # Centralized material
        omni.kit.commands.execute(
            "BindMaterialCommand",
            prim_path=cloth_mesh_path,  # Target cloth mesh
            material_path=material_path,  # Same material for all
            strength=None
        )

        # ✅ Reuse existing particle system & material
        particle_system_path = Sdf.Path("/World/particleSystem")

        # ✅ Apply existing particle system to cloth (instead of creating a new one)
        stretchStiffness = 10000.0
        bendStiffness = 200.0
        shearStiffness = 100.0
        damping = 0.2

        self._particle_api = particleUtils.add_physx_particle_cloth(
            stage=stage,
            path=cloth_mesh_path,
            dynamic_mesh_path=None,
            particle_system_path=particle_system_path,  # Use the same system
            spring_stretch_stiffness=stretchStiffness,
            spring_bend_stiffness=bendStiffness,
            spring_shear_stiffness=shearStiffness,
            spring_damping=damping,
            self_collision=True,
            self_collision_filter=True,
        )

        # ✅ Configure mass
        particle_mass = 0.002
        num_verts = len(cloth_mesh.GetPointsAttr().Get())
        mass = particle_mass * num_verts
        massApi = UsdPhysics.MassAPI.Apply(cloth_mesh.GetPrim())
        massApi.GetMassAttr().Set(mass)

        # Set color and access to positions
        cloth_prim = self.scene.stage.GetPrimAtPath(cloth_mesh_path)
        cloth_color_attr = cloth_prim.GetAttribute("primvars:displayColor")
        cloth_color_attr.Set([Gf.Vec3f(0.2, 0.0, 0.220)])

        return cloth_mesh_path

    def _setup_camera_writer(self, step):
        """
        Called when we enter a recording window to create a new VideoWriter.
        We generate a filename that includes the block index (e.g. “record_0.avi”, “record_1.avi”, …).
        """
        block_index = step // 20000
        filename = f"{self.video_folder}/record_block_{block_index}.mp4"
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')

        # Capture one frame to get width/height:
        img = self._camera.data.output['rgb'][0]
        H, W, _ = img.shape

        self.video_writer = cv2.VideoWriter(filename, fourcc, 15, (W, H))

    def _close_camera_writer(self):
        """Called when leaving a recording window to release the VideoWriter."""
        if self.video_writer is not None:
            self.video_writer.release()
            self.video_writer = None

    def _capture_and_write_frame(self):
        """
        Grabs a single frame from your Isaac camera and writes it to disk via OpenCV.
        """
        # 1) Capture an RGB numpy array from the Isaac camera (HxWx3, dtype=uint8):
        self._camera.update(self.physics_dt)
        img_rgb = self._camera.data.output['rgb'][0]
        # 2) Convert to BGR (OpenCV expects BGR):
        img_bgr = cv2.cvtColor(img_rgb.cpu().numpy(), cv2.COLOR_RGB2BGR)
        # 3) Write to the open VideoWriter:
        self.video_writer.write(img_bgr)

    def _setup_scene(self):
        # ✅ Create robots
        self._robot_1 = Articulation(self.cfg.robot)
        self.cloth_lengths = torch.zeros(self.scene.cfg.num_envs, device=self.device)

        self.scene.articulations["robot_1"] = self._robot_1

        # ✅ Configure terrain
        self.cfg.terrain.num_envs = self.scene.cfg.num_envs
        self.cfg.terrain.env_spacing = self.scene.cfg.env_spacing
        self._terrain = self.cfg.terrain.class_type(self.cfg.terrain)

        # self._coat_on_hanger = self.cfg.coat_on_hanger.class_type(self.cfg.coat_on_hanger)
        self._cloth_plain = self.cfg.cloth_plain.class_type(self.cfg.cloth_plain)
        self._handle_1 = self.cfg.handle.class_type(self.cfg.handle)

        # ✅ Clone environments
        self.scene.clone_environments(copy_from_source=True)

        # for env_idx in range(self.scene.cfg.num_envs):
        #     # Get the Cloth prim for the current environment
        #     cloth_prim_path = f"/World/envs/env_{env_idx}/cuboid/cuboid"
        #     cloth_prim = get_prim_at_path(cloth_prim_path)

        #     # Generate random scale factors for x, y, z axes
        #     scale = torch.tensor([0.2, 0.7, 0.005])

        #     # Apply the random scale to the Cloth prim
        #     cloth_prim.GetAttribute("xformOp:scale").Set(Gf.Vec3f(*scale.tolist()))
        #     # cloth_prim.GetAttribute('xformOp:translate').Set(
        #     #     Gf.Vec3f(0.0, 0.0, 0.356 + ((1 - random_scale[1]) * 0.356).item()))
        #     self.cloth_lengths[env_idx] = scale[1].item() * 0.7

        if self.cfg.enable_camera_recording:
            self._camera = self.cfg.camera.class_type(self.cfg.camera)

        # ✅ Add lights (optional)
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

        # ✅ Get the stage
        stage = omni.usd.get_context().get_stage()

        # ✅ Hide the environment
        environment_prim_path = "/World/ground/Environment"
        environment_prim = stage.GetPrimAtPath(environment_prim_path)
        if environment_prim:
            UsdGeom.Imageable(environment_prim).GetVisibilityAttr().Set(UsdGeom.Tokens.invisible)

        # ----------------------------
        # ✅ CREATE A VISIBLE GROUND PLANE WITH MATERIAL
        # ----------------------------
        visual_plane_path = "/World/ground/GroundPlane/VisualPlane"
        material_path = "/World/Materials/GroundMaterial"

        # ✅ Create the ground plane if it doesn't exist
        if not stage.GetPrimAtPath(visual_plane_path):
            visual_plane = UsdGeom.Mesh.Define(stage, visual_plane_path)

            # ✅ Set plane size and position
            visual_plane.GetPointsAttr().Set([
                Gf.Vec3f(-50, -50, 0), Gf.Vec3f(50, -50, 0),
                Gf.Vec3f(50, 50, 0), Gf.Vec3f(-50, 50, 0)
            ])
            visual_plane.GetFaceVertexIndicesAttr().Set([0, 1, 2, 2, 3, 0])
            visual_plane.GetFaceVertexCountsAttr().Set([3, 3])

            # ✅ Move it to the same height as the collision plane
            visual_plane.AddTranslateOp().Set(Gf.Vec3f(0, 0, 0))
            visual_plane.GetDisplayColorAttr().Set([Gf.Vec3f(0.00, 0.50, 0.00)])  # Green

        # ✅ Create a new material for the ground
        material_path = "/World/Materials/GroundMaterial"
        material_prim = stage.DefinePrim(material_path, "Material")
        material = UsdShade.Material(material_prim)

        # ✅ Create a shader
        shader_path = material_path + "/Shader"
        shader = UsdShade.Shader.Define(stage, shader_path)
        shader.CreateIdAttr("UsdPreviewSurface")  # Use USD's default shader

        # ✅ Set ground color to dark gray
        shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(0.1, 0.1, 0.1))  # Dark gray
        shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.0)  # Rough texture
        shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(0.0)  # Non-metallic

        # ✅ Bind the material to the ground plane
        UsdShade.MaterialBindingAPI(visual_plane).Bind(material)

        # self._spawn_box_walls(x_center=0.3, side=0.4, thickness=0.01, height=0.3)
        # self._spawn_box_walls_with_wrappers(x_center=self.cfg.box_x_center, 
        #                                     side=self.cfg.box_side, thickness=0.01, height=0.30)

    def _get_robot_ee_pose(self):
        ee_pose_1_w = self._robot_1.data.body_state_w[
            :, self.robot_entity_cfg['robot_1'].body_ids[0], 0:7
        ]
        return ee_pose_1_w

    # pre-physics step calls
    def _pre_physics_step(self, actions: torch.Tensor):

        print("Pre-physics step")
        dmp_parameters = actions.clone()
        dmp_parameters[:,3] = dmp_parameters[:,2].clamp(0.2, 0.7) # X
        dmp_parameters[:,4] = dmp_parameters[:,2].clamp(-0.6, 0.6) # Y
        dmp_parameters[:,5] = dmp_parameters[:,3].clamp(0.0, 1.0) # Z
        dmp_parameters = dmp_parameters.clamp(-0.5, 0.5)
        # dmp_parameters[:,-1] = (dmp_parameters[:,-1].abs() + 2).clamp(2.0, 3.0)  # Tau in seconds

        self.actions = dmp_parameters

        current_absolute_pose = self._get_robot_ee_pose()
        current_absolute_pose[:,0:3] -= self.scene.env_origins
        dmp_parameters[:,0] = current_absolute_pose[:,0]
        dmp_parameters[:,1] = current_absolute_pose[:,1]
        dmp_parameters[:,2] = current_absolute_pose[:,2]

        # dmp_tau = dmp_parameters[:,-1].clone()
        dmp_tau = (torch.zeros(self.num_envs, device=self.device) + 
                   self.cfg.episode_length_s // self.cfg.num_updates_per_episode) - 0.5
        self.actions[:,-1] = dmp_tau
        print("Tau mean:", dmp_tau.mean())

        ### Reset only at the beginning of the episode
        # reset_dmp_indices = self.episode_length_buf == 0
        # reset_dmp_indices = torch.where(reset_dmp_indices == True)[0]
        ### Reset at every step (e.g. twice per episode)
        reset_dmp_indices = torch.arange(0, dmp_parameters.shape[0])

        # self.corner_traj = self.prev_corners_buf.permute(1,0,2)
        # self.corner_dmp_weights = self.corner_dmp_integrator.encode(
        #     self.corner_traj, time=torch.tensor(self.physics_dt, device=self.device))
        # self.corner_y0 = self.corner_traj[:,0,:].clone()
        # self.corner_goal = self.corner_traj[:,-1,:].clone()

        if len(reset_dmp_indices) > 0:
            print("Resetting DMP parameters")
            self.dmp_integrator.reset_indices(reset_dmp_indices, dmp_parameters, 
                                              dmp_tau, dt=self.physics_dt, variant=2)
        elif not self.dmp_initialized:
            reset_dmp_indices = torch.arange(0, dmp_parameters.shape[0])
            self.dmp_integrator.reset_indices(reset_dmp_indices, dmp_parameters, 
                                              dmp_tau, dt=self.physics_dt, variant=2)
            self.dmp_initialized = True

        self.action_steps *= 0

    def _apply_action(self):
        """Apply joint position targets for multiple robots dynamically."""

        if self.action_steps[0] == 0:
            for robot_key in self.robots.keys():
                q0 = self.robots[robot_key].data.joint_pos[:, :7]
                self.lp1[robot_key].reset(q0)

        t, y, dy, ddy = self.dmp_integrator.step()
        self.dmp_integrator.x[t >= self.actions[:,-1]] = 0.1353

        current_absolute_pose = self._get_robot_ee_pose()
        # self.ee_distances[self.action_steps] = self._get_ee_distance()
        self.abs_pos[self.action_steps] = current_absolute_pose[:, 0:3].clone()

        if self.cfg.enable_camera_recording:
            in_window = (self.iteration_step % 20000) < 720
            just_entered = (in_window and self.video_writer is None)

            if just_entered:
                self.render_count = 0
                self._setup_camera_writer(self.iteration_step)
                # self._camera.set_world_poses(self.init_camera_pose[0:3].unsqueeze(0))

            # If we are in the window, capture & write one frame:
            if in_window and self.video_writer is not None:
                ### Make camera move linearly
                # new_camera_x = self.camera_path[self.render_count]
                # new_camera_y = self.camera_path[self.render_count]
                # self.new_camera_pose[0] = new_camera_x
                # self.new_camera_pose[1] = new_camera_y
                # self._camera.set_world_poses(self.new_camera_pose[0:3].unsqueeze(0))

                ### Make camera move in a circle
                # self._camera.set_world_poses_from_view(self.circle_camera_path[self.render_count].unsqueeze(0),
                #                                        torch.tensor([0,0,0.5], device=self.device))

                ### Set camera to a fixed position
                ### Good for top view of 4 envs
                # self._camera.set_world_poses_from_view(torch.tensor([0., 0., 6.5], device=self.device).unsqueeze(0),
                #                                     torch.tensor([0., 0., 0.], device=self.device))
                # self.camera_position_set = 1
                ### Good for watching one env
                # self._camera.set_world_poses_from_view(torch.tensor([1.6, 1.6, 1.6], device=self.device).unsqueeze(0),
                #                                        torch.tensor([0.3, 0.0, 0.35], device=self.device))

                if self.render_count % self.cfg.sim.render_interval == 0:
                    self._capture_and_write_frame()
                self.render_count += 1

            # If we just left a window, close the writer:
            just_left = ((not in_window) and self.video_writer is not None)
            if just_left:
                self.render_count = 0
                self._close_camera_writer()

        if (self.action_steps < (self.cfg.max_episode_length - 60)).all():
            self.inverse_kinematics(y)
            for robot_key in self.robots.keys():
                # ✅ Set joint position target separately for each robot
                if self.cfg.write_joint_state:
                    q_des = self.robot_dof_targets[robot_key][:, :7] 
                    q_smooth = self.lp1[robot_key].step(q_des) # [B, D]
                    self.robot_dof_targets[robot_key][:, :7]  = q_smooth
                    self.robots[robot_key].write_joint_position_to_sim(
                        self.robot_dof_targets[robot_key])
                else:
                    self.robots[robot_key].set_joint_position_target(
                        self.robot_dof_targets[robot_key])
                
                # self.target_joints_buffer[self.action_steps[0]] = self.robot_dof_targets[robot_key][:,0:7]
                # self.actual_joints_buffer[self.action_steps[0]] = self.robots[robot_key].data.joint_pos[:,0:7]
        else:
            target_y = current_absolute_pose[:,0:3] - self.scene.env_origins
            self.inverse_kinematics(target_y)
            for robot_key in self.robots.keys():
                # ✅ Set joint position target separately for each robot
                if self.cfg.write_joint_state:
                    q_des = self.robot_dof_targets[robot_key][:, :7] 
                    q_smooth = self.lp1[robot_key].step(q_des) # [B, D]
                    self.robot_dof_targets[robot_key][:, :7]  = q_smooth
                    self.robots[robot_key].write_joint_position_to_sim(
                        self.robot_dof_targets[robot_key])
                else:
                    self.robots[robot_key].set_joint_position_target(
                        self.robot_dof_targets[robot_key])
                
                self.target_joints_buffer[self.action_steps[0]] = self.robot_dof_targets[robot_key][:,0:7]
                self.actual_joints_buffer[self.action_steps[0]] = self.robots[robot_key].data.joint_pos[:,0:7]

        if self.cfg.plot_trajectories:
            self.log_trajectory_panel()
            self.plot_final_poses()

        self.iteration_step += 1
        self.action_steps += 1

    def inverse_kinematics(self, y):
        current_absolute_pose = self._get_robot_ee_pose()
        target_absolute_pose = current_absolute_pose.clone() * 0.0
        target_absolute_pose[:,0] = y[:,0]
        target_absolute_pose[:,1] = y[:,1]
        target_absolute_pose[:,2] = y[:,2]
        target_absolute_pose[:,0:3] += self.scene.env_origins
        
        target_absolute_pose[:, 3:] = torch.tensor([0, 1, 0, 0], device=self.device)
        
        # Transform from global into each robot coordinate system
        larm_pose = target_absolute_pose

        root_pose_w = self._robot_1.data.root_state_w[:, 0:7]
        larm_pos_b, _ = subtract_frame_transforms(
            root_pose_w[:, 0:3], root_pose_w[:, 3:7], 
            larm_pose[:, 0:3], larm_pose[:, 3:7]
        )
        
        rot_around_x = self.prev_rot_around_x + self.actions[:, 2]
        rot_around_x = rot_around_x.clamp(-0.6, 0.6)

        larm_goal_quat_b = torch.zeros((self.num_envs, 4), device=self.device)
        larm_goal_quat_b[:,1] = 1.0

        larm_pose_b = torch.cat((larm_pos_b, larm_goal_quat_b), dim=-1)

        self._ik_controller['robot_1'].set_command(larm_pose_b)
        self.prev_rot_around_x = rot_around_x.clone()

        for i, robot_key in enumerate(self.robots.keys()):  # Loop through all robots dynamically
            self.robot_dof_targets[robot_key][:,-2:] = self.gripper_actions[:,i].unsqueeze(1).repeat(1,2)
            # ✅ Compute new target positions for each robot separately
            robot = self.robots[robot_key]
            robot_entity_cfg = self.robot_entity_cfg[robot_key]

            ee_pose_w = robot.data.body_state_w[
                :, robot_entity_cfg.body_ids[0], 0:7]

            # Obtain robot's Jacobian matrix
            jacobian_w = robot.root_physx_view.get_jacobians()[
                :, self.ee_jacobi_idx, :, 
                robot_entity_cfg.joint_ids]
            base_rot = robot.data.root_quat_w
            jacobian = self._ik_controller[robot_key].get_jacobian_in_root_frame(jacobian_w, base_rot)

            # Get root pose and joint positions
            root_pose_w = robot.data.root_state_w[:, 0:7]
            current_joint_pos = robot.data.joint_pos[
                :, robot_entity_cfg.joint_ids]

            ee_pos_b, ee_quat_b = subtract_frame_transforms(
                root_pose_w[:, 0:3], root_pose_w[:, 3:7], 
                ee_pose_w[:, 0:3], ee_pose_w[:, 3:7]
            )

            # Compute new joint positions using IK
            self.robot_dof_targets[robot_key][:, 0:7] = self._ik_controller[robot_key].compute(
                ee_pos_b, ee_quat_b, jacobian, current_joint_pos)

            padded_joint_pos = F.pad(input=current_joint_pos, pad=(0,2), mode='constant', value=0)
            self.joint_change = self.robot_dof_targets[robot_key] - padded_joint_pos

            # self.scaled_joint_change = self.robot_dof_speed_scales[robot_key] * self.dt * (
            #     self.robot_dof_targets[robot_key] - padded_joint_pos) * self.cfg.action_scale

            self.action_penalties[robot_key][self.action_steps] = self.joint_change.abs().mean(dim=-1)

            if self.cfg.write_joint_state:
                ### Limit change of joints if writing directly
                self.clamped_joint_change = self.joint_change # .clamp(-1e-1, 1e-1)
                new_targets = padded_joint_pos + self.clamped_joint_change
            else:
                ### Or just send it to target positions
                new_targets = self.robot_dof_targets[robot_key]

            # ✅ Clamp values within each robot's DOF limits
            new_targets = torch.clamp(new_targets, self.robot_dof_lower_limits[robot_key], self.robot_dof_upper_limits[robot_key])

            self.filtered_dof_targets[robot_key] = new_targets
            # self.filtered_dof_targets[robot_key] = self.one_euro_filter.filter(new_targets)

            self.prev_joints[robot_key] = self.prev_joints[robot_key].roll(-1, dims=0)
            self.prev_joints[robot_key][-1] = new_targets[:, 0:7]

            self.robot_dof_targets[robot_key] = self.filtered_dof_targets[robot_key]
        return current_absolute_pose

    # post-physics step calls
    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        # Calculate distance between grippers
        terminated = 0
        truncated = self.episode_length_buf >= self.max_episode_length

        if truncated.any():
            truncated[:] = True

        return terminated, truncated

    def _get_ee_distance(self):
        robot_keys = list(self.robots.keys())  # Get robot keys dynamically
        ee_1_pos = self.robots[robot_keys[0]].data.body_pos_w[:, self.left_finger_link_idx[robot_keys[0]]]
        ee_2_pos = self.robots[robot_keys[0]].data.body_pos_w[:, self.left_finger_link_idx[robot_keys[0]]]

        ee_distance = torch.norm(ee_1_pos - ee_2_pos, dim=-1)

        return ee_distance

    def _get_corners(self):
        corners = [self._handle_1.root_physx_view.get_transforms()[:,0:3],
                   self._handle_1.root_physx_view.get_transforms()[:,0:3],
                   self._handle_1.root_physx_view.get_transforms()[:,0:3],
                   self._handle_1.root_physx_view.get_transforms()[:,0:3]]
        corners = torch.stack(corners).permute(1,0,2)
        return corners

    def compute_rewards(self):
        """
        Compute individual reward term values and return a dict of reward_name -> value tensor.
        """
        # Gather cloth positions (shape: [num_envs, num_points, 3])
        cloth_positions = self._cloth_plain.root_physx_view.get_nodal_positions().reshape(self.num_envs, -1, 3)

        # Ideal area based on cloth area / 3
        ideal_area = (0.7 * 0.2 / 3.0)

        # Reward initialization
        hull_reward = torch.zeros(self.num_envs, device=cloth_positions.device)
        height_balance_reward = torch.zeros(self.num_envs, device=cloth_positions.device)

        for env_idx in range(self.num_envs):
            nodes = cloth_positions[env_idx].cpu().numpy()

            # XY projection for Convex Hull
            xy = nodes[:, :2]  # Only x and y for the hull
            hull = ConvexHull(xy)
            hull_area = hull.volume  # 2D hull -> area
            
            hull_reward[env_idx] = 1.0 - (torch.abs(torch.tensor(
                hull_area - ideal_area, device=self.device)) * 10)

        # # Height distribution reward
        # node_heights = cloth_positions[:, :, 2]  # Z-coordinates for each cloth node
        # height_threshold = 0.08  # Threshold for balancing (can adjust this)

        # # Count how many nodes are above and below the height threshold
        # below_threshold = (node_heights <= height_threshold)

        # # Reward for balancing: encourage half nodes above and half below the height threshold
        # height_balance_reward = 1 / (0.1 + torch.abs(below_threshold.float().mean(1) - 0.5))
        
        mean_cloth_height = cloth_positions[:, :, 2].mean(dim=-1)
        height_balance_reward = (1.0 / (0.1 + mean_cloth_height))

        x_min = self.cfg.box_x_center - self.cfg.box_side / 2
        x_max = self.cfg.box_x_center + self.cfg.box_side / 2
        y_min = 0 - self.cfg.box_side / 2
        y_max = 0 + self.cfg.box_side / 2

        # XY coordinates of all vertices
        xy = cloth_positions[:, :, :2]  # [num_envs, num_nodes, 2]
        xy -= self.scene.env_origins[:, :2].unsqueeze(1)

        # Check which vertices are inside box
        inside_x = (xy[:, :, 0] >= x_min) & (xy[:, :, 0] <= x_max)
        inside_y = (xy[:, :, 1] >= y_min) & (xy[:, :, 1] <= y_max)

        inside_box = inside_x & inside_y  # [num_envs, num_nodes]

        # Fraction of vertices inside the box → reward
        containment_reward = inside_box.float().mean(dim=1)  # [num_envs]

        # Compute each reward term
        rewards = {}

        if not self.cfg.use_weighted_atan_rewards and not self.cfg.use_weighted_exp_rewards:
            rewards["hull_reward"] = hull_reward * self._rewards["hull_reward"].scale
            rewards["height_balance_reward"] = height_balance_reward * self._rewards["height_balance_reward"].scale
            rewards["containment_reward"] = containment_reward * self._rewards["containment_reward"].scale
        # elif self.cfg.use_weighted_atan_rewards:
        #     rewards["height_reward"] = 0.5 + (1.0 / torch.pi) * torch.atan(1.0 / (1.0 + mean_cloth_height))
        #     rewards["spread_reward"] = 0.5 + (1.0 / torch.pi) * torch.atan(pairwise_sum / count)
        #     rewards["corner_x_reward"] = 0.5 + (1.0 / torch.pi) * torch.atan(free_x.mean(-1))
        #     rewards["direction_reward"] = 0.5 + (1.0 / torch.pi) * torch.atan(free_x.mean(-1) - grasped_x.mean(-1))
        # elif self.cfg.use_weighted_exp_rewards:
        #     rewards["height_reward"] = torch.exp(1.0 / (1.0 + mean_cloth_height))
        #     rewards["spread_reward"] = torch.exp(pairwise_sum / count)
        #     rewards["corner_x_reward"] = torch.exp(free_x.mean(-1))
        #     rewards["direction_reward"] = torch.exp(free_x.mean(-1) - grasped_x.mean(-1))

        rewards["action_penalty"] = torch.zeros(self.num_envs, device=self.device)
        for robot_key in self.action_penalties.keys():
            rewards["action_penalty"] += self.action_penalties[robot_key].mean(0) * self._rewards["action_penalty"].scale

        return rewards

    def _get_rewards(self) -> torch.Tensor:
        """
        Calls compute_rewards(), assigns values to self._rewards, sums total,
        applies invalidation and logging, returns total reward.
        """
        # Compute raw reward terms
        reward_dict = self.compute_rewards()

        # Assign values back to self._rewards
        for name, value in reward_dict.items():
            self._rewards[name].value = value

        total_reward = torch.zeros(self.num_envs, device=self.device)

        for reward_key in self._rewards.keys():
            if self._rewards[reward_key].use: # If the reward is enabled
                if 'penalty' in reward_key:
                    total_reward -= self._rewards[reward_key].value / 1
                else:
                    total_reward += self._rewards[reward_key].value / 1

        # If the time step is not last, divide the total reward
        if (self.episode_length_buf != self.cfg.num_updates_per_episode).any():
            total_reward[self.episode_length_buf != self.cfg.num_updates_per_episode] *= 0.0

        self.extras["log"] = {n: v.value.mean() for n, v in self._rewards.items()}

        ### If EE distances are bad in any time step, set that env's reward to -1
        invalid_env = (self.abs_pos[:,:,2] < 0.3).any(0)
        invalid_env = torch.bitwise_or(
            invalid_env, 
            ((self.abs_pos[:,:,1].abs() - self.scene.env_origins[:,1]) > (self.cfg.box_side/2)).any(0))
        total_reward[invalid_env] -= 5

        self.print_rewards(reward_dict=self._rewards, total_reward=total_reward)

        return total_reward

    def print_rewards(self, reward_dict=None, total_reward=None):
        """
        Prints the reward terms in a clear, formatted way.
        
        Parameters:
            - reward_dict (dict): Reward terms computed in compute_rewards.
            - total_reward (torch.Tensor): The total reward for the episode or timestep.
        """

        print("---- Rewards Summary ----")
        
        for reward_key in reward_dict.keys():
            reward_name = reward_key
            reward_value = reward_dict[reward_key].value
            # Format each reward with its name and value
            print(f"{reward_name}: \t {reward_value.mean().item():.4f}")
        
        if total_reward is not None:
            print(f"\nTotal Reward: \t {total_reward.mean().item():.4f}")
        
        print("-------------------------")

    def _reset_idx(self, env_ids: torch.Tensor | None):
        """Reset the state of multiple robots and environment objects properly."""
        super()._reset_idx(env_ids)

        ### Plot both target and actual joint trajectories for comparison
        # plt.cla()
        # plt.clf()
        # plt.plot(self.target_joints_buffer[:,0,:].cpu().numpy()) # just for env_0 and 4 joints
        # plt.gca().set_prop_cycle(None)
        # plt.plot(self.actual_joints_buffer[:,0,:].cpu().numpy(), '--') # just for env_0 and 4 joints
        # plt.savefig('./logs/snapshots/joint_comparison.jpg')

        ### Calculate convex hull
        # Get the cloth positions for the current environment
        # cloth_nodes = self._cloth_plain.root_physx_view.get_positions().reshape(self.num_envs, -1, 3)
        # current_area = []
        # for env_idx in env_ids:
        #     # Calculate the convex hull of the cloth nodes
        #     try:
        #         current_hull = ConvexHull(cloth_nodes[env_idx][:,0:2].cpu().numpy())
        #         current_area.append(current_hull.volume)
        #         print(f"Current convex hull area for env {env_idx}: {current_area[-1]}")
        #     except Exception as e:
        #         print(f"Error calculating convex hull for env {env_idx}: {e}")
        #         current_area.append(0.0)

        self.action_steps[env_ids] = 0

        if self.default_states is None:
            try:
                for key in self.robots.keys():
                    self.robot_entity_cfg[key].resolve(self.scene)
            except Exception:
                pass

        # self.sim.pause()
        for robot_key in self.robots.keys():  # Loop through all robots dynamically
            robot = self.robots[robot_key]
            joint_pos = robot.data.default_joint_pos[env_ids].clone()
            joint_vel = robot.data.default_joint_vel[env_ids].clone()
            robot.set_joint_position_target(joint_pos, env_ids=env_ids)
            robot.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)

        # # Create joints
        # if not self.joints_created:
        #     self.create_joints(env_ids)
        #     self.joints_created = True

        if self.default_states is not None:
            self._cloth_plain.write_nodal_state_to_sim(self.default_states['_cloth_plain'])

            self._handle_1.write_root_state_to_sim(
                self.default_states['_handle_1'][env_ids], env_ids=env_ids)

        if 0: # self.reset_count % 100 == 0:
            random_scale = torch.zeros((self.num_envs, 3), device=self.device)
            random_scale[:, 0] = 1.0
            random_scale[:, 1] = 1.0 # torch.rand(self.num_envs) * (1.0 - 0.8) + 0.8
            random_scale[:, 2] = 1.0
            self.change_cloth_scale(random_scale, env_ids=env_ids)
            self.sim.reset()
            self.default_states = None

        if self.default_states is None:
            try:
                for key in self.robots.keys():
                    self.robot_entity_cfg[key].resolve(self.scene)
            except Exception:
                pass
            self.default_states = {}
            self.default_states['_cloth_plain'] = self._cloth_plain.data.nodal_state_w.clone()
            self.default_states['_cloth_plain'][:,:,3:] = 0.0
            self.default_states['_handle_1'] = self._handle_1.data.root_state_w.clone()
            self.default_states['_handle_1'][:,7:] = 0.0
            self._initial_corners = torch.stack((
                self.default_states['_handle_1'][:,0:3],
            ), axis=1)

        self.sim.play()

        H  = self.cfg.decimation
        # self.prev_corners_buf = torch.zeros((H, self.num_envs, 6), device=self.device)
        # self.prev_abs_buf = torch.zeros((H, self.num_envs, 2), device=self.device)

        self.reset_count += 1

    def change_cloth_scale(self,
                           scales: torch.Tensor,
                           env_ids: torch.Tensor | None = None):

        for idx in env_ids.tolist():
            prim_cloth_path = f"/World/envs/env_{idx}/cuboid"
            prim_cloth = get_prim_at_path(prim_cloth_path)
            sx, sy, sz = scales[idx].tolist()
            prim_cloth.GetAttribute("xformOp:scale").Set(Gf.Vec3f(sx, sy, sz))
            prim_cloth.GetAttribute('xformOp:translate').Set(
                Gf.Vec3f(0.0, 0.0, 0.356 + ((1 - scales[idx][1]) * 0.356).item()))

    def create_joints(self, env_ids):
        for idx in env_ids:
            hanger_1_path = Sdf.Path("/World/envs/env_" + str(int(idx)) + "/cuboid/coat_hanger")
            panda_1_finger_path = Sdf.Path("/World/envs/env_" + str(int(idx)) + "/Robot1/panda_leftfinger")
            joint_1_path = panda_1_finger_path.AppendElementString("fixedJoint")
            fixedJoint_1 = UsdPhysics.FixedJoint.Define(self.scene.stage, joint_1_path)
            fixedJoint_1.CreateBody0Rel().SetTargets([panda_1_finger_path])
            fixedJoint_1.CreateBody1Rel().SetTargets([hanger_1_path])
            fixedJoint_1.CreateLocalPos0Attr().Set(Gf.Vec3f(0.0,0.0,0.2))
            fixedJoint_1.CreateLocalRot0Attr().Set(Gf.Quatf(0.5, -0.5, 0.5, -0.5))
            fixedJoint_1.CreateLocalPos1Attr().Set(Gf.Vec3f(0,0,0))
            fixedJoint_1.CreateLocalRot1Attr().Set(Gf.Quatf(1,0,0,0))

        self.joints_created = True

    def _get_observations(self) -> torch.Tensor:
        """
        Return a single policy observation per env:
        - last H actions
        - last H corner-trajectories
        """

        observations = []  # List to store tensors for all robots

        for robot_key in self.robots.keys(): # Loop through all robots dynamically
            robot = self.robots[robot_key]

            # ✅ Normalize DOF positions separately for each robot
            dof_pos_scaled = (2.0 * (robot.data.joint_pos - self.robot_dof_lower_limits[robot_key])
                / (self.robot_dof_upper_limits[robot_key] - self.robot_dof_lower_limits[robot_key])
                - 1.0)

            # ✅ Concatenate observations for the current robot
            obs = torch.cat((dof_pos_scaled, robot.data.joint_vel * self.cfg.dof_velocity_scale
                ), dim=-1)

            # ✅ Append the observation tensor to the list
            observations.append(obs)

        joint_obs = torch.cat(observations, axis=-1)

        act_hist = self.prev_abs_buf.permute(1,0,2)
        act_hist = act_hist[:,::48].reshape(self.num_envs, -1)

        self.corner_traj = self.prev_corners_buf.permute(1,0,2)
        corner_traj_obs = self.corner_traj[:,::48].reshape(self.num_envs, -1)

        if self.cfg.disable_init_motion:
            corner_traj_obs *= 0.0
            act_hist *= 0.0

        box_obs = torch.zeros((self.num_envs, 2), device=self.device)
        box_obs[:,0] = self.cfg.box_x_center
        box_obs[:,1] = self.cfg.box_side

        current_corner_obs = self._get_corners()
        current_corner_obs = current_corner_obs.reshape(self.num_envs, -1)

        ### Either pass the action history and corner trajectory observations
        # observations = torch.cat((act_hist, corner_traj_obs), dim=-1)
        ### Or action history and cloth lengths
        # observations = torch.cat((joint_obs, self.cloth_lengths.unsqueeze(-1)), dim=-1)
        ### Or current absolute pose and previous actions and corner trajectories
        observations = torch.cat((joint_obs, current_corner_obs, box_obs), dim=-1)

        return {"policy": observations}
