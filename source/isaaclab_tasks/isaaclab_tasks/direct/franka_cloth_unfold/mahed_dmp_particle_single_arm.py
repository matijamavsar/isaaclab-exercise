# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
from __future__ import annotations

import torch
import numpy as np
from pathlib import Path

import os

from isaacsim.core.utils.stage import get_current_stage
from isaacsim.core.utils.torch.transformations import tf_combine, tf_inverse, tf_vector
from pxr import UsdGeom, UsdShade, Sdf, Gf, PhysxSchema, Usd
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
from pxr import UsdPhysics, UsdGeom, Gf, Sdf, UsdGeom, UsdShade, UsdLux
import omni.usd
import omni.kit.commands
from omni.physx.scripts import physicsUtils, particleUtils
from isaaclab.controllers import DifferentialIKController, DifferentialIKControllerCfg
from isaacsim.core.utils.prims import get_prim_at_path
import torch.nn.functional as F
from isaaclab_assets import FRANKA_PANDA_HIGH_PD_CFG, FRANKA_PANDA_CFG
from torchvision.utils import save_image
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull
import omni.usd
from .dmp_integrator import BatchDMPIntegrator
from .min_jerk_traj import generate_minimum_jerk
import carb
""" Run this training using
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py --task DMP-Based-Particle-Randomized-Position --num_envs 64 --max_iterations 16000 --headless --enable_cameras

Run inference using
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/play.py --task DMP-Based-Particle-Randomized-Position --num_envs 64 --load_run RUN --headless --enable_cameras 


"""
def ensure_contact_api(root_path: str):
    stage = omni.usd.get_context().get_stage()
    root = stage.GetPrimAtPath(root_path)
    if not root: return
    for prim in Usd.PrimRange(root):
        if (prim.IsA(UsdGeom.Xform) or prim.IsA(UsdGeom.Mesh)) and \
           not prim.HasAPI(PhysxSchema.PhysxContactReportAPI):
            PhysxSchema.PhysxContactReportAPI.Apply(prim)



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
class FrankaDMPClothPlaceEnvCfg(DirectRLEnvCfg):
    # env
    episode_length_s = 3
    max_episode_length = episode_length_s*120
    num_updates_per_episode = 1
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
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=1, env_spacing=2.0, replicate_physics=False)

    cloth = ParticleClothObjectCfg(
        prim_path="/World/envs/env_.*/Cloth",
        init_state=ParticleClothObjectCfg.InitialStateCfg(pos=(0.0, 0.1, 0.41), rot=(0.5, 0.5, 0.5, 0.5)),
        spawn=sim_utils.UsdFileCfg(
            # usd_path="/workspace/isaaclab/source/isaaclab_tasks/isaaclab_tasks/sim/particle_cloth_one_robot.usd",
            # usd_path="/workspace/isaaclab/source/isaaclab_tasks/isaaclab_tasks/sim/particle_cloth_two_pinch_half_width_extra_handle.usd",
            usd_path="/workspace/isaaclab/source/isaaclab_tasks/isaaclab_tasks/sim/particle_cloth_two_pinch_half_width_box_bendy.usd",
            scale=(1.0, 1.0 , 1.0),
        ),
            )
    if enable_camera_recording:
        camera = CameraCfg(
            prim_path="/World/Camera",
            offset=CameraCfg.OffsetCfg(pos=(-5.0, -5.0, 3.0), rot=( 0.9020, -0.0828,  0.2000,  0.3736), convention="world"),
            data_types=["rgb"],
            spawn=sim_utils.PinholeCameraCfg(
                focal_length=24.0, focus_distance=400.0, horizontal_aperture=20.955, clipping_range=(0.01, 80.0)
            ),
            width=1920,
            height=1080,
        )


    # cloth_plain = ParticleClothObjectCfg(
    #     prim_path="/World/envs/env_.*/Cloth",
    #     init_state=ParticleClothObjectCfg.InitialStateCfg(pos=(0, 0, 0), rot=(1, 0, 0, 0)),
    #     spawn=None,
    # )

    handle_1 = RigidObjectCfg(
        prim_path="/World/envs/env_.*/Cloth/RightCube",
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0, +0.0, 0), rot=(1, 0, 0, 0)),
        spawn=None,
    )

    handle_2 = RigidObjectCfg(
        prim_path="/World/envs/env_.*/Cloth/LeftCube",
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0, 0, 0), rot=(1, 0, 0, 0)),
        spawn=None,
    )
    ### Robots
    robot_1 = FRANKA_PANDA_HIGH_PD_CFG.replace(
        spawn=FRANKA_PANDA_HIGH_PD_CFG.spawn.replace(
        activate_contact_sensors=True   # ✅ this applies PhysxContactReportAPI to all rigid bodies
    ),

        prim_path="/World/envs/env_.*/Robot1",
        init_state=ArticulationCfg.InitialStateCfg(
            # line grasp defautl
            # joint_pos={
            #     "panda_joint1": -0.0039,
            #     "panda_joint2": 0.0110,
            #     "panda_joint3":  -0.0030,
            #     "panda_joint4": -1.4823,
            #     "panda_joint5":  0.0031,
            #     "panda_joint6":  1.4909,
            #     "panda_joint7":  0.7877,
            #     # "panda_joint7":  0.0,
            #     "panda_finger_joint.*": 0.0,
            # },
            # joint_pos={
            #     "panda_joint1": -0.0311,
            #     "panda_joint2": -0.5669,
            #     "panda_joint3":  0.0147,
            #     "panda_joint4": -1.5801,
            #     "panda_joint5":  0.0091,
            #     "panda_joint6":  1.0138,
            #     "panda_joint7":  0.7715,
            #     "panda_finger_joint.*": 0.005,
            # },
            # box mode 0 [0.4,0.0.7]
            # joint_pos={
            #     "panda_joint1": -0.0775,
            #     "panda_joint2": -0.2247,
            #     "panda_joint3":  0.0590,
            #     "panda_joint4": -1.3154,
            #     "panda_joint5":  0.0132,
            #     "panda_joint6":  1.0915,
            #     "panda_joint7":  0.77734,
            #     "panda_finger_joint.*": 0.0,
            # },
            joint_pos={
                "panda_joint1": -0.0574,
                "panda_joint2": -0.1658,
                "panda_joint3":  0.0457,
                "panda_joint4": -1.1837,
                "panda_joint5":  0.0084,
                "panda_joint6":  1.0149,
                "panda_joint7":  0.7793,
                "panda_finger_joint.*": 0.0,
            },
            pos=(0.0, -0.3, 0.03), # TODO changed this
            rot=(0.7071, 0, 0, 0.7071),
            # rot=(1.0, 0.0, 0.0, 0.0),
        ),)
    
    box = RigidObjectCfg(
        prim_path="/World/envs/env_.*/Box",
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.1, 0.185), rot=(0.707, 0, 0, 0.707)),
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"/workspace/isaaclab/source/isaaclab_tasks/isaaclab_tasks/sim/Collected_small_KLT/small_KLT_.usd",
            scale=(3.1, 1.0, 2.5),
            mass_props=sim_utils.MassPropertiesCfg(mass=30.01),
            # visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 1.0, 1.0), metallic=0.6),
            activate_contact_sensors=True
        ),
    )


    robot_box_contacts = ContactSensorCfg(
        prim_path="/World/envs/env_.*/Robot1/(panda_link[0-9]+|panda_hand|panda_leftfinger|panda_rightfinger)",
        update_period=0.0,
        history_length=1,
        debug_vis=False,
        # filter_prim_paths_expr=["/World/envs/env_.*"],
    )
    # ground plane
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",
        collision_group=-1,
        # physics_material=sim_utils.RigidBodyMaterialCfg(
        #     friction_combine_mode="multiply",
        #     restitution_combine_mode="multiply",
        #     static_friction=1.0,
        #     dynamic_friction=1.0,
        # ),
        debug_vis=False,
    )

    action_scale = 15 # default is 7.5
    dof_velocity_scale = 0.1
    filter_kernel_size = 7
    use_dynamic_rewards = False

    rewards = {
        "spread_reward": 
            Reward(False, 1.0),
        "height_reward": 
            Reward(True, 1.0),
        "corner_x_reward": 
            Reward(False, 10.0),
        "direction_reward": 
            Reward(False, 15.0),
        "endspeed_reward": 
            Reward(False, 1.0),
        "action_penalty": 
            Reward(True, 1e-3),
        "folded_reward": 
            Reward(False, 5.0),
        "horizontal_stretch_reward":
            Reward(True, 1.0),
        "x_mid_box_reward":
            Reward(False, 1.0),
        "y_mid_box_reward":
            Reward(True, 1.0),
        "contact_penalty":
            Reward(True, 1.0),
        "neighboring_pairs_reward":
            Reward(True, 1.0),
        "general_fold_reward":
            Reward(True, 4.0),
        "success_fold_reward":
            Reward(True, 100.0),
    }

    # dataset logging
    enable_dataset_logging = True
    dataset_root = "./logs/datasets"
    dataset_format = "npz"   # 'npz' or 'pt'

    # randomize box positions
    randomize_box = False
    # box_position_range = (1.3, 2.6)

    # randomize fold lengths
    randomize_fold_length = False
    fold_length_range = (0.22, 0.33)  # min, max
    default_fold_length = 0.33

def make_circle_path_torch(center=(0, 0, 3), radius=10, num_points=720, device='cpu'):
    cx, cy, cz = center
    angles = torch.linspace(torch.pi, torch.pi + 2*torch.pi, num_points, device=device, requires_grad=False)
    xs = cx + radius * torch.cos(angles)
    ys = cy + radius * torch.sin(angles)
    zs = torch.full_like(xs, cz)
    return torch.stack([xs, ys, zs], dim=1)

class FrankaDMPClothPlaceEnv(DirectRLEnv):
    # pre-physics step calls
    #   |-- _pre_physics_step(action)
    #   |-- _apply_action()
    # post-physics step calls
    #   |-- _get_dones()
    #   |-- _get_rewards()
    #   |-- _reset_idx(env_ids)
    #   |-- _get_observations()

    cfg: FrankaDMPClothPlaceEnvCfg

    def __init__(self, cfg: FrankaDMPClothPlaceEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        # Set the seed
        self.seed(self.cfg.seed)

        # Setup video recording
        now = datetime.now().strftime("%Y%m%d_%H%M%S")
        if self.cfg.enable_camera_recording:
            self.video_folder = os.path.join("./logs/videos", now)
            os.makedirs(self.video_folder, exist_ok=True) 

        # Define the number of robots dynamically
        self.num_robots = 1
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
        # self.prev_joints = {}  # Store previous joint velocities
        self.one_euro_filter = OneEuroFilter(beta=0.002, min_cutoff=1.5, d_cutoff=1.0, dt=self.physics_dt)
        self.filtered_dof_targets = {}  # Store smoothed action buffers
        self.robot_dof_targets = {}  # Store DOF targets
        self.gripper_actions = torch.zeros((self.num_envs, 2), device=self.device)
        self.ee_distances = torch.zeros((self.cfg.max_episode_length, self.num_envs), device=self.device)
        self.z_abs_pos = torch.zeros((self.cfg.max_episode_length, self.num_envs), device=self.device)
        # self.action_penalties = {'robot_1': torch.zeros((self.cfg.max_episode_length, self.num_envs), device=self.device),
        #                          'robot_2': torch.zeros((self.cfg.max_episode_length, self.num_envs), device=self.device)}
        self.action_penalties = {'robot_1': torch.zeros((self.cfg.max_episode_length, self.num_envs), device=self.device)}

        H  = self.cfg.decimation
        self.prev_corners_buf = torch.zeros((H, self.num_envs, 6), device=self.device)
        self.prev_abs_buf = torch.zeros((H, self.num_envs, 2), device=self.device)
        self.target_joints_buffer = torch.zeros((H, self.num_envs, 7), device=self.device)
        self.actual_joints_buffer = torch.zeros((H, self.num_envs, 7), device=self.device)

        self.robot_dof_lower_limits = {}  # Store lower joint limits
        self.robot_dof_upper_limits = {}  # Store upper joint limits
        self.robot_dof_speed_scales = {}  # Store joint speed scales

        # Generate minimum jerk trajectory for probing
        abs_points = [
            [0.0, 0.0, 0.81],
            [-0.3, 0.0, 0.81],
            [0.2, 0.0, 0.81],
            [0.0, 0.0, 0.81],
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


        for i in range(1, self.num_robots + 1):
            robot_key = f"robot_{i}"

            # Instantiate robot and store in dictionary
            self.robots[robot_key] = getattr(self, "_" + robot_key)

            # Initialize buffers
            num_joints = self.robots[robot_key].num_joints

            # self.prev_joints[robot_key] = torch.zeros((self.cfg.filter_kernel_size, 
            #                                            self.num_envs, 7), device=self.device)
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
            # finger_joints = ["panda_finger_joint1"]
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

        self.dmp_integrator = BatchDMPIntegrator(N_basis=25, dof=2, device=self.device)
        self.corner_dmp_integrator = BatchDMPIntegrator(N_basis=25, dof=12, device=self.device)
        self.corner_traj = torch.zeros((self.num_envs, self.cfg.decimation, 6), device=self.device)
        self.corner_dmp_weights = torch.zeros((self.num_envs, 12, 25), device=self.device)
        self.corner_y0 = torch.zeros((self.num_envs, 12), device=self.device)
        self.corner_goal = torch.zeros((self.num_envs, 12), device=self.device)

        self.lp1 = {}
        self.lp1['robot_1'] = BiquadLP2Batch(self.scene.cfg.num_envs, 7, fc=1.0, dt=self.physics_dt, device=self.device)
        # self.lp1['robot_2'] = BiquadLP2Batch(self.scene.cfg.num_envs, 7, fc=1.0, dt=self.physics_dt, device=self.device)
                # --- Dataset directory & buffers ---
        self.run_timestamp = now
        self.dataset_dir = os.path.join(self.cfg.dataset_root, self.run_timestamp)
        os.makedirs(self.dataset_dir, exist_ok=True)

        # Per-episode counters (per env)
        self._episode_counters = torch.zeros(self.num_envs, dtype=torch.long)

        # Pre-allocate trajectory buffers (length = max_episode_length)
        T = self.cfg.max_episode_length
        self._traj_q = {}     # per robot: [T, N, DOF]
        self._traj_ee = {}    # per robot: [T, N, 7]  (x,y,z, qw,qx,qy,qz)
        for robot_key in range(1, self.num_robots + 1):
            rk = f"robot_{robot_key}"
            nj = self.robots[rk].num_joints
            # keep on device; we’ll move to CPU on save
            self._traj_q[rk]  = torch.zeros((T, self.num_envs, nj), device=self.device)
            self._traj_ee[rk] = torch.zeros((T, self.num_envs, 7),  device=self.device)
        # DMP parameters buffer (whatever dimension your action space is)
        self._traj_dmp = torch.zeros((T, self.num_envs, self.cfg.action_space), device=self.device)

        self._box_markers = None
        self.contact_forces = torch.zeros((self.num_envs, self.cfg.max_episode_length), device=self.device)

        # fold lengths 
        self._fold_pair_idx = None                 # [N, P_max, 2] (flat node indices)
        self._fold_pair_valid_counts = None        # [N]
        self._fold_params = {"include_edges": True, "anchor_on_top": False}
        # scalar default expanded to N
        self._default_fold_length = torch.full((self.num_envs,), 0.33, device=self.device)

        self._fold_lengths = self._default_fold_length.clone()

        # OR tie it to your box length, etc.
        # self._default_fold_length = self.box_lengths.clone() * 1.32
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
        # img_rgb = self._camera.data.output['rgb'][0]
        img_rgb = self.capture_clean_frame()

        # 2) Convert to BGR (OpenCV expects BGR):
        img_bgr = cv2.cvtColor(img_rgb.cpu().numpy(), cv2.COLOR_RGB2BGR)
        # 3) Write to the open VideoWriter:
        self.video_writer.write(img_bgr)



    def _setup_scene(self):
        # ✅ Create robots
        self.scene.clone_environments(copy_from_source=True)
        # self._activate_box_marker_visuals()

        self._robot_1 = Articulation(self.cfg.robot_1)
        # self.cloth_lengths = torch.zeros(self.scene.cfg.num_envs, device=self.device)

        # fill the cloth_lengths with default value of 0.66
        self.cloth_lengths = torch.full((self.scene.cfg.num_envs,), 0.66, device=self.device)


        self.scene.articulations["robot_1"] = self._robot_1

        # ✅ Configure terrain
        self.cfg.terrain.num_envs = self.scene.cfg.num_envs
        self.cfg.terrain.env_spacing = self.scene.cfg.env_spacing
        self._terrain = self.cfg.terrain.class_type(self.cfg.terrain)

        self._cloth = self.cfg.cloth.class_type(self.cfg.cloth)
        # self._cloth_plain = self.cfg.cloth_plain.class_type(self.cfg.cloth_plain)
        self._handle_1 = self.cfg.handle_1.class_type(self.cfg.handle_1)
        self._handle_2 = self.cfg.handle_2.class_type(self.cfg.handle_2)
        self._box = self.cfg.box.class_type(self.cfg.box)
        self.box_lengths = torch.zeros(self.scene.cfg.num_envs, device=self.device)

        self._contact_robot_box = self.cfg.robot_box_contacts.class_type(self.cfg.robot_box_contacts)
        self.scene.sensors["robot_box_contacts"] = self._contact_robot_box  # auto-updated post-physics


        # ✅ Clone environments

        # for env_idx in range(self.scene.cfg.num_envs):
        #     # Get the Cloth prim for the current environment
        #     cloth_prim_path = f"/World/envs/env_{env_idx}/Cloth"
        #     cloth_prim = get_prim_at_path(cloth_prim_path)

        #     # Generate random scale factors for x, y, z axes
        #     random_scale = torch.tensor([
        #         1.0,
        #         torch.rand(1).item() * (1.0 - 0.6) + 0.6,  # Scale y-axis
        #         1.0,
        #     ])

        #     # Apply the random scale to the Cloth prim
        #     cloth_prim.GetAttribute("xformOp:scale").Set(Gf.Vec3f(*random_scale.tolist()))
        #     cloth_prim.GetAttribute('xformOp:translate').Set(
        #         Gf.Vec3f(0.0, 0.1, 0.39 + ((0.66 - (random_scale[1]*0.66))/2.0).item()))
        #     self.cloth_lengths[env_idx] = random_scale[1].item() * 0.66

        #     print(" Cloth scale for env", env_idx, ":", random_scale.tolist())
        #     print(" cloth h change", env_idx, ":", ((0.66 - (random_scale[1]*0.66))/2.0), "\n \n \n \n")

        # Box randomization in y direction
        for env_idx in range(self.scene.cfg.num_envs):
            box_prim_path = f"/World/envs/env_{env_idx}/Box"
            box_prim = get_prim_at_path(box_prim_path)

            # Generate random position for y axis
            if self.cfg.randomize_box:
                random_scale_y = torch.rand(1).item() * (1.3) + (1.3)  # Position y-axis
                random_scale = torch.tensor([
                    3.1,
                    random_scale_y,
                    2.5,
                ])
            else:
                random_scale = torch.tensor([
                    3.1,
                    2.6,
                    2.5,
                ])
            # Apply the random scale to the Box prim
            box_prim.GetAttribute("xformOp:scale").Set(Gf.Vec3f(*random_scale.tolist()))
            # box_prim.GetAttribute('xformOp:translate').Set(
            self.box_lengths[env_idx] = random_scale[1].item() * 0.25
        

            
        #     # TODO: randomize also other parameters (drag, friction etc)

        if self.cfg.enable_camera_recording:
            self._camera = self.cfg.camera.class_type(self.cfg.camera)

        # ✅ Add lights (optional)
        # light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        # light_cfg.func("/World/Light", light_cfg)

        # # ✅ Get the stage
        # stage = omni.usd.get_context().get_stage()

        # ✅ Hide the environment
        # environment_prim_path = "/World/ground/Environment"
        # environment_prim = stage.GetPrimAtPath(environment_prim_path)
        # # if environment_prim:
        #     UsdGeom.Imageable(environment_prim).GetVisibilityAttr().Set(UsdGeom.Tokens.invisible)


        # self.setup_paper_render()
        self.setup_clean_stage(renderer="HydraStorm")  # simplest, no noise/ghosting
        # self.no_ghost_mode(aggressive=True)
        # self.set_ground_as_grid(cell_size_m=0.12, line_px=2)
        # self.set_cloth_texture("/workspace/isaaclab/source/isaaclab_tasks/isaaclab_tasks/sim/cloth_texture/cloth_tex0.jpeg",
        #                     # fallback_tint=(0.9, 0.9, 0.9),
        #                     roughness=0.6)
        self.set_cloth_texture("/workspace/isaaclab/source/isaaclab_tasks/isaaclab_tasks/sim/cloth_texture/cloth_texture_1.jpeg",
                            # fallback_tint=(0.9, 0.9, 0.9),
                            roughness=0.6)
        # self.set_cloth_color((0.65, 0.20, 0.20))

        self._box_markers = self._get_markers()
        self._create_handle_rods(radius=0.006)
        # print("box markers:", self._box_markers)
        # import omni
        # from pxr import Sdf, UsdPhysics, PhysxSchema
        # stage = omni.usd.get_context().get_stage()
        # material_path = '/World/envs/env_0/Box/Visuals/Looks/FOF_Mesh_LabelsSG/Shader'

        # prim = stage.GetPrimAtPath(material_path)
        # texture = prim.GetAttribute('inputs:diffuse_texture')
        # texture.Set('./yellow_color_image.png')

        # print(prim.GetPropertyNames())        
        # # self.sim.pause()



    def _get_absolute_pose(self):
        absolute_pose = torch.zeros((self.num_envs, 7), device=self.device)
        # ee_pose_1_w = self._robot_1.data.body_state_w[
        #     :, self.robot_entity_cfg['robot_1'].body_ids[0], 0:7
        # ]
        ee_pose_1_w = self.robots['robot_1'].data.body_state_w[:, self.left_finger_link_idx['robot_1'], 0:7]
        absolute_pose[:, 0:3] = ee_pose_1_w[:, 0:3]
        absolute_pose[:, 3:7] = ee_pose_1_w[:, 3:7]
        return absolute_pose

    def _get_ee_robot_base_pose(self):
        relative_pose = torch.zeros((self.num_envs, 7), device=self.device)
        # ee_pose_1_w = self._robot_1.data.body_state_w[
        #     :, self.robot_entity_cfg['robot_1'].body_ids[0], 0:7
        # ]
        # root_pose_w = self._robot_1.data.root_state_w[:, 0:7]
        # left finger pose
        ee_pose_1_w = self.robots['robot_1'].data.body_state_w[
            :, self.left_finger_link_idx['robot_1'], 0:7
        ]
        root_pose_w = self.robots['robot_1'].data.root_state_w[:, 0:7]
        # Convert WORLD -> ROBOT BASE frame
        ee_pos_b, ee_quat_b = subtract_frame_transforms(
            root_pose_w[:, 0:3], root_pose_w[:, 3:7],
            ee_pose_1_w[:, 0:3],   ee_pose_1_w[:, 3:7]
        )
        relative_pose[:, 0:3] = ee_pos_b
        relative_pose[:, 3:7] = ee_quat_b
        return relative_pose

    def _get_ee_env_base_pose(self):
        # relative_pose = torch.zeros((self.num_envs, 7), device=self.device)
        # ee_pose_1_w = self._robot_1.data.body_state_w[
        #     :, self.robot_entity_cfg['robot_1'].body_ids[0], 0:7
        # ]
        # relative pose for left finger
        relative_pose = torch.zeros((self.num_envs, 7), device=self.device)
        ee_pose_1_w = self.robots['robot_1'].data.body_state_w[
            :, self.left_finger_link_idx['robot_1'], 0:7
        ]
        env_origins = self.scene.env_origins

        # Convert WORLD -> ENV frame
        ee_pos_b, ee_quat_b = subtract_frame_transforms(
            env_origins, torch.tensor([1, 0, 0, 0], device=self.device).repeat(self.num_envs, 1),
            ee_pose_1_w[:, 0:3],   ee_pose_1_w[:, 3:7]
        )

        relative_pose[:, 0:3] = ee_pos_b
        relative_pose[:, 3:7] = ee_quat_b
        return relative_pose
    # pre-physics step calls
    def _pre_physics_step(self, actions: torch.Tensor):

        # print("Pre-physics step and action:", actions)
        dmp_parameters = actions.clone()
        dmp_parameters[:,2] = dmp_parameters[:,2].clamp(-1.0, 1.0) # TODO: you can limit X here
        dmp_parameters[:,3] = dmp_parameters[:,3].clamp(0.0, 1.0) # TODO: you can limit Z here

        self.actions = dmp_parameters

        current_absolute_pose = self._get_absolute_pose()
        current_absolute_pose[:,0:3] -= self.scene.env_origins
        dmp_parameters[:,0] = current_absolute_pose[:,0]
        dmp_parameters[:,1] = current_absolute_pose[:,2]

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
    
    def _apply_action(self):
        """Apply joint position targets for multiple robots dynamically."""
        # TODO

      
        if self.action_steps[0] == 0:
            for robot_key in self.robots.keys():
                q0 = self.robots[robot_key].data.joint_pos[:, :7]
                self.lp1[robot_key].reset(q0)

        t, y, dy, ddy = self.dmp_integrator.step()
        # print("actions", self.actions[0])
        # print("Absolute pose:", self._get_absolute_pose()[:,:3], "relative pose:", self._get_ee_relative_pose()[:,:3])
        self.dmp_integrator.x[t >= self.actions[:,-1]] = 0.1353

        current_absolute_pose = self._get_absolute_pose()
        # self.ee_distances[self.action_steps] = self._get_ee_distance()
        self.z_abs_pos[self.action_steps] = current_absolute_pose[:, 2].clone()

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

                ## Make camera move in a circle
                # self._camera.set_world_poses_from_view(self.circle_camera_path[self.render_count].unsqueeze(0),
                #                                        torch.tensor([0,0,0.5], device=self.device))

                ### Set camera to a fixed position
                ### Good for top view of 4 envs
                # self._camera.set_world_poses_from_view(torch.tensor([0., 0., 6.5], device=self.device).unsqueeze(0),
                #                                        torch.tensor([0., 0., 0.], device=self.device))
                ### Good for watching one env
                self._camera.set_world_poses_from_view(torch.tensor([12.0, 12.0, 5.0], device=self.device).unsqueeze(0),
                                                       torch.tensor([5.0, 3.0, 0.0], device=self.device))
                self.switch_to_storm_for_capture(msaa=16)

                if self.render_count % self.cfg.sim.render_interval == 0:
                    self.pulse_reset_temporal_history()
                    self._capture_and_write_frame()
                self.render_count += 1

            # If we just left a window, close the writer:
            just_left = ((not in_window) and self.video_writer is not None)
            if just_left:
                self.render_count = 0
                self._close_camera_writer()

        if not self.probing_done:
            y = torch.zeros((self.num_envs, 2), device=self.device)
            y[:,0] = self._probing_traj[self.action_steps][:,0]
            y[:,1] = self._probing_traj[self.action_steps][:,2]

            self.inverse_kinematics(y)

            for robot_key in self.robots.keys():
                # ✅ Set joint position target separately for each robot
                if self.cfg.write_joint_state:
                    self.robots[robot_key].write_joint_position_to_sim(
                        self.robot_dof_targets[robot_key])
                else:
                    self.robots[robot_key].set_joint_position_target(
                        self.robot_dof_targets[robot_key])

            # get free current corners, flatten them360---
            corners = self._get_corners()[:, [2, 3]] - self.scene.env_origins.unsqueeze(1)  # (N,2,3)
            corners_flat = corners.reshape(self.num_envs, -1)
            self.prev_corners_buf = self.prev_corners_buf.roll(-1, dims=0)
            self.prev_corners_buf[-1] = corners_flat

            self.prev_abs_buf = self.prev_abs_buf.roll(-1, dims=0)
            self.prev_abs_buf[-1] = current_absolute_pose[:, [0, 2]] - self.scene.env_origins[:, [0, 2]]

            if (self.action_steps == self.cfg.max_episode_length - 1).all():
                self.probing_done = True

        elif (self.action_steps < (self.cfg.max_episode_length - 60)).all():
            # print("dmp start:", self.dmp_integrator.x, "dmp goal:", self.dmp_integrator.goal)
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
                
                self.target_joints_buffer[self.action_steps[0]] = self.robot_dof_targets[robot_key][:,0:7]
                self.actual_joints_buffer[self.action_steps[0]] = self.robots[robot_key].data.joint_pos[:,0:7]
        else:
            target_y = current_absolute_pose[:,0:3] - self.scene.env_origins
            target_y = target_y[:,[0,2]]
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

            # --- log a timestep into dataset buffers ---
        if self.cfg.enable_dataset_logging:
            step_idx = int(self.action_steps[0].item())
            step_idx = max(0, min(step_idx, self.cfg.max_episode_length - 1))

            for rk in self.robots.keys():
                self._traj_q[rk][step_idx] = self.robots[rk].data.joint_pos

                # Get EE pose in WORLD
                body_id = self.robot_entity_cfg[rk].body_ids[0]
                ee_pose_w  = self.robots[rk].data.body_state_w[:, body_id, 0:7]
                root_pose_w = self.robots[rk].data.root_state_w[:, 0:7]

                # Convert WORLD -> ROBOT BASE frame
                ee_pos_b, ee_quat_b = subtract_frame_transforms(
                    root_pose_w[:, 0:3], root_pose_w[:, 3:7],
                    ee_pose_w[:, 0:3],   ee_pose_w[:, 3:7]
                )
                # print("EE pos z in base frame:", ee_pos_b[0,2], "root z:", root_pose_w[0,2])
                # Store base-frame (x,y,z,qw,qx,qy,qz)
                # self._traj_ee[rk][step_idx, :, 0:3] = ee_pos_b
                # self._traj_ee[rk][step_idx, :, 3:7] = ee_quat_b
                # print("ee_pose_w", ee_pose_w.shape)
                self._traj_ee[rk][step_idx, :, 0:7] = self._get_ee_robot_base_pose()[:, 0:7]

            self._traj_dmp[step_idx] = self.actions
        # self.contact_forces[:, self.action_steps[0]] = self._log_robot_box_contacts().squeeze(-1)
        self.contact_forces[:, self.action_steps] = self._log_robot_box_contacts()
        # print("contact forces at cuurent step:", self.contact_forces[:, self.action_steps], " action step:", self.action_steps)
        self._update_handle_rods()  
        self.iteration_step += 1
        self.action_steps += 1

    def inverse_kinematics(self, y):
        current_absolute_pose = self._get_absolute_pose()
        target_absolute_pose = current_absolute_pose.clone() * 0.0
        target_absolute_pose[:,1] = 0.1 # TODO: this sets robot's x at 0.4
        target_absolute_pose[:,0] = y[:,0]
        target_absolute_pose[:,2] = y[:,1]
        target_absolute_pose[:,0:3] += self.scene.env_origins
        # add to the y coordinate a fixed offset
        
        target_absolute_pose[:, 3:] = torch.tensor([0, 1, 0, 0], device=self.device)
        
        # Fixed Y-distance between grippers
        # fixed_rel_pos = torch.tensor([0.0, 0.66, 0.0], device=self.device).repeat(self.num_envs, 1)
        # fixed_rel_quat = torch.tensor([0, 0.0, 0.0, 1], device=self.device).repeat(self.num_envs, 1)
        # relative_pose = torch.cat((fixed_rel_pos, fixed_rel_quat), dim=-1)

        # Transform from global into each robot coordinate system
        # larm_pose, rarm_pose = self._abs_to_arm_poses(target_absolute_pose, relative_pose)

        # root_pose_w = self._robot_2.data.root_state_w[:, 0:7]
        # larm_pos_b, _ = subtract_frame_transforms(
        #     root_pose_w[:, 0:3], root_pose_w[:, 3:7], 
        #     larm_pose[:, 0:3], larm_pose[:, 3:7]
        # )
        root_pose_w = self._robot_1.data.root_state_w[:, 0:7]
        rarm_pos_b, _ = subtract_frame_transforms(
            root_pose_w[:, 0:3], root_pose_w[:, 3:7], 
            target_absolute_pose[:, 0:3], target_absolute_pose[:, 3:7]
        )

        rot_around_x = self.prev_rot_around_x + self.actions[:, 2]
        rot_around_x = rot_around_x.clamp(-0.6, 0.6)

        rarm_goal_quat_b = torch.zeros((self.num_envs, 4), device=self.device)
        rarm_goal_quat_b[:,1] = 1.0

        rarm_pose_b = torch.cat((rarm_pos_b, rarm_goal_quat_b), dim=-1)

        self._ik_controller['robot_1'].set_command(rarm_pose_b)
        self.prev_rot_around_x = rot_around_x.clone()

        for i, robot_key in enumerate(self.robots.keys()):  # Loop through all robots dynamically
            self.robot_dof_targets[robot_key][:,-2:] = self.gripper_actions[:,i].unsqueeze(1).repeat(1,2)
            # print the shape of robot_dof_targets
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

            # self.prev_joints[robot_key] = self.prev_joints[robot_key].roll(-1, dims=0)
            # self.prev_joints[robot_key][-1] = new_targets[:, 0:7]

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

    # def _get_ee_distance(self):
    #     robot_keys = list(self.robots.keys())  # Get robot keys dynamically
    #     ee_1_pos = self.robots[robot_keys[0]].data.body_pos_w[:, self.left_finger_link_idx[robot_keys[0]]]
    #     # ee_2_pos = self.robots[robot_keys[1]].data.body_pos_w[:, self.left_finger_link_idx[robot_keys[1]]]

    #     ee_distance = torch.norm(ee_1_pos - ee_2_pos, dim=-1)

    #     return ee_distance

    def _get_corners(self):
        # cloth_positions = self._cloth_plain.root_physx_view.get_positions().reshape(self.num_envs, -1, 3)
        cloth_positions = self._cloth.root_physx_view.get_positions().reshape(self.num_envs, -1, 3)
        side_len = int(torch.sqrt(torch.tensor(cloth_positions.shape[1])).to(self.device))
        corners = cloth_positions[:, [-side_len, -1, 0, side_len-1], :]
        return corners

    def compute_rewards(self):
        """
        Compute individual reward term values and return a dict of reward_name -> value tensor.
        """
        # Gather data
        corners = self._get_corners()
        corners_relative = corners - self.scene.env_origins.unsqueeze(1)
        # print("Corners ", corners)
        # cloth_positions = self._cloth_plain.root_physx_view.get_positions().reshape(self.num_envs, -1, 3)
        cloth_positions = self._cloth.root_physx_view.get_positions().reshape(self.num_envs, -1, 3)
        mean_cloth_height = cloth_positions[:, :, 2].mean(dim=-1) - self.scene.env_origins[:, 2]
        mean_cloth_y = cloth_positions[:, :, 1].mean(dim=-1) - self.scene.env_origins[:, 1]
        mean_cloth_x = cloth_positions[:, :, 0].mean(dim=-1) - self.scene.env_origins[:, 0]
        # print("Mean cloth y: ", mean_cloth_y)
        # print("Mean cloth x: ", mean_cloth_x)
        # print("Mean cloth height: ", mean_cloth_height)
        # print(cloth_positions[:, :, 2].mean(dim=-1).shape)
        # print(self.scene.env_origins[:, 2].shape)

        # Compute each reward term
        rewards = {}

        # Spread reward
        pairwise_sum = torch.zeros(self.num_envs, device=self.device)
        count = 0.0
        for i in range(4):
            for j in range(i + 1, 4):
                pairwise_sum += torch.norm(corners[:, i] - corners[:, j], dim=-1)
                count += 1.0

        # Corner X and direction rewards
        free_x = corners[:, [2, 3], 0] - self.scene.env_origins[:, 0].unsqueeze(1)
        grasped_x = corners[:, [0, 1], 0] - self.scene.env_origins[:, 0].unsqueeze(1)
        free_y = corners[:, [2, 3], 1] - self.scene.env_origins[:, 1].unsqueeze(1)
        grasped_y = corners[:, [0, 1], 1] - self.scene.env_origins[:, 1].unsqueeze(1)
        # rewards["corner_x_reward"] = (free_x.mean(-1) + grasped_x.mean(-1)) * self._rewards["corner_x_reward"].scale

        # endspeed = self._robot_1.data.joint_vel.abs().sum(-1) + self._robot_2.data.joint_vel.abs().sum(-1)
        endspeed = self._robot_1.data.joint_vel.abs().sum(-1)
        _free_horizontal_distance = torch.norm(corners[:, 2] - corners[:, 3], dim=-1) 
        _grasped_horizontal_distance = torch.norm(corners[:, 0] - corners[:, 1], dim=-1)
        _side1_distance = torch.norm(corners[:, 0] - corners[:, 2], dim=-1)
        _side2_distance = torch.norm(corners[:, 1] - corners[:, 3], dim=-1)
        # p_world, q_world = self.get_marker_world_pose(env_i=0)
        # print("marker @ world:", p_world, q_world)
        markers  = self._box_markers if self._box_markers is not None else self._get_markers()
        _box_mid_y = (markers[0][0][1] + markers[0][1][1]) / 2.0
        _box_mid_x = (markers[0][0][0] + markers[0][3][0]) / 2.0
        # _desired_mean_y_ = 0.0
        # _mean_y_


        _y_corners_mid_box_dist = torch.abs( ((free_y[:,0] + free_y[:,1] + grasped_y[:, 0] + grasped_y[:, 1]) / 4.0 - _box_mid_y ))
        _x_corners_mid_box_dist = torch.abs( ((free_x[:,0] + free_x[:,1] + grasped_x[:, 0] + grasped_x[:, 1]) / 4.0 - _box_mid_x ))

        try:
            fold_pairs_pos, valid_counts = self.get_edge_and_fold_pairs_pos()
            fold_lengths_dist = torch.zeros(self.num_envs, device=self.device) # distance between odd pairs and even pairs
            for env_i, env_fold_pairs in enumerate(fold_pairs_pos):
                env_fold_pairs_clean = env_fold_pairs[:valid_counts[env_i]]
                if env_fold_pairs_clean.size(0) > 1:
                    for i in range(0, len(env_fold_pairs_clean), 2):
                        if i + 2 < len(env_fold_pairs_clean):
                            distance = torch.sum(torch.norm(env_fold_pairs_clean[i] - env_fold_pairs_clean[i + 2], dim=-1))
                            fold_lengths_dist[env_i] += distance
                    for i in range(1, len(env_fold_pairs_clean), 2):
                        if i + 2 < len(env_fold_pairs_clean):
                            distance = torch.sum(torch.norm(env_fold_pairs_clean[i] - env_fold_pairs_clean[i + 2], dim=-1))
                            fold_lengths_dist[env_i] += distance
                fold_lengths_dist[env_i] /= 2.0
                # fold_lengths_dist[env_i] /= valid_counts[env_i]  # normalize by number of pairs

            neighboring_pairs_dist = torch.zeros(self.num_envs, device=self.device)
            neighboring_pairs_dist_cloth_length_diff = torch.zeros(self.num_envs, device=self.device)
            for env_i, env_edge_pairs in enumerate(fold_pairs_pos):
                env_edge_pairs_clean = env_edge_pairs[:valid_counts[env_i]]
                if env_edge_pairs_clean.size(0) > 1:
                    for i in range( len(env_edge_pairs_clean) -1 ):
                        # distance = torch.norm(env_edge_pairs_clean[i] - env_edge_pairs_clean[i + 1])
                        distance = torch.sum(torch.norm(env_edge_pairs_clean[i] - env_edge_pairs_clean[i + 1], dim=-1))
                        # print(f"difference between two pair with i {i} is  {env_edge_pairs_clean[i] - env_edge_pairs_clean[i + 1]} and distance is {distance}")
                        neighboring_pairs_dist[env_i] += distance

                neighboring_pairs_dist[env_i] /= 2.0
                # print(f"env {env_i} and neighboring pairs dist {neighboring_pairs_dist[env_i]}")   
                neighboring_pairs_dist_cloth_length_diff[env_i] = torch.abs(self.cloth_lengths[env_i] - neighboring_pairs_dist[env_i])
                # neighboring_pairs_dist[env_i] /= self._fold_lengths[env_i]  # normalize by initial fold length
            # print(f"fold pair pos {fold_pairs_pos}, valid counts {valid_counts}")
            # print("fold lengths dist:", fold_lengths_dist)
            # print(f"neighboring pairs dist: {neighboring_pairs_dist} neighboring_pairs_dist_cloth_length_diff {neighboring_pairs_dist_cloth_length_diff}" )        
        except Exception as e:
            print("Error in get_edge_and_fold_pairs_pos:", e)
        # print(" _x_corners_mid_box_dist:",  _x_corners_mid_box_dist, "_y_corners_mid_box_dist:", _y_corners_mid_box_dist)

        if not self.cfg.use_weighted_atan_rewards and not self.cfg.use_weighted_exp_rewards:
            rewards["height_reward"] = (1.0 / (0.1 + mean_cloth_height)) * self._rewards["height_reward"].scale
            rewards["height_reward"][mean_cloth_height > 0.4] *= 0.0
            rewards["spread_reward"] = (pairwise_sum / count) * self._rewards["spread_reward"].scale
            rewards["corner_x_reward"] = (free_x.mean(-1)) * self._rewards["corner_x_reward"].scale
            rewards["direction_reward"] = (free_x.mean(-1) - grasped_x.mean(-1)) * self._rewards["direction_reward"].scale
            rewards["endspeed_reward"] = (1 / (0.1 + endspeed)) * self._rewards["endspeed_reward"].scale
            rewards["horizontal_stretch_reward"] = ( (_free_horizontal_distance + _grasped_horizontal_distance) / 2.0) * self._rewards["horizontal_stretch_reward"].scale
            rewards["folded_reward"] = ((1 / (0.1 + _side1_distance)) + (1 / (0.1 + _side2_distance))) * 0.5 * self._rewards["folded_reward"].scale
            rewards["x_mid_box_reward"] = (1.0 / (0.1 + _x_corners_mid_box_dist)) * self._rewards["x_mid_box_reward"].scale
            rewards["y_mid_box_reward"] = (1.0 / (0.1 + _y_corners_mid_box_dist)) * self._rewards["y_mid_box_reward"].scale
            rewards["general_fold_reward"] = (1.0 / (0.1 + fold_lengths_dist)) * self._rewards["general_fold_reward"].scale
            rewards["neighboring_pairs_reward"] = (1.0 / (0.1 + neighboring_pairs_dist_cloth_length_diff)) * self._rewards["neighboring_pairs_reward"].scale
        elif self.cfg.use_weighted_atan_rewards:
            rewards["height_reward"] = 0.5 + (1.0 / torch.pi) * torch.atan(1.0 / (0.1 + mean_cloth_height))
            rewards["spread_reward"] = 0.5 + (1.0 / torch.pi) * torch.atan(pairwise_sum / count)
            rewards["corner_x_reward"] = 0.5 + (1.0 / torch.pi) * torch.atan(free_x.mean(-1))
            rewards["direction_reward"] = 0.5 + (1.0 / torch.pi) * torch.atan(free_x.mean(-1) - grasped_x.mean(-1))
            rewards["horizontal_stretch_reward"] = 0.5 + (1.0 / torch.pi) * torch.atan(
                (_free_horizontal_distance + _grasped_horizontal_distance) / 2.0)
            rewards["folded_reward"] = 0.5 + (1.0 / torch.pi) * torch.atan(
                ((1 / (0.1 + _side1_distance)) + (1 / (0.1 + _side2_distance))) * 0.5)
        elif self.cfg.use_weighted_exp_rewards:
            rewards["height_reward"] = torch.exp(1.0 / (0.1 + mean_cloth_height))
            rewards["spread_reward"] = torch.exp(pairwise_sum / count)
            rewards["corner_x_reward"] = torch.exp(free_x.mean(-1))
            rewards["direction_reward"] = torch.exp(free_x.mean(-1) - grasped_x.mean(-1))
            rewards["horizontal_stretch_reward"] = (torch.exp(-5 * _free_horizontal_distance) + torch.exp(-5 * _grasped_horizontal_distance)) / 2.0
            rewards["folded_reward"] = (torch.exp(_side1_distance) + torch.exp( _side2_distance)) / 2.0

        _general_fold_reward_normalized = (rewards["general_fold_reward"] / (10.0 * self._rewards["general_fold_reward"].scale)).clamp(max=1.0)
        _neighboring_pairs_reward_normalized = (rewards["neighboring_pairs_reward"] / (10.0 * self._rewards["neighboring_pairs_reward"].scale)).clamp(max=1.0)
        print(f"general fold reward: {rewards['general_fold_reward']}, normalized: {_general_fold_reward_normalized}")
        print(f"neighboring pairs reward: {rewards['neighboring_pairs_reward']}, normalized: {_neighboring_pairs_reward_normalized}")
        # "success_fold_reward" is given when both general fold and neighboring pairs rewards are aligned
        _success_fold_reward = (_general_fold_reward_normalized * _neighboring_pairs_reward_normalized)
        _success_fold_reward = torch.where(_success_fold_reward < 0.2, 0.0, _success_fold_reward)
        rewards["success_fold_reward"] = _success_fold_reward * self._rewards["success_fold_reward"].scale

        print(f"success fold reward: {rewards['success_fold_reward']}")
        rewards["action_penalty"] = torch.zeros(self.num_envs, device=self.device)
        for robot_key in self.action_penalties.keys():
            rewards["action_penalty"] += self.action_penalties[robot_key].mean(0) * self._rewards["action_penalty"].scale
        rewards["contact_penalty"] = torch.zeros(self.num_envs, device=self.device)
        # rewards["contact_penalty"] += (self.contact_forces.mean(1) * self._rewards["contact_penalty"].scale)
        top_5_values, _ = torch.topk(self.contact_forces, k=5, dim=1)
        # lowest_5_values, _ = torch.topk(self.contact_forces, k=5, dim=1, largest=False)
        mean_contact_values = self.contact_forces.mean(1, keepdim=True)
        # print("Top contact forces:", top_5_values)
        # print(" mean contact forces:", mean_contact_values)
        top_5_values = top_5_values - mean_contact_values
        rewards["contact_penalty"] += (top_5_values.sum(1) * self._rewards["contact_penalty"].scale)
        # print(" top 5 values after subtracting mean:", top_5_values, " reward contact penalty:", rewards["contact_penalty"], "action penalty:", rewards["action_penalty"])
        # rewards["contact_penalty"] -= (lowest_5_values.sum(1) * self._rewards["contact_penalty"].scale)
        rewards["contact_penalty"] = torch.clamp(rewards["contact_penalty"], min=0.0, max=20.0)
        # print("contact penalty:", rewards["contact_penalty"])

        # print("Contact forces:", rewards["contact_penalty"])
        # print("contact forces", self.contact_forces)
        
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

        # cloth_positions = self._cloth_plain.root_physx_view.get_positions().reshape(self.num_envs, -1, 3)
        cloth_positions = self._cloth.root_physx_view.get_positions().reshape(self.num_envs, -1, 3)
        mean_cloth_height = cloth_positions[:, :, 2].mean(dim=-1)

        ### If some values are not valid, set total reward to zero for those envs
        # TODO
        # invalid_env = torch.bitwise_or(
        #     # ((self.ee_distances > 0.8) | (self.ee_distances < 0.5)).any(0), 
        #     (self.z_abs_pos < 0.15).any(0)) # | (mean_cloth_height > 0.2)
        invalid_env = (self.z_abs_pos < 0.09).any(0)
        total_reward[invalid_env] -= 5
        self.print_rewards(reward_dict=self._rewards, total_reward=total_reward)
        # total_reward = torch.zeros(self.num_envs, device=self.device)
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
            print(f"{reward_name} mean: \t {reward_value.mean().item():.4f}")
            print(f"{reward_name} std : \t {reward_value.std().item():.4f}")
        
        if total_reward is not None:
            print(f"\nTotal Reward: \t {total_reward.mean().item():.4f}")
        
        print("-------------------------")

    def _reset_idx(self, env_ids: torch.Tensor | None):
        """Reset the state of multiple robots and environment objects properly."""
        super()._reset_idx(env_ids)

        # --- dump finished episodes for these envs ---
        if self.cfg.enable_dataset_logging and self.reset_count > 0:
            self._dump_episodes(env_ids)

        ### Plot both target and actual joint trajectories for comparison
        # plt.cla()
        # plt.clf()
        # plt.plot(self.target_joints_buffer[:,0,:].cpu().numpy()) # just for env_0 and 4 joints
        # plt.gca().set_prop_cycle(None)
        # plt.plot(self.actual_joints_buffer[:,0,:].cpu().numpy(), '--') # just for env_0 and 4 joints
        # plt.savefig('./logs/snapshots/joint_comparison.jpg')

        ### Calculate convex hull
        # # Get the cloth positions for the current environment
        # cloth_nodes = self._cloth_plain.root_physx_view.get_positions().reshape(self.num_envs, -1, 3)
        # current_area = []
        # for env_idx in env_ids:
        #     # Calculate the convex hull of the cloth nodes
        #     try:
        #         current_hull = ConvexHull(cloth_nodes[env_idx][:,0:2].cpu().numpy())
        #         current_area.append(current_hull.volume)
        #         # print(f"Current convex hull area for env {env_idx}: {current_area[-1]}")
        #     except Exception as e:
        #         print(f"Error calculating convex hull for env {env_idx}: {e}")
        #         current_area.append(0.0)
        # current_area = torch.tensor(current_area, device=self.device)
        # print(f"Average convex hull area: {current_area.mean()}")
        # print(f"SD of convex hull area: {current_area.std()}")
        # print(f"Mean percentages of convex hull area: {(current_area / (self.cloth_lengths * 0.7)).mean()}")
        # print(f"Mean SD of convex hull area: {(current_area / (self.cloth_lengths * 0.7)).std()}")
        # breakpoint()

        self.action_steps[env_ids] = 0
        # Create joints
        if not self.joints_created:
            self.create_joints(env_ids)
            self.joints_created = True
        # self.sim.pause()

        if self.default_states is None:
            try:
                for key in self.robots.keys():
                    self.robot_entity_cfg[key].resolve(self.scene)
            except Exception:
                pass


        if self.default_states is not None:
            # self._cloth_plain.root_physx_view.set_velocities(
            #     torch.zeros((self.num_envs, self._cloth_plain.root_physx_view.max_particles_per_cloth * 3), 
            #     device=self.device), indices=env_ids)
            # self._cloth_plain.root_physx_view.set_positions(
            #     self.default_states['_cloth_plain'][env_ids], indices=env_ids)
            # self._cloth_plain.update(self.physics_dt)
            self._cloth.root_physx_view.set_velocities(
                torch.zeros((self.num_envs, self._cloth.root_physx_view.max_particles_per_cloth * 3), 
                device=self.device), indices=env_ids)
            self._cloth.root_physx_view.set_positions(
                self.default_states['_cloth'][env_ids], indices=env_ids)
            self._cloth.update(self.physics_dt)


            self._handle_1.write_root_state_to_sim(
                self.default_states['_handle_1'][env_ids], env_ids=env_ids)

            self._handle_2.write_root_state_to_sim(
                self.default_states['_handle_2'][env_ids], env_ids=env_ids)
            self._box.write_root_state_to_sim(
                self.default_states['_box'][env_ids], env_ids=env_ids)
        if self.default_states is None:
            try:
                for key in self.robots.keys():
                    self.robot_entity_cfg[key].resolve(self.scene)
            except Exception:
                pass
            self.default_states = {}
            # cloth_positions = self._cloth_plain.root_physx_view.get_positions()
            cloth_positions_ = self._cloth.root_physx_view.get_positions()


            
            # self.default_states['_cloth_plain'] = cloth_positions.clone()
            self.default_states['_cloth'] = cloth_positions_.clone()
            self._cloth.root_physx_view.set_positions(
                self.default_states['_cloth'][env_ids], indices=env_ids)
            self._cloth.update(self.physics_dt)

            self.default_states['_handle_1'] = self._handle_1.data.root_state_w.clone()
            self.default_states['_handle_1'][:,7:] = 0.0
            self.default_states['_handle_2'] = self._handle_2.data.root_state_w.clone()
            self.default_states['_handle_2'][:,7:] = 0.0
            # self._initial_corners = torch.stack((
            #     self.default_states['_handle_1'][:,0:3],
            #     self.default_states['_handle_2'][:,0:3],
            # ), axis=1)

            self.default_states['_box'] = self._box.data.root_state_w.clone()
            # self.default_states['_box'][:,7:] = 0.0

        for robot_key in self.robots.keys():  # Loop through all robots dynamically
            robot = self.robots[robot_key]
            joint_pos = robot.data.default_joint_pos[env_ids].clone()
            joint_vel = robot.data.default_joint_vel[env_ids].clone()
            robot.set_joint_position_target(joint_pos, env_ids=env_ids)
            robot.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)

        # self._cloth_plain.root_physx_view.set_velocities(
        #     torch.zeros((len(env_ids), self._cloth_plain.root_physx_view.max_particles_per_cloth * 3), 
        #     device=self.device), indices=env_ids)
        # self._cloth_plain.root_physx_view.set_positions(
        #     self.default_states['_cloth_plain'][env_ids], indices=env_ids)
        # self._cloth_plain.update(self.physics_dt)
        self._cloth.root_physx_view.set_velocities(
            torch.zeros((len(env_ids), self._cloth.root_physx_view.max_particles_per_cloth * 3), 
            device=self.device), indices=env_ids)
        self._cloth.root_physx_view.set_positions(
            self.default_states['_cloth'][env_ids], indices=env_ids)
        self._cloth.update(self.physics_dt)

        self._handle_1.write_root_state_to_sim(
            self.default_states['_handle_1'][env_ids], env_ids=env_ids)

        self._handle_2.write_root_state_to_sim(
            self.default_states['_handle_2'][env_ids], env_ids=env_ids)


        self._cloth.root_physx_view.set_velocities(
            torch.zeros((len(env_ids), self._cloth.root_physx_view.max_particles_per_cloth * 3), 
            device=self.device), indices=env_ids)
        self._cloth.root_physx_view.set_positions(
            self.default_states['_cloth'][env_ids], indices=env_ids)
        self._cloth.update(self.physics_dt)

        if self.cfg.randomize_fold_length:
            fold_length = torch.empty(env_ids.size(0), device=self.device).uniform_(*self.cfg.fold_length_range)
            self._default_fold_length[env_ids] = fold_length
            self._fold_lengths[env_ids] = fold_length
        else:
            fold_length =  torch.full((env_ids.size(0),), self.cfg.default_fold_length, device=self.device)
            self._fold_lengths[env_ids] = fold_length
            self._default_fold_length = torch.full((self.num_envs,), self.cfg.default_fold_length, device=self.device)
        # print(f"Randomized fold lengths for envs {env_ids.tolist()}: {fold_length.tolist()}")

        self._compute_fold_pairs_on_reset(env_ids)
        # print(f"[fold pairs] computed for {env_ids.numel()} envs, "
        #         f"{pair_idx_sub.size(1)} pairs (max {self._fold_pair_idx.size(1)})")
        # print(f"[fold pairs] fold length {fold_len_vec[env_ids]}")
        # # printing fold pairs for all envs
        # for i in env_ids:
        #     print(f" env {i}: {self._fold_pair_idx[i, :self._fold_pair_valid_counts[i]]} "
        #             f"(count {self._fold_pair_valid_counts[i]})")

        # print(f"[fold lengths] {fold_len_vec[env_ids]}")



        # self.sim.play()

        # H  = self.cfg.decimation
        # self.prev_corners_buf = torch.zeros((H, self.num_envs, 6), device=self.device)
        # self.prev_abs_buf = torch.zeros((H, self.num_envs, 2), device=self.device)

        self.reset_count += 1

    def _compute_fold_pairs_on_reset(self, env_ids: torch.Tensor):
        """Compute and cache fold pairs for all envs, using default fold length."""
        assert self._default_fold_length is not None and torch.is_tensor(self._default_fold_length), \
        "Default fold length must be a tensor and not None"
        fold_len_vec = self._default_fold_length.to(self.device).view(-1)
        pair_idx_sub, valid_counts_sub = self.get_edge_and_fold_pairs_idx(
            fold_length=fold_len_vec[env_ids],
            include_edges=self._fold_params.get("include_edges", True),
            anchor_on_top=self._fold_params.get("anchor_on_top", False),
            env_ids=env_ids,  # ensures K = len(env_ids)
        )

        # ensure cached storage exists and is large enough in P_max
        need_init = (self._fold_pair_idx is None) or (self._fold_pair_valid_counts is None)
        if need_init:
            self._fold_pair_idx = torch.full(
                (self.num_envs, pair_idx_sub.size(1), 2), -1, device=self.device, dtype=torch.long
            )
            self._fold_pair_valid_counts = torch.zeros(
                (self.num_envs,), device=self.device, dtype=torch.long
            )
        elif self._fold_pair_idx.size(1) < pair_idx_sub.size(1):
            # grow P_max
            new_P = pair_idx_sub.size(1)
            new_idx = torch.full(
                (self.num_envs, new_P, 2), -1, device=self.device, dtype=torch.long
            )
            new_idx[:, : self._fold_pair_idx.size(1), :] = self._fold_pair_idx
            self._fold_pair_idx = new_idx  # counts tensor remains same length

        # write subset back
        self._fold_pair_idx[env_ids, : pair_idx_sub.size(1), :] = pair_idx_sub
        self._fold_pair_valid_counts[env_ids] = valid_counts_sub
        # print(f"[fold pair idx] {self._fold_pair_idx}")
        # print(f"[fold pair counts] {self._fold_pair_valid_counts}")

    def create_joints(self, env_ids):
        for idx in env_ids:
            # cube_1_path = Sdf.Path("/World/envs/env_" + str(int(idx)) + "/Cloth/RightCube")
            # cube_2_path = Sdf.Path("/World/envs/env_" + str(int(idx)) + "/Cloth/LeftCube")
            # cylinder_path = Sdf.Path("/World/envs/env_" + str(int(idx)) + "/Robot1/particle_cloth_one_robot/Cube_01")
            # panda_1_finger_path = Sdf.Path("/World/envs/env_" + str(int(idx)) + "/Robot1/panda_leftfinger")
            # panda_2_finger_path = Sdf.Path("/World/envs/env_" + str(int(idx)) + "/Robot2/panda_rightfinger")
            # panda_finger_path = Sdf.Path("/World/envs/env_" + str(int(idx)) + "/Robot1/panda_leftfinger")
            # joint_1_path = panda_1_finger_path.AppendElementString("fixedJoint")
            # joint_2_path = panda_2_finger_path.AppendElementString("fixedJoint")
            # joint_path = panda_finger_path.AppendElementString("fixedJoint")

            handle_1 = Sdf.Path("/World/envs/env_" + str(int(idx)) + "/Cloth/LeftCube")
            handle_2 = Sdf.Path("/World/envs/env_" + str(int(idx)) + "/Cloth/RightCube")
            
            panda_left_finger_path = Sdf.Path("/World/envs/env_" + str(int(idx)) + "/Robot1/panda_leftfinger")
            panda_right_finger_path = Sdf.Path("/World/envs/env_" + str(int(idx)) + "/Robot1/panda_rightfinger")

            joint_left_path = panda_left_finger_path.AppendElementString("fixedJoint")
            joint_right_path = panda_right_finger_path.AppendElementString("fixedJoint")

            fixedJoint_1 = UsdPhysics.FixedJoint.Define(self.scene.stage, joint_left_path)
            fixedJoint_1.CreateBody0Rel().SetTargets([panda_left_finger_path])
            fixedJoint_1.CreateBody1Rel().SetTargets([handle_1])
            fixedJoint_1.CreateLocalPos0Attr().Set(Gf.Vec3f(+0.25,0,0.07))
            fixedJoint_1.CreateLocalRot0Attr().Set(Gf.Quatf(0.707,0.707,0.0,0.0))
            # fixedJoint_1.CreateLocalPos1Attr().Set(Gf.Vec3f(0,0,0))
            # fixedJoint_1.CreateLocalRot1Attr().Set(Gf.Quatf(1,0,0,0))

            fixedJoint_2 = UsdPhysics.FixedJoint.Define(self.scene.stage, joint_right_path)
            fixedJoint_2.CreateBody0Rel().SetTargets([panda_right_finger_path])
            fixedJoint_2.CreateBody1Rel().SetTargets([handle_2])
            fixedJoint_2.CreateLocalPos0Attr().Set(Gf.Vec3f(-0.25,0,0.07))
            fixedJoint_2.CreateLocalRot0Attr().Set(Gf.Quatf(0.707,0.707,0.0,0.0))

            # fixedJoint_2.CreateLocalRot0Attr().Set(Gf.Quatf(0.5,0.5,0.5,0.5))
            # fixedJoint_2.CreateLocalRot1Attr().Set(Gf.Quatf(1,0,0,0))
            # fixedJoint_2.CreateLocalPos1Attr().Set(Gf.Vec3f(0,0,0))

            # joint between two cubes
            # fixed_joint_3 = UsdPhysics.FixedJoint.Define(self.scene.stage, joint_right_path)
            # fixed_joint_3.CreateBody0Rel().SetTargets([handle_1])
            # fixed_joint_3.CreateBody1Rel().SetTargets([handle_2])

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
        

        ## new mahed
        box_length = self.box_lengths
        fold_length = self._fold_lengths
        cloth_lengths = self.cloth_lengths

        if self.cfg.disable_init_motion:
            corner_traj_obs *= 0.0
            act_hist *= 0.0


    

        ### Either pass the action history and corner trajectory observations
        # observations = torch.cat((act_hist, corner_traj_obs), dim=-1)
        ### Or action history and cloth lengths
        # observations = torch.cat((joint_obs, self.cloth_lengths.unsqueeze(-1)), dim=-1)
        ### Or current absolute pose and previous actions and corner trajectories
        # observations = torch.cat((joint_obs, act_hist, corner_traj_obs), dim=-1)
        # observations = torch.cat((act_hist, corner_traj_obs), dim=-1)

        # # box length and fold length added to observation
        # observations = torch.cat((joint_obs, act_hist, corner_traj_obs, box_length.unsqueeze(-1), fold_length.unsqueeze(-1)), dim=-1)
        observations = torch.cat((joint_obs[:,:7], self.cloth_lengths.unsqueeze(-1), box_length.unsqueeze(-1), fold_length.unsqueeze(-1), cloth_lengths.unsqueeze(-1)), dim=-1)
        # print(f"joint_obs shape {joint_obs[:,:7].shape}, box_length shape {box_length.unsqueeze(-1).shape}, fold_length shape {fold_length.unsqueeze(-1).shape}, cloth_lengths shape {cloth_lengths.unsqueeze(-1).shape}")

        # observations = torch.cat((joint_obs, act_hist, corner_traj_obs, box_length.unsqueeze(-1)), dim=-1)
        # print("observations.shape:", observations.shape)
        print("observations:", observations)

        return {"policy": observations}
    

    def _dump_episodes(self, env_ids: torch.Tensor):
        """
        Save per-env episode trajectories:
          q:  [T, DOF]          (robot_1 only; extend if you add more robots)
          ee: [T, 7]            (x,y,z, qw,qx,qy,qz)
          dmp:[T, action_dim]
        """
        T = min(int(self.action_steps[0].item()), self.cfg.max_episode_length)
        if T <= 0:
            return

        for env_idx in env_ids.tolist():
            episode_idx = int(self._episode_counters[env_idx].item())

            # Choose output path
            suffix = "npz" if self.cfg.dataset_format == "npz" else "pt"
            out_path = os.path.join(
                self.dataset_dir, f"env{env_idx:02d}_ep{episode_idx:05d}.{suffix}"
            )

            # Prepare payload (CPU numpy for npz; torch tensors for pt)
            q_robot1  = self._traj_q["robot_1"][:T, env_idx].detach().cpu()
            ee_robot1 = self._traj_ee["robot_1"][:T, env_idx].detach().cpu() 
            # ee_robot1[:,0:3] -= self.scene.env_origins[env_idx].detach().cpu() # make relative to env origin
            dmp_all   = self._traj_dmp[:T, env_idx].detach().cpu()

            meta = {
                "dt": float(self.physics_dt),
                "env_id": int(env_idx),
                "episode": int(episode_idx),
                "run_timestamp": self.run_timestamp,
            }

            if self.cfg.dataset_format == "npz":
                np.savez_compressed(
                    out_path,
                    q=q_robot1.numpy(),          # [T, DOF]
                    ee=ee_robot1.numpy(),        # [T, 7]
                    dmp=dmp_all.numpy(),         # [T, action_dim]
                    **meta,
                )
            else:  # torch .pt
                torch.save(
                    {"q": q_robot1, "ee": ee_robot1, "dmp": dmp_all, **meta},
                    out_path,
                )

            # Increment episode counter for that env
            self._episode_counters[env_idx] += 1

            # (Optional) clear the recorded slices for that env
            self._traj_q["robot_1"][:, env_idx].zero_()
            self._traj_ee["robot_1"][:, env_idx].zero_()
            self._traj_dmp[:, env_idx].zero_()

    def _get_markers(self):
        """
        Like _get_corners, but for the 8 box markers.
        Returns env-relative positions (env-origin offset removed).
        Shape: [num_envs, 8, 3] as torch.float32 on self.device.
        Missing markers are left as NaNs.
        """

        stage = omni.usd.get_context().get_stage()
        xcache = UsdGeom.XformCache(Usd.TimeCode.Default())

        N = self.scene.cfg.num_envs
        pos_world = torch.full((N, 8, 3), float('nan'), device=self.device, dtype=torch.float32)

        for env_i in range(N):
            base = f"/World/envs/env_{env_i}/Box"
            for k in range(8):
                # try both flattened and nested under SmallKLT
                for p in (f"{base}/Visuals/marker_{k}", f"{base}/SmallKLT/Visuals/marker_{k}"):
                    prim = stage.GetPrimAtPath(p)
                    if prim and prim.IsValid():
                        # if prim.IsInactive():
                        prim.SetActive(True)
                        if prim.HasPayload():
                            prim.Load()
                        xf = xcache.GetLocalToWorldTransform(prim)
                        t = Gf.Transform(xf).GetTranslation()
                        pos_world[env_i, k] = torch.tensor([t[0], t[1], t[2]],
                                                        device=self.device, dtype=torch.float32)
                        break  # found this marker_k, go to next

        # Remove env-origin offset so output is env-relative (like your cloth APIs)
        pos_env = pos_world - self.scene.env_origins.unsqueeze(1)  # [N,1,3] broadcast
        return pos_env
 
    def _log_robot_box_contacts(self):
        s = self._contact_robot_box
        Fm, Fn = s.data.force_matrix_w, s.data.net_forces_w

        N = self.scene.cfg.num_envs
        maxF = torch.zeros(N, device=self.device)

        if Fm is not None and Fm.numel() > 0:
            v = Fm.norm(dim=-1)                      # [N, M, C]
            v = v.amax(dim=2, keepdim=False)         # [N, M]
            maxF = v.amax(dim=1, keepdim=False)      # [N]
        elif Fn is not None and Fn.numel() > 0:
            v = Fn.norm(dim=-1)                      # [N, M]
            maxF = v.amax(dim=1, keepdim=False)      # [N]

        # ✅ reshape to (N, 1) so it works like other reward terms
        maxF = maxF.unsqueeze(-1)
        # print(f"[contact] max robot-box forces: {maxF.squeeze(-1)}")
        return maxF
 
    def set_cloth_color(self, rgb=(0.75, 0.75, 0.75), roughness=0.6):
        from pxr import Usd, UsdGeom, UsdShade, Sdf, Gf
        import omni.usd
        stage = omni.usd.get_context().get_stage()

        # material + preview surface
        mat_path = "/World/Materials/ClothTint"
        mat = UsdShade.Material.Define(stage, mat_path)
        sh  = UsdShade.Shader.Define(stage, f"{mat_path}/PreviewSurface")
        sh.CreateIdAttr("UsdPreviewSurface")
        sh.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*rgb))
        sh.CreateInput("roughness",    Sdf.ValueTypeNames.Float).Set(float(roughness))
        sh.CreateInput("metallic",     Sdf.ValueTypeNames.Float).Set(0.0)

        # ✅ create the shader's surface output, then connect it to the material output
        surf_out = sh.CreateOutput("surface", Sdf.ValueTypeNames.Token)
        mat.CreateSurfaceOutput().ConnectToSource(surf_out)

        # bind to all meshes under each cloth prim
        for env_i in range(self.scene.cfg.num_envs):
            root = stage.GetPrimAtPath(f"/World/envs/env_{env_i}/Cloth")
            if not root:
                continue
            for prim in Usd.PrimRange(root):
                if prim.IsA(UsdGeom.Mesh):
                    UsdShade.MaterialBindingAPI(prim).Bind(mat)

    def set_cloth_texture(self, image_path, fallback_tint=(0.9, 0.9, 0.9), roughness=0.55):
        from pxr import Usd, UsdGeom, UsdShade, Sdf, Gf
        import omni.usd
        stage = omni.usd.get_context().get_stage()

        mat_path = "/World/Materials/ClothTex"
        mat = UsdShade.Material.Define(stage, mat_path)

        # primvar reader for 'st' UVs
        st = UsdShade.Shader.Define(stage, f"{mat_path}/stReader")
        st.CreateIdAttr("UsdPrimvarReader_float2")
        st.CreateInput("varname", Sdf.ValueTypeNames.Token).Set("st")
        st_out = st.CreateOutput("result", Sdf.ValueTypeNames.Float2)

        # texture node
        tex = UsdShade.Shader.Define(stage, f"{mat_path}/BaseColorTex")
        tex.CreateIdAttr("UsdUVTexture")
        tex.CreateInput("file",  Sdf.ValueTypeNames.Asset).Set(image_path)
        tex.CreateInput("wrapS", Sdf.ValueTypeNames.Token).Set("repeat")
        tex.CreateInput("wrapT", Sdf.ValueTypeNames.Token).Set("repeat")
        tex.CreateInput("st",    Sdf.ValueTypeNames.Float2).ConnectToSource(st_out)
        tex_rgb = tex.CreateOutput("rgb", Sdf.ValueTypeNames.Float3)   # ✅ explicit output

        # preview surface
        sh = UsdShade.Shader.Define(stage, f"{mat_path}/PreviewSurface")
        sh.CreateIdAttr("UsdPreviewSurface")
        sh.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*fallback_tint))
        sh.CreateInput("roughness",    Sdf.ValueTypeNames.Float).Set(float(roughness))
        sh.CreateInput("metallic",     Sdf.ValueTypeNames.Float).Set(0.0)
        # drive diffuse with texture
        sh.GetInput("diffuseColor").ConnectToSource(tex_rgb)

        # connect shader → material
        surf_out = sh.CreateOutput("surface", Sdf.ValueTypeNames.Token)
        mat.CreateSurfaceOutput().ConnectToSource(surf_out)

        # bind to cloth meshes
        for env_i in range(self.scene.cfg.num_envs):
            root = stage.GetPrimAtPath(f"/World/envs/env_{env_i}/Cloth")
            if not root:
                continue
            for prim in Usd.PrimRange(root):
                if prim.IsA(UsdGeom.Mesh):
                    UsdShade.MaterialBindingAPI(prim).Bind(mat)

    # def no_ghost_mode(self, aggressive=True):
    #     import carb
    #     s = carb.settings.get_settings()
    #     def _set(k, v):
    #         try: s.set(k, v)
    #         except: pass

    #     # Absolutely no temporal AA/upsampling/denoise/motion vectors
    #     _set("/rtx/antialiasing/taa/enable", False)
    #     _set("/rtx/antialiasing/enable",   False)   # disable post AA too
    #     _set("/rtx/post/aa/enable",        False)
    #     _set("/rtx/dlss/enable",           False)
    #     _set("/rtx/dlaa/enable",           False)
    #     _set("/rtx/motionBlur/enabled",    False)
    #     _set("/rtx/post/motionBlur/enable",False)

    #     # Accumulation toggles used across RTX stacks
    #     _set("/rtx/accumulation/enabled",  False)
    #     _set("/rtx/accumulation/numFrames",1)
    #     _set("/rtx/accumulation/reset",    True); _set("/rtx/accumulation/reset", False)

    #     # Reuse/temporal toggles in DI/GI/AO
    #     for k in [
    #         "/rtx/di/temporalReuse",
    #         "/rtx/gi/temporalReuse",
    #         "/rtx/ambientOcclusion/temporal",
    #         "/rtx/di/denoiser/enabled",
    #         "/rtx/gi/denoiser/enabled",
    #         "/rtx/optixDenoiser/enabled",
    #         "/rtx/post/denoise/enable",
    #         "/rtx/motionVectors/enabled",
    #     ]:
    #         _set(k, False)

    #     # Keep pipeline synchronous while recording
    #     _set("/app/asyncRendering", False)

    #     # Optional: cut all indirect terms (pure direct lighting)
    #     if aggressive:
    #         _set("/rtx/indirectDiffuse", False)
    #         _set("/rtx/reflections",     False)

    #     # Nudge any remaining nodes to discard history
    #     for k in ("/rtx/resetHistory", "/rtx/post/aa/resetHistory"):
    #         _set(k, True); _set(k, False)

    def pulse_reset_temporal_history(self):
        """Call once per frame before grabbing a sensor image, if trails persist."""
        import carb
        s = carb.settings.get_settings()
        for k in ("/rtx/resetHistory", "/rtx/post/aa/resetHistory"):
            try:
                s.set(k, True); s.set(k, False)
            except Exception:
                pass
        # Viewport fallback (may be no-op for sensors, but harmless):
        try:
            import omni.kit.viewport.utility as vp_util
            vp_util.reset_viewport_accumulation()
        except Exception:
            pass

    def capture_clean_frame(self):
        s = carb.settings.get_settings()
        # pulse all reset flags
        for k in ("/rtx/resetHistory", "/rtx/post/aa/resetHistory", "/rtx/accumulation/reset"):
            try: s.set(k, True); s.set(k, False)
            except: pass

        # force a renderer tick that matches this request
        try:
            # If your camera has a render() or update() call, call it exactly once now:
            self._camera.update(self.physics_dt)
        except Exception:
            pass

        # IMPORTANT: clone() to decouple from the GPU ring buffer
        img_rgb = self._camera.data.output["rgb"][0].clone()
        return img_rgb

    def switch_to_storm_for_capture(self, msaa=8):
        import carb
        s = carb.settings.get_settings()
        def _set(k, v):
            try: s.set(k, v)
            except: pass
        # Switch Hydra engine to Storm
        for k in ["/app/renderer/hydraEngine", "/app/renderer/pipeline"]:
            _set(k, "HydraStorm")
        # Typical Storm AA controls (silently ignored if not present)
        for k in ["/hydra/storm/msaaSamples", "/hydra/storm/antialiasing"]:
            _set(k, int(msaa) if "msaa" in k else True)
        # Make sure RTX-specific features are off
        _set("/rtx/antialiasing/enable", False)

    # def setup_clean_stage(
    #     self,
    #     renderer="HydraStorm",
    #     key_intensity=1400.0,          # ↓ from 1800
    #     key_exposure=-1.0,            # new: 1 stop darker (~2× dimmer)
    #     key_angle_deg=1.5,            # ↑ softer sun for gentler highlights
    #     key_rotate_xyz=(60.0, -40.0, 0.0),
    #     ground_color=(0.3, 0.7, 0.3),
    #     ground_roughness=0.15,
    # ):

    #     s = carb.settings.get_settings()
    #     stage = omni.usd.get_context().get_stage()

    #     def _set(k, v):
    #         try: s.set(k, v)
    #         except Exception: pass

    #     # --- Stage conventions ---
    #     UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    #     UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)

    #     # --- Renderer ---
    #     if renderer == "HydraStorm":
    #         _set("/app/renderer/hydraEngine", "HydraStorm")
    #         _set("/app/renderer/pipeline", "HydraStorm")
    #         _set("/hydra/storm/antialiasing", True)
    #         _set("/hydra/storm/msaaSamples", 16)   # use the highest MSAA you can afford
    #         # kill RTX toggles to avoid conflicts
    #         _set("/rtx/antialiasing/enable", False)
    #     else:  # RTX direct lighting (no GI/denoise/temporal)
    #         _set("/rtx/rendermode", "RayTracedLighting")
    #         for k, v in {
    #             "/rtx/antialiasing/enable": True,
    #             "/rtx/antialiasing/taa/enable": False,   # keep off for video (no ghosting)
    #             "/rtx/post/aa/enable": True,
    #             "/rtx/dlss/enable": False,
    #             "/rtx/dlaa/enable": False,
    #             "/rtx/motionBlur/enabled": False,
    #             "/rtx/post/motionBlur/enable": False,
    #             "/rtx/optixDenoiser/enabled": False,
    #             "/rtx/post/denoise/enable": False,
    #             "/rtx/ao": False,
    #             "/rtx/indirectDiffuse": False,
    #             "/rtx/reflections": False,
    #             "/rtx/shadows/softShadows": False,
    #             "/rtx/post/tonemapper/enable": True,
    #             "/rtx/post/exposure/auto": False,
    #             "/rtx/post/exposure/value": 0.0,
    #             "/rtx/post/saturation": 1.0,
    #             "/rtx/post/contrast": 1.0,
    #         }.items():
    #             _set(k, v)

    #     # --- Nuke all temporal history/accumulation ---
    #     for k in ("/rtx/accumulation/enabled",
    #             "/rtx/di/temporalReuse",
    #             "/rtx/gi/temporalReuse",
    #             "/rtx/ambientOcclusion/temporal",
    #             "/rtx/motionVectors/enabled"):
    #         _set(k, False)
    #     for k in ("/rtx/resetHistory", "/rtx/accumulation/reset", "/rtx/post/aa/resetHistory"):
    #         _set(k, True); _set(k, False)

    #     # --- Lights: single distant key with hard shadows ---
    #     if not stage.GetPrimAtPath("/World/Lights"):
    #         UsdGeom.Xform.Define(stage, "/World/Lights")

    #     def _enable_shadows(prim):
    #         attr = prim.GetAttribute("shadow:enable")
    #         (attr or prim.CreateAttribute("shadow:enable", Sdf.ValueTypeNames.Bool)).Set(True)

    #     key = UsdLux.DistantLight.Define(stage, "/World/Lights/Key")
    #     key.CreateIntensityAttr().Set(float(key_intensity))
    #     key.CreateExposureAttr().Set(float(key_exposure))
    #     key.CreateAngleAttr().Set(float(key_angle_deg))
    #     key.CreateColorAttr().Set(Gf.Vec3f(0.8, 0.8, 0.8))
    #     UsdGeom.XformCommonAPI(key.GetPrim()).SetRotate(tuple(key_rotate_xyz), UsdGeom.XformCommonAPI.RotationOrderXYZ)

    #     _enable_shadows(key.GetPrim())

    #     # Remove any other lights/domes
    #     for path in ("/World/Lights/Fill", "/World/Lights/Rim", "/World/Lights/Dome"):
    #         prim = stage.GetPrimAtPath(path)
    #         if prim: prim.SetActive(False)

    #     # --- Make sure only ONE visible ground exists (avoid z-fighting) ---
    #     # Hide any visual plane the TerrainImporter may have put under /World/ground
    #     for p in ("/World/ground/Visuals", "/World/ground/visuals",
    #             "/World/ground/Plane", "/World/ground/VisualPlane"):
    #         prim = stage.GetPrimAtPath(p)
    #         if prim and prim.IsValid():
    #             try:
    #                 UsdGeom.Imageable(prim).GetVisibilityAttr().Set(UsdGeom.Tokens.invisible)
    #             except Exception:
    #                 prim.SetActive(False)

    #     # Create our own visual ground slightly above the PhysX plane
    #     GROUND_ROOT = "/World/CleanGround"
    #     if not stage.GetPrimAtPath(GROUND_ROOT):
    #         UsdGeom.Xform.Define(stage, GROUND_ROOT)

    #     ground_path = f"{GROUND_ROOT}/VisualPlane"
    #     z = 0.001  # 1 mm offset to prevent coplanar overlap with PhysX ground
    #     if not stage.GetPrimAtPath(ground_path):
    #         plane = UsdGeom.Mesh.Define(stage, ground_path)
    #         # Keep it modest in size to reduce shadow-map artifacts
    #         plane.GetPointsAttr().Set([
    #             Gf.Vec3f(-25, -25, z), Gf.Vec3f(25, -25, z),
    #             Gf.Vec3f(25,  25, z), Gf.Vec3f(-25, 25,  z),
    #         ])
    #         plane.GetFaceVertexIndicesAttr().Set([0,1,2, 0,2,3])
    #         plane.GetFaceVertexCountsAttr().Set([3,3])

    #     # --- Matte ground material (no mirror-like specular) ---
    #     mat_path = "/World/Materials/GroundMaterial"
    #     mat  = UsdShade.Material.Define(stage, mat_path)
    #     sh   = UsdShade.Shader.Define(stage, f"{mat_path}/PBR")
    #     sh.CreateIdAttr("UsdPreviewSurface")
    #     sh.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*ground_color))
    #     sh.CreateInput("roughness",    Sdf.ValueTypeNames.Float).Set(float(ground_roughness))  # e.g., 0.55
    #     sh.CreateInput("metallic",     Sdf.ValueTypeNames.Float).Set(0.0)
    #     sh.CreateInput("specular",     Sdf.ValueTypeNames.Float).Set(0.0)  # ↓ kill specular aliasing
    #     surf = sh.CreateOutput("surface", Sdf.ValueTypeNames.Token)
    #     mat.CreateSurfaceOutput().ConnectToSource(surf)
    #     UsdShade.MaterialBindingAPI(stage.GetPrimAtPath(ground_path)).Bind(mat)

    #     # --- Hide any environment/backdrop meshes ---
    #     env_prim = stage.GetPrimAtPath("/World/ground/Environment")
    #     if env_prim:
    #         UsdGeom.Imageable(env_prim).GetVisibilityAttr().Set(UsdGeom.Tokens.invisible)

    def setup_clean_stage(
        self,
        renderer="HydraStorm",
        key_intensity=1400.0,      # even, not blinding in Storm
        key_exposure=-2.0,        # ~2^7 dimmer; tweak +/-1 if needed
        key_angle_deg=6.0,        # softer shadows across the grid
        key_rotate_xyz=(65.0, -35.0, 15.0),  # off-top, gentle direction
        ground_color=(0.18, 0.6, 0.18),     # neutral mid-gray (no bounce)
        ground_roughness=0.25,    # matte ground = no glare
    ):

        s = carb.settings.get_settings()
        stage = omni.usd.get_context().get_stage()

        def _set(k, v):
            try: s.set(k, v)
            except Exception: pass

        # --- Stage conventions ---
        UsdGeom.SetStageMetersPerUnit(stage, 1.0)
        UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)

        # --- Renderer ---
        if renderer == "HydraStorm":
            _set("/app/renderer/hydraEngine", "HydraStorm")
            _set("/app/renderer/pipeline", "HydraStorm")
            _set("/hydra/storm/antialiasing", True)
            _set("/hydra/storm/msaaSamples", 16)   # use the highest MSAA you can afford
            # kill RTX toggles to avoid conflicts
            _set("/hydra/storm/enableSceneLights", True)   # ⬅️ use real lights
            _set("/hydra/storm/enableCameraLight", False)  # ⬅️ kill headlight
            _set("/rtx/antialiasing/enable", False)
        else:  # RTX direct lighting (no GI/denoise/temporal)
            _set("/rtx/rendermode", "RayTracedLighting")
            for k, v in {
                "/rtx/antialiasing/enable": True,
                "/rtx/antialiasing/taa/enable": False,   # keep off for video (no ghosting)
                "/rtx/post/aa/enable": True,
                "/rtx/dlss/enable": False,
                "/rtx/dlaa/enable": False,
                "/rtx/motionBlur/enabled": False,
                "/rtx/post/motionBlur/enable": False,
                "/rtx/optixDenoiser/enabled": False,
                "/rtx/post/denoise/enable": False,
                "/rtx/ao": False,
                "/rtx/indirectDiffuse": False,
                "/rtx/reflections": False,
                "/rtx/shadows/softShadows": False,
                "/rtx/post/tonemapper/enable": True,
                "/rtx/post/exposure/auto": False,
                "/rtx/post/exposure/value": 0.0,
                "/rtx/post/saturation": 1.0,
                "/rtx/post/contrast": 1.0,
            }.items():
                _set(k, v)

        # --- Nuke all temporal history/accumulation ---
        for k in ("/rtx/accumulation/enabled",
                "/rtx/di/temporalReuse",
                "/rtx/gi/temporalReuse",
                "/rtx/ambientOcclusion/temporal",
                "/rtx/motionVectors/enabled"):
            _set(k, False)
        for k in ("/rtx/resetHistory", "/rtx/accumulation/reset", "/rtx/post/aa/resetHistory"):
            _set(k, True); _set(k, False)

        # --- Lights: single distant key with hard shadows ---
        if not stage.GetPrimAtPath("/World/Lights"):
            UsdGeom.Xform.Define(stage, "/World/Lights")

        def _enable_shadows(prim):
            attr = prim.GetAttribute("shadow:enable")
            (attr or prim.CreateAttribute("shadow:enable", Sdf.ValueTypeNames.Bool)).Set(True)

        key = UsdLux.DistantLight.Define(stage, "/World/Lights/Key")
        key.CreateIntensityAttr().Set(float(key_intensity))
        key.CreateExposureAttr().Set(float(key_exposure))
        key.CreateAngleAttr().Set(float(key_angle_deg))
        key.CreateColorAttr().Set(Gf.Vec3f(1.0, 1.0, 1.0))
        UsdGeom.XformCommonAPI(key.GetPrim()).SetRotate(tuple(key_rotate_xyz), UsdGeom.XformCommonAPI.RotationOrderXYZ)

        _enable_shadows(key.GetPrim())

        # Remove any other lights/domes
        for path in ("/World/Lights/Fill", "/World/Lights/Rim", "/World/Lights/Dome"):
            prim = stage.GetPrimAtPath(path)
            if prim: prim.SetActive(False)

        # --- Make sure only ONE visible ground exists (avoid z-fighting) ---
        # Hide any visual plane the TerrainImporter may have put under /World/ground
        for p in ("/World/ground/Visuals", "/World/ground/visuals",
                "/World/ground/Plane", "/World/ground/VisualPlane"):
            prim = stage.GetPrimAtPath(p)
            if prim and prim.IsValid():
                try:
                    UsdGeom.Imageable(prim).GetVisibilityAttr().Set(UsdGeom.Tokens.invisible)
                except Exception:
                    prim.SetActive(False)

        # Create our own visual ground slightly above the PhysX plane
        GROUND_ROOT = "/World/CleanGround"
        if not stage.GetPrimAtPath(GROUND_ROOT):
            UsdGeom.Xform.Define(stage, GROUND_ROOT)

        ground_path = f"{GROUND_ROOT}/VisualPlane"
        z = 0.001  # 1 mm offset to prevent coplanar overlap with PhysX ground
        if not stage.GetPrimAtPath(ground_path):
            plane = UsdGeom.Mesh.Define(stage, ground_path)
            # Keep it modest in size to reduce shadow-map artifacts
            plane.GetPointsAttr().Set([
                Gf.Vec3f(-25, -25, z), Gf.Vec3f(25, -25, z),
                Gf.Vec3f(25,  25, z), Gf.Vec3f(-25, 25,  z),
            ])
            plane.GetFaceVertexIndicesAttr().Set([0,1,2, 0,2,3])
            plane.GetFaceVertexCountsAttr().Set([3,3])

        # --- Matte ground material (no mirror-like specular) ---
        mat_path = "/World/Materials/GroundMaterial"
        mat  = UsdShade.Material.Define(stage, mat_path)
        sh   = UsdShade.Shader.Define(stage, f"{mat_path}/PBR")
        sh.CreateIdAttr("UsdPreviewSurface")
        sh.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*ground_color))
        sh.CreateInput("roughness",    Sdf.ValueTypeNames.Float).Set(float(ground_roughness))  # e.g., 0.55
        sh.CreateInput("metallic",     Sdf.ValueTypeNames.Float).Set(0.0)
        sh.CreateInput("specular",     Sdf.ValueTypeNames.Float).Set(0.05)  # ↓ kill specular aliasing
        surf = sh.CreateOutput("surface", Sdf.ValueTypeNames.Token)
        mat.CreateSurfaceOutput().ConnectToSource(surf)
        UsdShade.MaterialBindingAPI(stage.GetPrimAtPath(ground_path)).Bind(mat)

        # --- Hide any environment/backdrop meshes ---
        env_prim = stage.GetPrimAtPath("/World/ground/Environment")
        if env_prim:
            UsdGeom.Imageable(env_prim).GetVisibilityAttr().Set(UsdGeom.Tokens.invisible)
 
    def get_edge_and_fold_pairs_idx(
        self,
        fold_length,                         # float or Tensor: scalar, [N], or [N,1]
        include_edges: bool = True,
        anchor_on_top: bool = False,         # True: anchor at top (grasped)
        env_ids: torch.Tensor | None = None  # 1D LongTensor of env indices to compute for
    ):
        """
        Compute (left,right) flat node index pairs per env for edge + interior fold lines.

        Args:
            fold_length: float scalar or per-env tensor (N,) or (N,1). If env_ids is given,
                        you can also pass (len(env_ids),) or (len(env_ids),1).
            include_edges, anchor_on_top: same semantics as before.
            env_ids: compute only for these envs (subset).

        Returns:
            pair_idx:     [K, P_max, 2] LongTensor (K = len(env_ids) or N), padded with -1
            valid_counts: [K] LongTensor
        """
        pos_all = self._cloth.root_physx_view.get_positions().reshape(self.num_envs, -1, 3)
        device = self.device

        # select subset
        if env_ids is None:
            pos = pos_all
        else:
            env_ids = env_ids.to(device=device, dtype=torch.long).view(-1)
            pos = pos_all.index_select(0, env_ids)

        K, V, _ = pos.shape

        # normalize fold_length to a (K,) vector
        if torch.is_tensor(fold_length):
            fl = fold_length.to(device=device, dtype=torch.float32)
            fl = fl.view(-1)
            # if fold_length is sized for all envs, slice by env_ids
            if fl.numel() == self.num_envs and env_ids is not None:
                fl = fl.index_select(0, env_ids)
            elif fl.numel() not in (1, K):
                raise ValueError(f"fold_length numel={fl.numel()} incompatible with K={K} and N={self.num_envs}")
            if fl.numel() == 1:
                fl = fl.expand(K)
        else:
            fl = torch.full((K,), float(fold_length), device=device, dtype=torch.float32)

        # infer grid side
        side_len = int(round(float(torch.sqrt(torch.tensor(V, device=device, dtype=torch.float32)).item())))
        side_len = max(side_len, 2)

        def row_lr_flat_indices(r: int):
            li = r * side_len
            ri = li + (side_len - 1)
            return li, ri

        # centers of top/bottom rows for metric length
        top_li, top_ri = row_lr_flat_indices(side_len - 1)
        bot_li, bot_ri = row_lr_flat_indices(0)
        center_top    = 0.5 * (pos[:, top_li, :] + pos[:, top_ri, :])   # [K,3]
        center_bottom = 0.5 * (pos[:, bot_li, :] + pos[:, bot_ri, :])   # [K,3]
        total_len = (center_top - center_bottom).norm(dim=1)             # [K]

        rows_span = max(side_len - 1, 1)
        m_per_row = total_len / rows_span                                # [K]
        steps_per_fold = torch.clamp(
            (fl / torch.clamp_min(m_per_row, 1e-9)).round().to(torch.long),
            min=1, max=rows_span
        )                                                                 # [K]

        use_only_edges = fl >= (0.5 * total_len)                         # [K] bool-ish

        # per-env row lists
        row_lists = []
        for i in range(K):
            rows = []
            if anchor_on_top:
                if include_edges:
                    rows.append(side_len - 1)
                if not bool(use_only_edges[i].item()):
                    step = int(steps_per_fold[i].item())
                    r = (side_len - 1) - step
                    while r > 0:
                        rows.append(r)
                        r -= step
                if include_edges:
                    rows.append(0)
            else:
                if include_edges:
                    rows.append(0)
                if not bool(use_only_edges[i].item()):
                    step = int(steps_per_fold[i].item())
                    r = step
                    while r < (side_len - 1):
                        rows.append(r)
                        r += step
                if include_edges:
                    rows.append(side_len - 1)
            row_lists.append(rows)

        # pack (pad with -1)
        P_max = max((len(r) for r in row_lists), default=0)
        pair_idx = torch.full((K, P_max, 2), -1, device=device, dtype=torch.long)
        valid_counts = torch.zeros(K, dtype=torch.long, device=device)

        for i, rows in enumerate(row_lists):
            valid_counts[i] = len(rows)
            for j, r in enumerate(rows):
                li, ri = row_lr_flat_indices(int(r))
                pair_idx[i, j, 0] = li
                pair_idx[i, j, 1] = ri

        return pair_idx, valid_counts

    def get_edge_and_fold_pairs_pos(
        self,
        pair_idx: torch.Tensor | None = None,
        valid_counts: torch.Tensor | None = None,
        env_ids: torch.Tensor | None = None,
    ):
        """
        Convert cached or provided (left,right) indices to positions.
        If env_ids is given with None inputs, uses cached values for that subset.

        Returns:
            pairs_pos:    [K, P_max, 2, 3] (NaN-padded)
            valid_counts: [K]
        """
        pos_all = self._cloth.root_physx_view.get_positions().reshape(self.num_envs, -1, 3)
        device = self.device
        dtype = pos_all.dtype

        if pair_idx is None or valid_counts is None:
            if self._fold_pair_idx is None or self._fold_pair_valid_counts is None:
                raise RuntimeError("No cached fold pairs.")
            if env_ids is None:
                pair_idx = self._fold_pair_idx
                valid_counts = self._fold_pair_valid_counts
                pos = pos_all
            else:
                env_ids = env_ids.to(device=device, dtype=torch.long).view(-1)
                pair_idx = self._fold_pair_idx.index_select(0, env_ids)
                valid_counts = self._fold_pair_valid_counts.index_select(0, env_ids)
                pos = pos_all.index_select(0, env_ids)
        else:
            pos = pos_all if env_ids is None else pos_all.index_select(0, env_ids.to(device=device, dtype=torch.long).view(-1))

        K, P_max, _ = pair_idx.shape
        pairs_pos = torch.full((K, P_max, 2, 3), float('nan'), device=device, dtype=dtype)

        for n in range(K):
            k = int(valid_counts[n].item())
            if k <= 0:
                continue
            L = pair_idx[n, :k, 0]   # [k]
            R = pair_idx[n, :k, 1]   # [k]
            pairs_pos[n, :k, 0, :] = pos[n].index_select(0, L)
            pairs_pos[n, :k, 1, :] = pos[n].index_select(0, R)

        return pairs_pos, valid_counts


    def set_ground_as_grid(
        self,
        cell_size_m: float = 0.12,
        line_px: int = 2,
        bg_rgb=(0.20, 0.20, 0.20),
        line_rgb=(0.34, 0.34, 0.34),
    ):
        stage = omni.usd.get_context().get_stage()

        # 1) build/reuse grid texture
        tex_dir = "./logs/materials"
        os.makedirs(tex_dir, exist_ok=True)
        grid_png = os.path.join(tex_dir, f"grid_{line_px}px.png")
        if not os.path.exists(grid_png):
            SZ = 1024
            cells = 64
            step = SZ // cells
            px = max(1, min(line_px, step - 1))  # prevent painting whole texS
            img = np.full((SZ, SZ, 3), (np.array(bg_rgb)*255).astype(np.uint8), np.uint8)
            color = (np.array(line_rgb)*255).astype(np.uint8).tolist()
            for i in range(0, SZ, step):
                cv2.line(img, (i, 0), (i, SZ-1), color, px, lineType=cv2.LINE_AA)
                cv2.line(img, (0, i), (SZ-1, i), color, px, lineType=cv2.LINE_AA)
            cv2.imwrite(grid_png, img)

        # 2) material + UVs on the plane
        ground_path = "/World/CleanGround/VisualPlane"
        plane_size_m = 20.0
        repeats = max(1, int(round(plane_size_m / float(cell_size_m))))

        prim = stage.GetPrimAtPath(ground_path)
        mesh = UsdGeom.Mesh(prim)

        # Use PrimvarsAPI to read/create primvars:st
        pvars = UsdGeom.PrimvarsAPI(prim)
        st_pv = pvars.GetPrimvar("st")
        if (not st_pv) or (not st_pv.IsDefined()):
            st_pv = pvars.CreatePrimvar(
                "st", Sdf.ValueTypeNames.TexCoord2fArray, UsdGeom.Tokens.vertex
            )
            # 4 verts of the quad in the same order you created points 0..3
            st_pv.Set([
                Gf.Vec2f(0.0, 0.0), Gf.Vec2f(1.0, 0.0),
                Gf.Vec2f(1.0, 1.0), Gf.Vec2f(0.0, 1.0),
            ])

        # Shader network
        mat_path = "/World/Materials/GroundGrid"
        mat = UsdShade.Material.Define(stage, mat_path)

        st = UsdShade.Shader.Define(stage, f"{mat_path}/st")
        st.CreateIdAttr("UsdPrimvarReader_float2")
        st.CreateInput("varname", Sdf.ValueTypeNames.Token).Set("st")
        st_out = st.CreateOutput("result", Sdf.ValueTypeNames.Float2)

        xf = UsdShade.Shader.Define(stage, f"{mat_path}/xf")
        xf.CreateIdAttr("UsdTransform2d")
        xf.CreateInput("in", Sdf.ValueTypeNames.Float2).ConnectToSource(st_out)
        xf.CreateInput("scale", Sdf.ValueTypeNames.Float2).Set(Gf.Vec2f(repeats, repeats))
        xf_out = xf.CreateOutput("result", Sdf.ValueTypeNames.Float2)

        tex = UsdShade.Shader.Define(stage, f"{mat_path}/tex")
        tex.CreateIdAttr("UsdUVTexture")
        from os.path import abspath
        tex.CreateInput("file", Sdf.ValueTypeNames.Asset).Set(Sdf.AssetPath(abspath(grid_png)))
        tex.CreateInput("st", Sdf.ValueTypeNames.Float2).ConnectToSource(xf_out)
        tex.CreateInput("wrapS", Sdf.ValueTypeNames.Token).Set("repeat")
        tex.CreateInput("wrapT", Sdf.ValueTypeNames.Token).Set("repeat")
        tex_rgb = tex.CreateOutput("rgb", Sdf.ValueTypeNames.Float3)

        pbr = UsdShade.Shader.Define(stage, f"{mat_path}/pbr")
        pbr.CreateIdAttr("UsdPreviewSurface")
        pbr.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).ConnectToSource(tex_rgb)
        pbr.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.6)
        pbr.CreateInput("specular",  Sdf.ValueTypeNames.Float).Set(0.0)
        surf = pbr.CreateOutput("surface", Sdf.ValueTypeNames.Token)
        mat.CreateSurfaceOutput().ConnectToSource(surf)

        UsdShade.MaterialBindingAPI(prim).Bind(mat)



    def _create_handle_rods(self, radius=0.006, color=(0.85, 0.85, 0.85)):
        """Create one render-only cylinder per env under /World/envs/env_i/Debug/HandleRod.
        The cylinder is authored in the **env-local** space (parented to the env's Debug xform)."""
        stage = omni.usd.get_context().get_stage()

        # shared PBR material
        mat_path = "/World/Materials/HandleRod"
        mat = UsdShade.Material.Define(stage, mat_path)
        sh  = UsdShade.Shader.Define(stage, f"{mat_path}/PBR")
        sh.CreateIdAttr("UsdPreviewSurface")
        sh.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*color))
        sh.CreateInput("roughness",    Sdf.ValueTypeNames.Float).Set(0.3)
        sh.CreateInput("specular",     Sdf.ValueTypeNames.Float).Set(0.0)
        mat.CreateSurfaceOutput().ConnectToSource(sh.CreateOutput("surface", Sdf.ValueTypeNames.Token))

        for env_i in range(self.scene.cfg.num_envs):
            parent_path = f"/World/envs/env_{env_i}/Debug"
            if not stage.GetPrimAtPath(parent_path):
                UsdGeom.Xform.Define(stage, parent_path)

            rod_path = f"{parent_path}/HandleRod"
            if not stage.GetPrimAtPath(rod_path):
                cyl = UsdGeom.Cylinder.Define(stage, rod_path)
                cyl.CreateAxisAttr().Set(UsdGeom.Tokens.z)          # length along +Z
                cyl.CreateRadiusAttr(float(radius))
                cyl.CreateHeightAttr(0.01)                          # will be set each frame
                UsdShade.MaterialBindingAPI(cyl.GetPrim()).Bind(mat)
                UsdGeom.Imageable(cyl.GetPrim()).CreatePurposeAttr().Set(UsdGeom.Tokens.render)

                # ensure there's exactly one transform op we can overwrite each frame
                xform = UsdGeom.Xformable(cyl.GetPrim())
                xform.ClearXformOpOrder()
                xform.AddTransformOp().Set(Gf.Matrix4d(1.0))

    def _update_handle_rods(self):
        """Update each rod in **env-local** space so multi-row layouts stay correct."""
        stage = omni.usd.get_context().get_stage()

        # pull latest physics state
        try:
            self._handle_1.update(self.physics_dt)
            self._handle_2.update(self.physics_dt)
        except Exception:
            pass

        # world-space handle positions for all envs
        pL_w = self._handle_1.data.root_state_w[:, 0:3]  # [N,3] torch
        pR_w = self._handle_2.data.root_state_w[:, 0:3]  # [N,3] torch
        origins = self.scene.env_origins                  # [N,3] torch

        for env_i in range(self.scene.cfg.num_envs):
            prim = stage.GetPrimAtPath(f"/World/envs/env_{env_i}/Debug/HandleRod")
            if not prim:
                continue

            # ---- compute in env-local space ----
            o = origins[env_i]                  # world origin of this env
            pL = (pL_w[env_i] - o).tolist()     # env-local
            pR = (pR_w[env_i] - o).tolist()     # env-local

            # guard against NaNs / degenerate geometry
            if not (np.isfinite(pL).all() and np.isfinite(pR).all()):
                continue

            v = Gf.Vec3d(pR[0]-pL[0], pR[1]-pL[1], pR[2]-pL[2])  # local direction
            L = float(v.GetLength())
            if not np.isfinite(L) or L < 1e-9:
                continue
            dir_ = v / L
            mid  = Gf.Vec3d(0.5*(pL[0]+pR[0]), 0.5*(pL[1]+pR[1]), 0.5*(pL[2]+pR[2]))

            # author cylinder params (height) and local transform
            cyl = UsdGeom.Cylinder(prim)
            cyl.CreateAxisAttr().Set(UsdGeom.Tokens.z)   # keep axis stable
            cyl.CreateHeightAttr(L)

            # rotate +Z -> dir_, then translate to midpoint (all in env-local frame)
            rot = Gf.Rotation(Gf.Vec3d(0, 0, 1), dir_)
            M_local = Gf.Matrix4d(1.0)
            M_local.SetRotate(rot)
            M_local.SetTranslateOnly(mid)

            xform = UsdGeom.Xformable(prim)
            xform.ClearXformOpOrder()
            xform.AddTransformOp().Set(M_local)
