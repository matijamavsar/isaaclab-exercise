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
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py --task DMP-Based-Cloth-Place-Decimated-Randomized --num_envs 64 --max_iterations 16000 --headless --enable_cameras
"""

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
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=1, env_spacing=2.5, replicate_physics=False)

    cloth = DeformableObjectCfg(
        prim_path="/World/envs/env_.*/cuboid",
        init_state=DeformableObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 0.356), rot=(0.5, 0.5, 0.5, 0.5)),
        spawn=sim_utils.UsdFileCfg(
            usd_path="/workspace/isaaclab/source/isaaclab_tasks/isaaclab_tasks/sim/deformable_cloth_simple_flattened.usd",
            # scale=(1, 1, 1),
        ),
    )

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

    cloth_plain = DeformableObjectCfg(
        prim_path="/World/envs/env_.*/cuboid/cuboid",
        init_state=DeformableObjectCfg.InitialStateCfg(pos=(0, 0, 0), rot=(1, 0, 0, 0)),
        spawn=None,
    )

    handle_1 = RigidObjectCfg(
        prim_path="/World/envs/env_.*/cuboid/handle_2",
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0, 0, 0), rot=(1, 0, 0, 0)),
        spawn=None,
    )

    handle_2 = RigidObjectCfg(
        prim_path="/World/envs/env_.*/cuboid/handle_1",
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0, 0, 0), rot=(1, 0, 0, 0)),
        spawn=None,
    )

    free_corner_1 = RigidObjectCfg(
        prim_path="/World/envs/env_.*/cuboid/free_corner_2",
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0, 0, 0), rot=(1, 0, 0, 0)),
        spawn=None,
    )

    free_corner_2 = RigidObjectCfg(
        prim_path="/World/envs/env_.*/cuboid/free_corner_1",
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0, 0, 0), rot=(1, 0, 0, 0)),
        spawn=None,
    )

    ### Robots
    robot_1 = FRANKA_PANDA_HIGH_PD_CFG.replace(
        prim_path="/World/envs/env_.*/Robot1",
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
            pos=(0.0, -0.6, 0.0),
            rot=(0.7071, 0, 0, 0.7071),
            # rot=(1.0, 0.0, 0.0, 0.0),
        ),)
    
    robot_2 = FRANKA_PANDA_HIGH_PD_CFG.replace(
        prim_path="/World/envs/env_.*/Robot2",
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
            pos=(0.0, 0.6, 0.0),
            rot=(0.7071, 0, 0, -0.7071),
            # rot=(1.0, 0.0, 0.0, 0.0),
        ),)

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
        "spread_reward": 
            Reward(True, 1.0),
        "height_reward": 
            Reward(True, 1.0),
        "corner_x_reward": 
            Reward(True, 5.0),
        "direction_reward": 
            Reward(True, 15.0),
        "action_penalty": 
            Reward(True, 1e-4),
    }

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
        self.z_abs_pos = torch.zeros((self.cfg.max_episode_length, self.num_envs), device=self.device)
        self.action_penalties = {'robot_1': torch.zeros((self.cfg.max_episode_length, self.num_envs), device=self.device),
                                 'robot_2': torch.zeros((self.cfg.max_episode_length, self.num_envs), device=self.device)}

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

        # Define the number of robots dynamically
        self.num_robots = 2

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

        self.dmp_integrator = BatchDMPIntegrator(N_basis=25, dof=2, device=self.device)
        self.corner_dmp_integrator = BatchDMPIntegrator(N_basis=25, dof=12, device=self.device)
        self.corner_traj = torch.zeros((self.num_envs, self.cfg.decimation, 6), device=self.device)
        self.corner_dmp_weights = torch.zeros((self.num_envs, 12, 25), device=self.device)
        self.corner_y0 = torch.zeros((self.num_envs, 12), device=self.device)
        self.corner_goal = torch.zeros((self.num_envs, 12), device=self.device)

    def log_pose_and_snapshot(self):
        self.frame_log_counter += 1

        # Only log every 20 steps
        if self.frame_log_counter % 20 != 0:
            return

        # Get absolute poses of robots
        absolute_poses = self._get_absolute_pose()  # shape: (num_envs, 7) => [x, y, z, qw, qx, qy, qz]

        # Store current pose for env 0
        if "absolute_pose_history" not in self.__dict__:
            self.absolute_pose_history = []
        self.absolute_pose_history.append(absolute_poses.clone())

        # Convert to tensor of shape (T, num_envs, 7)
        traj = torch.stack(self.absolute_pose_history, dim=0)  # (T, B, 7)

        _, axs = plt.subplots(2, 1, figsize=(8, 10))

        # ---- First row: pose trace (x vs z)
        x_plot = traj[:, 0, 0].cpu().numpy()  # env 0, x over time
        z_plot = traj[:, 0, 2].cpu().numpy()  # env 0, z over time
        axs[0].plot(x_plot, z_plot, marker='o', linestyle='-', color='tab:blue')
        axs[0].set_title("Absolute Pose Trajectory (Env 0) — X vs Z")
        axs[0].set_xlabel("X")
        axs[0].set_ylabel("Z")
        axs[0].grid(True)
        axs[0].axis('equal')

        # ---- Second row: RGB snapshot from camera
        if hasattr(self, 'camera') and self.camera.data.output["rgb"] is not None:
            img = self._camera.data.output['rgb'][0]
            # img = img[:, :, :3] / 255.0  # (H, W, 3)
            axs[1].imshow(img.cpu())
            axs[1].axis('off')
            axs[1].set_title("Camera Snapshot (Env 0)")
        
        # Save plot
        os.makedirs('./logs/snapshots', exist_ok=True)
        fname = f"./logs/snapshots/step_{self.frame_log_counter:04d}.png"
        plt.tight_layout()
        plt.savefig(fname)
        plt.close()

    def log_trajectory_panel(self):
        if not hasattr(self, "absolute_pose_buffer"):
            self.absolute_pose_buffer = []
            self.rgb_buffer = []
            self.frame_log_counter = 0

        num_subplots = 6
        log_step = 360 // num_subplots
        self.frame_log_counter += 1

        # Record absolute pose
        abs_pose = self._get_absolute_pose().clone()  # (num_envs, 7)
        self.absolute_pose_buffer.append(abs_pose[0].cpu())  # Only log env 0

        if (self.frame_log_counter % log_step != 0) or (self.episode_length_buf.sum() == 0):
            return

        # Record RGB snapshot
        if hasattr(self, '_camera') and self._camera.data.output["rgb"] is not None:
            rgb_img = self._camera.data.output['rgb'][0] / 255
            self.rgb_buffer.append(rgb_img)

        # Wait until we have 20 samples
        if len(self.rgb_buffer) < num_subplots:
            return

        # Make the panel figure (20 in a row)
        fig, axs = plt.subplots(2, num_subplots, figsize=(3.5 * num_subplots, 2 * 2.5))  # 2 rows, 20 columns

        for i in range(num_subplots):
            # Pose subplot
            x = [pose[0].item() for pose in self.absolute_pose_buffer[:log_step*i+1]]
            z = [pose[2].item() for pose in self.absolute_pose_buffer[:log_step*i+1]]
            if i==0:
                axs[0, i].plot(x, z, marker='o', linestyle='-', color='tab:blue')
            else:
                axs[0, i].plot(x, z, linestyle='-', color='tab:purple')
            axs[0, i].set_title("Absolute Pose Trajectory (Env 0)")
            axs[0, i].set_xlim(-0.6, 0.6)
            axs[0, i].set_ylim(0.0, 1.0)
            axs[0, i].set_xlabel("x [m]")
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

    def _abs_to_arm_poses(self, abs_pose: torch.Tensor, rel_pose: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Convert absolute + relative pose to individual poses for two arms.
        
        Args:
            abs_pose (torch.Tensor): (N, 7) absolute pose [x, y, z, qw, qx, qy, qz]
            rel_pose (torch.Tensor): (N, 7) relative pose [dx, dy, dz, qw, qx, qy, qz]
                                    dx, dy, dz are relative positions (from mid to left/right).
                                    The quaternion is the relative rotation from left to right.

        Returns:
            left_pose, right_pose: each (N, 7), pose for each arm
        """
        assert abs_pose.shape[-1] == 7 and rel_pose.shape[-1] == 7, "Poses must be in (N, 7) format"

        # Extract absolute position and quaternion
        abs_pos = abs_pose[:, :3]
        abs_quat = abs_pose[:, 3:]

        # Get half of the relative offset vector
        rel_offset = rel_pose[:, :3]  # (N, 3)
        offset = 0.5 * rel_offset     # (N, 3)

        # Compute left and right positions
        left_pos = abs_pos + offset
        right_pos = abs_pos - offset

        # Get relative rotation from left to right
        rel_quat = rel_pose[:, 3:]  # (N, 4)

        # Compute rotation from absolute to left/right (halfway split)
        # q_abs = slerp(q_left, q_right, 0.5)
        # We assume rel_quat = q_right * q_left_inv => q_left = q_abs * inv(sqrt(rel_quat))
        half_rel_quat = torch.zeros_like(rel_quat)
        for i, quat in enumerate(rel_quat):
            half_rel_quat[i] = quat_slerp(torch.tensor([1.0, 0.0, 0.0, 0.0], 
                                                       device=quat.device), quat, 0.5)

        # Get q_left and q_right from q_abs and half_rel_quat
        left_quat = quat_mul(abs_quat, half_rel_quat)
        right_quat = quat_mul(abs_quat, quat_inv(half_rel_quat))

        # Compose final poses
        left_pose = torch.cat((left_pos, left_quat), dim=-1)
        right_pose = torch.cat((right_pos, right_quat), dim=-1)

        return left_pose, right_pose

    def _setup_scene(self):
        # ✅ Create robots
        self._robot_1 = Articulation(self.cfg.robot_1)
        self._robot_2 = Articulation(self.cfg.robot_2)
        self.cloth_lengths = torch.zeros(self.scene.cfg.num_envs, device=self.device)

        self.scene.articulations["robot_1"] = self._robot_1
        self.scene.articulations["robot_2"] = self._robot_2

        # ✅ Configure terrain
        self.cfg.terrain.num_envs = self.scene.cfg.num_envs
        self.cfg.terrain.env_spacing = self.scene.cfg.env_spacing
        self._terrain = self.cfg.terrain.class_type(self.cfg.terrain)

        self._cloth = self.cfg.cloth.class_type(self.cfg.cloth)
        self._cloth_plain = self.cfg.cloth_plain.class_type(self.cfg.cloth_plain)
        self._handle_1 = self.cfg.handle_1.class_type(self.cfg.handle_1)
        self._handle_2 = self.cfg.handle_2.class_type(self.cfg.handle_2)
        self._free_corner_1 = self.cfg.free_corner_1.class_type(self.cfg.free_corner_1)
        self._free_corner_2 = self.cfg.free_corner_2.class_type(self.cfg.free_corner_2)

        # ✅ Clone environments
        self.scene.clone_environments(copy_from_source=True)

        for env_idx in range(self.scene.cfg.num_envs):
            # Get the Cloth prim for the current environment
            cloth_prim_path = f"/World/envs/env_{env_idx}/cuboid/cuboid"
            cloth_prim = get_prim_at_path(cloth_prim_path)

            # Generate random scale factors for x, y, z axes
            random_scale = torch.tensor([
                0.7,
                0.7, # torch.rand(1).item() * (0.7 - 0.6) + 0.6,  # Scale y-axis
                0.005,
            ])

            # Apply the random scale to the Cloth prim
            cloth_prim.GetAttribute("xformOp:scale").Set(Gf.Vec3f(*random_scale.tolist()))
            # cloth_prim.GetAttribute('xformOp:translate').Set(
            #     Gf.Vec3f(0.0, 0.0, 0.356 + ((1 - random_scale[1]) * 0.356).item()))
            self.cloth_lengths[env_idx] = random_scale[1].item() * 0.7

            # TODO: randomize also other parameters (drag, friction etc)

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
            visual_plane.GetDisplayColorAttr().Set([Gf.Vec3f(0.00, 0.00, 0.02)])  # Green

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

    def _get_absolute_pose(self):
        absolute_pose = torch.zeros((self.num_envs, 7), device=self.device)
        ee_pose_1_w = self._robot_1.data.body_state_w[
            :, self.robot_entity_cfg['robot_1'].body_ids[0], 0:7
        ]
        ee_pose_2_w = self._robot_2.data.body_state_w[
            :, self.robot_entity_cfg['robot_2'].body_ids[0], 0:7
        ]
        for i in range(self.num_envs):
            absolute_pose[i,3:] = quat_slerp(ee_pose_1_w[i,3:], ee_pose_2_w[i,3:], 0.5)
        absolute_pose[:,0:3] = (ee_pose_1_w[:,0:3] + ee_pose_2_w[:,0:3]) / 2
        return absolute_pose

    # pre-physics step calls
    def _pre_physics_step(self, actions: torch.Tensor):

        print("Pre-physics step")

        dmp_parameters = actions.clone()
        dmp_parameters[:,2] = dmp_parameters[:,2].clamp(-1.0, 1.0)
        dmp_parameters[:,3] = dmp_parameters[:,3].clamp(0.0, 1.0)

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

        t, y, dy, ddy = self.dmp_integrator.step()
        self.dmp_integrator.x[t >= self.actions[:,-1]] = 0.1353

        current_absolute_pose = self._get_absolute_pose()
        self.ee_distances[self.action_steps] = self._get_ee_distance()
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

                ### Make camera move in a circle
                # self._camera.set_world_poses_from_view(self.circle_camera_path[self.render_count].unsqueeze(0),
                #                                        torch.tensor([0,0,0.5], device=self.device))

                ### Set camera to a fixed position
                # self._camera.set_world_poses_from_view(torch.tensor([1.0, 1.0, 1.0], device=self.device).unsqueeze(0),
                #                                        torch.tensor([0.0, 0.0, 0.2], device=self.device))

                if self.render_count % self.cfg.sim.render_interval == 0:
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
                    joint_vel = torch.zeros_like(self.robots[robot_key].data.default_joint_vel.clone())
                    self.robots[robot_key].write_joint_state_to_sim(
                        self.robot_dof_targets[robot_key], joint_vel)
                else:
                    self.robots[robot_key].set_joint_position_target(
                        self.robot_dof_targets[robot_key])

            # get free current corners, flatten them
            corners = self._get_corners()[:, [2, 3]] - self.scene.env_origins.unsqueeze(1)  # (N,2,3)
            corners_flat = corners.reshape(self.num_envs, -1)
            self.prev_corners_buf = self.prev_corners_buf.roll(-1, dims=0)
            self.prev_corners_buf[-1] = corners_flat

            self.prev_abs_buf = self.prev_abs_buf.roll(-1, dims=0)
            self.prev_abs_buf[-1] = current_absolute_pose[:, [0, 2]] - self.scene.env_origins[:, [0, 2]]

            if (self.action_steps == self.cfg.max_episode_length - 1).all():
                self.probing_done = True

        elif (self.action_steps < self.cfg.max_episode_length - 60).all():
            self.inverse_kinematics(y)
            for robot_key in self.robots.keys():
                # ✅ Set joint position target separately for each robot
                if self.cfg.write_joint_state:
                    joint_vel = self.robots[robot_key].data.default_joint_vel.clone()
                    self.robots[robot_key].write_joint_state_to_sim(
                        self.robot_dof_targets[robot_key], joint_vel)
                else:
                    self.robots[robot_key].set_joint_position_target(
                        self.robot_dof_targets[robot_key])
                
                self.target_joints_buffer[self.action_steps[0]] = self.robot_dof_targets[robot_key][:,0:7]
                self.actual_joints_buffer[self.action_steps[0]] = self.robots[robot_key].data.joint_pos[:,0:7]

        self.iteration_step += 1
        self.action_steps += 1

        if self.cfg.plot_trajectories:
            self.log_trajectory_panel()

    def inverse_kinematics(self, y):
        current_absolute_pose = self._get_absolute_pose()
        target_absolute_pose = current_absolute_pose.clone() * 0.0
        target_absolute_pose[:,0] = y[:,0]
        target_absolute_pose[:,2] = y[:,1]
        target_absolute_pose[:,0:3] += self.scene.env_origins
        
        target_absolute_pose[:, 3:] = torch.tensor([0, 1, 0, 0], device=self.device)
        
        # Fixed Y-distance between grippers
        fixed_rel_pos = torch.tensor([0.0, 0.66, 0.0], device=self.device).repeat(self.num_envs, 1)
        fixed_rel_quat = torch.tensor([0, 0.0, 0.0, 1], device=self.device).repeat(self.num_envs, 1)
        relative_pose = torch.cat((fixed_rel_pos, fixed_rel_quat), dim=-1)

        # Transform from global into each robot coordinate system
        larm_pose, rarm_pose = self._abs_to_arm_poses(target_absolute_pose, relative_pose)

        root_pose_w = self._robot_2.data.root_state_w[:, 0:7]
        larm_pos_b, _ = subtract_frame_transforms(
            root_pose_w[:, 0:3], root_pose_w[:, 3:7], 
            larm_pose[:, 0:3], larm_pose[:, 3:7]
        )
        root_pose_w = self._robot_1.data.root_state_w[:, 0:7]
        rarm_pos_b, _ = subtract_frame_transforms(
            root_pose_w[:, 0:3], root_pose_w[:, 3:7], 
            rarm_pose[:, 0:3], rarm_pose[:, 3:7]
        )

        rot_around_x = self.prev_rot_around_x + self.actions[:, 2]
        rot_around_x = rot_around_x.clamp(-0.6, 0.6)

        rarm_goal_quat_b = torch.zeros((self.num_envs, 4), device=self.device)
        larm_goal_quat_b = torch.zeros((self.num_envs, 4), device=self.device)
        rarm_goal_quat_b[:,1] = 1.0
        larm_goal_quat_b[:,1] = 1.0

        rarm_pose_b = torch.cat((rarm_pos_b, rarm_goal_quat_b), dim=-1)
        larm_pose_b = torch.cat((larm_pos_b, larm_goal_quat_b), dim=-1)

        self._ik_controller['robot_1'].set_command(rarm_pose_b)
        self._ik_controller['robot_2'].set_command(larm_pose_b)
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
            self.last_joint_change = self.robot_dof_speed_scales[robot_key] * self.dt * (
                self.robot_dof_targets[robot_key] - padded_joint_pos) * self.cfg.action_scale
            self.action_penalties[robot_key][self.action_steps] = self.last_joint_change.abs().mean(dim=-1)

            ### Limit change of joints
            # new_targets = (padded_joint_pos + self.last_joint_change)
            ### Or just send it to target positions
            new_targets = self.robot_dof_targets[robot_key]

            # ✅ Clamp values within each robot's DOF limits
            new_targets = torch.clamp(new_targets, self.robot_dof_lower_limits[robot_key], self.robot_dof_upper_limits[robot_key])
            self.prev_joints[robot_key] = self.prev_joints[robot_key].roll(-1, dims=0)
            self.prev_joints[robot_key][-1] = new_targets[:, 0:7]

            self.filtered_dof_targets[robot_key] = new_targets
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
        ee_2_pos = self.robots[robot_keys[1]].data.body_pos_w[:, self.left_finger_link_idx[robot_keys[1]]]

        ee_distance = torch.norm(ee_1_pos - ee_2_pos, dim=-1)

        return ee_distance

    def _get_corners(self):
        corners = [self._handle_1.root_physx_view.get_transforms()[:,0:3],
                   self._handle_2.root_physx_view.get_transforms()[:,0:3],
                   self._free_corner_1.root_physx_view.get_transforms()[:,0:3],
                   self._free_corner_2.root_physx_view.get_transforms()[:,0:3]]
        corners = torch.stack(corners).permute(1,0,2)
        return corners

    def compute_rewards(self):
        """
        Compute individual reward term values and return a dict of reward_name -> value tensor.
        """
        # Gather data
        corners = self._get_corners()
        cloth_positions = self._cloth_plain.root_physx_view.get_nodal_positions().reshape(self.num_envs, -1, 3)
        mean_cloth_height = cloth_positions[:, :, 2].mean(dim=-1)

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
        # rewards["corner_x_reward"] = (free_x.mean(-1) + grasped_x.mean(-1)) * self._rewards["corner_x_reward"].scale

        if not self.cfg.use_weighted_atan_rewards and not self.cfg.use_weighted_exp_rewards:
            rewards["height_reward"] = (1.0 / (1.0 + mean_cloth_height)) * self._rewards["height_reward"].scale
            rewards["spread_reward"] = (pairwise_sum / count) * self._rewards["spread_reward"].scale
            rewards["corner_x_reward"] = (free_x.mean(-1)) * self._rewards["corner_x_reward"].scale
            rewards["direction_reward"] = (free_x.mean(-1) - grasped_x.mean(-1)) * self._rewards["direction_reward"].scale
        elif self.cfg.use_weighted_atan_rewards:
            rewards["height_reward"] = 0.5 + (1.0 / torch.pi) * torch.atan(1.0 / (1.0 + mean_cloth_height))
            rewards["spread_reward"] = 0.5 + (1.0 / torch.pi) * torch.atan(pairwise_sum / count)
            rewards["corner_x_reward"] = 0.5 + (1.0 / torch.pi) * torch.atan(free_x.mean(-1))
            rewards["direction_reward"] = 0.5 + (1.0 / torch.pi) * torch.atan(free_x.mean(-1) - grasped_x.mean(-1))
        elif self.cfg.use_weighted_exp_rewards:
            rewards["height_reward"] = torch.exp(1.0 / (1.0 + mean_cloth_height))
            rewards["spread_reward"] = torch.exp(pairwise_sum / count)
            rewards["corner_x_reward"] = torch.exp(free_x.mean(-1))
            rewards["direction_reward"] = torch.exp(free_x.mean(-1) - grasped_x.mean(-1))

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
        invalid_env = (torch.bitwise_or(self.ee_distances > 0.8, self.ee_distances < 0.5)).any(0)
        invalid_env = (torch.bitwise_or(invalid_env, self.z_abs_pos < 0.15)).any(0)
        total_reward[invalid_env] = 0

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

        # Change some material parameters for the cloth
        # params = torch.zeros((self.num_envs, 4), device=self.device)
        # params[:, 0] = torch.rand(self.num_envs) * (0.75 - 0.25) + 0.25
        # params[:, 1] = torch.rand(self.num_envs) * (0.75 - 0.25) + 0.25
        # params[:, 2] = torch.rand(self.num_envs) * (0.49 - 0.10) + 0.10
        # params[:, 3] = torch.rand(self.num_envs) * (1000000 - 1000) + 1000
        # self._cloth_plain.material_physx_view.set_damping_scale(params, env_ids)
        # self._cloth_plain.material_physx_view.set_dynamic_friction(params, env_ids)
        # self._cloth_plain.material_physx_view.set_youngs_modulus(params, env_ids)
        # self._cloth_plain.material_physx_view.set_poissons_ratio(params, env_ids)

        if self.default_states is not None:
            self._cloth_plain.write_nodal_state_to_sim(
                self.default_states['_cloth_plain'][env_ids], env_ids=env_ids)

            self._handle_1.write_root_state_to_sim(
                self.default_states['_handle_1'][env_ids], env_ids=env_ids)

            self._handle_2.write_root_state_to_sim(
                self.default_states['_handle_2'][env_ids], env_ids=env_ids)

            self._free_corner_1.write_root_state_to_sim(
                self.default_states['_free_corner_1'][env_ids], env_ids=env_ids)

            self._free_corner_2.write_root_state_to_sim(
                self.default_states['_free_corner_2'][env_ids], env_ids=env_ids)

        self.sim.pause()
        if 0: # self.reset_count % 100 == 0:
            random_scale = torch.zeros((self.num_envs, 3), device=self.device)
            random_scale[:, 0] = 1.0
            random_scale[:, 1] = torch.rand(self.num_envs) * (1.0 - 0.8) + 0.8
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
            self.default_states['_handle_2'] = self._handle_2.data.root_state_w.clone()
            self.default_states['_handle_2'][:,7:] = 0.0
            self.default_states['_free_corner_1'] = self._free_corner_1.data.root_state_w.clone()
            self.default_states['_free_corner_1'][:,7:] = 0.0
            self.default_states['_free_corner_2'] = self._free_corner_2.data.root_state_w.clone()
            self.default_states['_free_corner_2'][:,7:] = 0.0
            self._initial_corners = torch.stack((
                self.default_states['_handle_1'][:,0:3],
                self.default_states['_handle_2'][:,0:3],
                self.default_states['_free_corner_1'][:,0:3],
                self.default_states['_free_corner_2'][:,0:3]
            ), axis=1)

        # Create joints
        if not self.joints_created:
            self.create_joints(env_ids)
            self.joints_created = True

        for robot_key in self.robots.keys():  # Loop through all robots dynamically
            robot = self.robots[robot_key]
            joint_pos = robot.data.default_joint_pos[env_ids].clone()
            joint_vel = robot.data.default_joint_vel[env_ids].clone()
            robot.set_joint_position_target(joint_pos, env_ids=env_ids)
            robot.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)

        self._cloth_plain.write_nodal_state_to_sim(
            self.default_states['_cloth_plain'][env_ids], env_ids=env_ids)

        self._handle_1.write_root_state_to_sim(
            self.default_states['_handle_1'][env_ids], env_ids=env_ids)

        self._handle_2.write_root_state_to_sim(
            self.default_states['_handle_2'][env_ids], env_ids=env_ids)

        self._free_corner_1.write_root_state_to_sim(
            self.default_states['_free_corner_1'][env_ids], env_ids=env_ids)

        self._free_corner_2.write_root_state_to_sim(
            self.default_states['_free_corner_2'][env_ids], env_ids=env_ids)

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
            cube_1_path = Sdf.Path("/World/envs/env_" + str(int(idx)) + "/cuboid/handle_2")
            cube_2_path = Sdf.Path("/World/envs/env_" + str(int(idx)) + "/cuboid/handle_1")
            panda_1_finger_path = Sdf.Path("/World/envs/env_" + str(int(idx)) + "/Robot1/panda_leftfinger")
            panda_2_finger_path = Sdf.Path("/World/envs/env_" + str(int(idx)) + "/Robot2/panda_rightfinger")
            joint_1_path = panda_1_finger_path.AppendElementString("fixedJoint")
            joint_2_path = panda_2_finger_path.AppendElementString("fixedJoint")
            fixedJoint_1 = UsdPhysics.FixedJoint.Define(self.scene.stage, joint_1_path)
            fixedJoint_1.CreateBody0Rel().SetTargets([panda_1_finger_path])
            fixedJoint_1.CreateBody1Rel().SetTargets([cube_1_path])
            fixedJoint_1.CreateLocalPos0Attr().Set(Gf.Vec3f(0,0,0))
            fixedJoint_1.CreateLocalRot0Attr().Set(Gf.Quatf(1,0,0,0))
            fixedJoint_1.CreateLocalPos1Attr().Set(Gf.Vec3f(0,0,0))
            fixedJoint_1.CreateLocalRot1Attr().Set(Gf.Quatf(1,0,0,0))

            fixedJoint_2 = UsdPhysics.FixedJoint.Define(self.scene.stage, joint_2_path)
            fixedJoint_2.CreateBody0Rel().SetTargets([panda_2_finger_path])
            fixedJoint_2.CreateBody1Rel().SetTargets([cube_2_path])
            fixedJoint_2.CreateLocalPos0Attr().Set(Gf.Vec3f(0,0,0))
            fixedJoint_2.CreateLocalRot0Attr().Set(Gf.Quatf(1,0,0,0))
            fixedJoint_2.CreateLocalPos1Attr().Set(Gf.Vec3f(0,0,0))
            fixedJoint_2.CreateLocalRot1Attr().Set(Gf.Quatf(1,0,0,0))

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

        ### Either pass the action history and corner trajectory observations
        # observations = torch.cat((act_hist, corner_traj_obs), dim=-1)
        ### Or action history and cloth lengths
        # observations = torch.cat((joint_obs, self.cloth_lengths.unsqueeze(-1)), dim=-1)
        ### Or current absolute pose and previous actions and corner trajectories
        observations = torch.cat((joint_obs, act_hist, corner_traj_obs), dim=-1)

        return {"policy": observations}
