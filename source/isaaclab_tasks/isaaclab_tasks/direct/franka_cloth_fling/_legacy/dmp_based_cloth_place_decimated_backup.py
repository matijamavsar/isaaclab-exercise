# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
from __future__ import annotations

import torch
import os

from pxr import UsdGeom, UsdShade, Sdf, Gf
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
from isaaclab.utils.math import subtract_frame_transforms, matrix_from_quat
from isaaclab.utils.math import quat_slerp, quat_mul, quat_inv
from isaaclab.sensors import CameraCfg
from pxr import UsdPhysics, UsdGeom, Gf, Sdf
import omni.usd
import omni.kit.commands
from isaaclab.controllers import DifferentialIKController, DifferentialIKControllerCfg
from isaacsim.core.utils.prims import get_prim_at_path
import torch.nn.functional as F
from isaaclab_assets import FRANKA_PANDA_HIGH_PD_CFG
import numpy as np

from .dmp_integrator import BatchDMPIntegrator

""" Run this training using
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py --task Bimanual-Imitation-Franka-Place-Cloth-Direct-v0 --num_envs 64 --max_iterations 16000 --headless --video --video_interval 8000
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
    decimation = max_episode_length
    action_space = 55
    observation_space = 48
    state_space = 0
    enable_camera_recording = True

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
            gpu_max_particle_contacts = 2**22, # Default is 2**20
            gpu_max_soft_body_contacts = 2**23,
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
            usd_path="/workspace/isaaclab/source/isaaclab_tasks/isaaclab_tasks/sim/deformable_for_joint_noCCD_freeCorners.usd",
            # usd_path="/home/matija/isaaclab-fork/source/isaaclab_tasks/isaaclab_tasks/sim/deformable_for_joint_noCCD_freeCorners.usd",
            scale=(1, 1, 0.5),
        ),
            )

    camera = CameraCfg(
        prim_path="/World/Camera",
        offset=CameraCfg.OffsetCfg(pos=(-5.0, -5.0, 3.0), rot=( 0.9020, -0.0828,  0.2000,  0.3736), convention="world"),
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=24.0, focus_distance=400.0, horizontal_aperture=20.955, clipping_range=(0.1, 20.0)
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
    use_simple_imitation_reward = True

    rewards = {
        "spread_reward": 
            Reward(True, 1.0),
        "height_reward": 
            Reward(True, 1.0),
        "corner_x_reward": 
            Reward(True, 5.0),
        "gripper_downward_reward": 
            Reward(False, 1e-3),
        "direction_reward": 
            Reward(True, 5.0),
        "action_penalty": 
            Reward(False, 1e-4),
    }


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

        # Setup video recording
        now = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.video_folder = os.path.join("./logs/videos", now)
        os.makedirs(self.video_folder, exist_ok=True)        
        self.video_writer = None
        self.render_count = 0
        self.action_steps = torch.zeros(self.num_envs, device=self.device).int()

        self.dt = self.cfg.sim.dt * self.cfg.decimation
        self.iteration_step = torch.tensor(0, device=self.device)
        self.default_states = None
        self.joints_created = False
        self.dmp_initialized = False

        L = 0.7
        self.ideal_spread_reward = L * (2.0 + np.sqrt(2.0)) / 3.0
        self.ideal_spread_reward = torch.tensor(self.ideal_spread_reward, device=self.device)

        self.robot_entity_cfg = {}
        self._rewards = self.cfg.rewards

        diff_ik_cfg = DifferentialIKControllerCfg(
            command_type="pose", use_relative_mode=False, ik_method="pinv")
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

        self.robot_dof_lower_limits = {}  # Store lower joint limits
        self.robot_dof_upper_limits = {}  # Store upper joint limits
        self.robot_dof_speed_scales = {}  # Store joint speed scales

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

    def _setup_camera_writer(self, step):
        """
        Called when we enter a recording window to create a new VideoWriter.
        We generate a filename that includes the block index (e.g. “record_0.avi”, “record_1.avi”, …).
        """
        block_index = step // 10000
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
        dmp_parameters = actions.clone()
        dmp_parameters[:,2] = dmp_parameters[:,2].clamp(-0.5, 1.0) # Goal X
        dmp_parameters[:,3] = dmp_parameters[:,3].clamp(0.1, 0.8) # Goal Z

        current_absolute_pose = self._get_absolute_pose()
        current_absolute_pose[:,0:3] -= self.scene.env_origins
        dmp_parameters[:,0] = current_absolute_pose[:,0]
        dmp_parameters[:,1] = current_absolute_pose[:,2]
        dmp_tau = dmp_parameters[:,-1] * 0 + self.cfg.episode_length_s

        # reset_dmp_indices = (self.episode_length_buf % (self.max_episode_length // 2)) == 0
        reset_dmp_indices = self.episode_length_buf == 0
        reset_dmp_indices = torch.where(reset_dmp_indices == True)[0]

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

        _t = time.time()
        ee_jacobi_idx = 7
        t, y, dy, ddy = self.dmp_integrator.step()

        self.ee_distances[self.action_steps] = self._get_ee_distance()

        if self.cfg.enable_camera_recording:
            in_window = (self.iteration_step % 10000) < 720
            just_entered = (in_window and self.video_writer is None)

            if just_entered:
                self.render_count = 0
                self._setup_camera_writer(self.iteration_step)

            # 4) If we are in the window, capture & write one frame:
            if in_window and self.video_writer is not None:
                if self.render_count % self.cfg.sim.render_interval == 0:
                    self._capture_and_write_frame()
                self.render_count += 1

            # 5) If we just left a window, close the writer:
            just_left = ((not in_window) and self.video_writer is not None)
            if just_left:
                self.render_count = 0
                self._close_camera_writer()

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
        larm_pos_b, larm_quat_b = subtract_frame_transforms(
            root_pose_w[:, 0:3], root_pose_w[:, 3:7], 
            larm_pose[:, 0:3], larm_pose[:, 3:7]
        )
        root_pose_w = self._robot_1.data.root_state_w[:, 0:7]
        rarm_pos_b, rarm_quat_b = subtract_frame_transforms(
            root_pose_w[:, 0:3], root_pose_w[:, 3:7], 
            rarm_pose[:, 0:3], rarm_pose[:, 3:7]
        )

        rot_around_x = self.prev_rot_around_x + self.actions[:, 2]
        rot_around_x = rot_around_x.clamp(-0.6, 0.6)
        half_angles = rot_around_x / 2.0
        cos_half = torch.cos(half_angles)
        sin_half = torch.sin(half_angles)

        # Construct quaternion representing rotation around X axis
        # [qw, qx, qy, qz] = [cos(θ/2), sin(θ/2), 0, 0]
        rot_around_x_quat_1 = torch.stack([cos_half, sin_half, 
            torch.zeros_like(cos_half), torch.zeros_like(cos_half)], dim=-1)
        rot_around_x_quat_2 = torch.stack([cos_half, -sin_half, 
            torch.zeros_like(cos_half), torch.zeros_like(cos_half)], dim=-1)

        # rarm_goal_quat_b = quat_mul(rarm_quat_b, rot_around_x_quat_1)
        # larm_goal_quat_b = quat_mul(larm_quat_b, rot_around_x_quat_2)
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
                :, robot_entity_cfg.body_ids[0], 0:7
            ]

            # Obtain robot's Jacobian matrix
            jacobian_w = robot.root_physx_view.get_jacobians()[
                :, ee_jacobi_idx, :, 
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

            ### Limit change of joints
            # new_targets = (padded_joint_pos + self.last_joint_change)
            ### Or
            new_targets = self.robot_dof_targets[robot_key]

            # ✅ Clamp values within each robot's DOF limits
            new_targets = torch.clamp(new_targets, self.robot_dof_lower_limits[robot_key], self.robot_dof_upper_limits[robot_key])
            self.prev_joints[robot_key] = self.prev_joints[robot_key].roll(-1, dims=0)
            self.prev_joints[robot_key][-1] = new_targets[:, 0:7]

            self.filtered_dof_targets[robot_key] = new_targets
            self.robot_dof_targets[robot_key] = self.filtered_dof_targets[robot_key]

        for robot_key in self.robots.keys():
            # ✅ Set joint position target separately for each robot
            self.robots[robot_key].set_joint_position_target(
                self.robot_dof_targets[robot_key])

        self.iteration_step += 1
        self.action_steps += 1

    # post-physics step calls
    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        # Calculate distance between grippers
        terminated = 0
        truncated = self.episode_length_buf >= self.max_episode_length - 1

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

    def _get_rewards(self) -> torch.Tensor:
        """Compute rewards for lifting the cloth with stable grasping."""

        corners = self._get_corners()
        print("Calculating rewards")

        # Try to keep height low
        cloth_positions = self._cloth_plain.root_physx_view.get_nodal_positions().reshape(self.num_envs, -1, 3)
        mean_cloth_height = cloth_positions[:, :, 2].mean(dim=-1)
        self._rewards["height_reward"].value = 1/(1 + mean_cloth_height) * self._rewards["height_reward"].scale

        pairwise_sum = torch.zeros(self.num_envs, device=self.device)
        count = 0.0
        # Loop over each pair (there are 6 pairs for 4 corners)
        for i in range(4):
            for j in range(i + 1, 4):
                pairwise_sum += torch.norm(corners[:, i] - corners[:, j], dim=-1)
                count += 1.0
        r_spread = pairwise_sum / count

        ### Old spread reward
        self._rewards["spread_reward"].value = self._rewards["spread_reward"].scale * r_spread
        ### Or new one
        # sigma = 0.1 * self.ideal_spread_reward  # choose σ as e.g. 10% of ideal
        # reward_spread = torch.exp(-0.5 * ((r_spread - self.ideal_spread_reward)/sigma)**2)
        # self._rewards["spread_reward"].value = self._rewards["spread_reward"].scale * reward_spread

        # Define grasped and free corners (in this example, grasped corners are 0 and 1, free are 2 and 3)
        free_corners_x = corners[:, [2, 3], 0] - self.scene.env_origins[:,0].unsqueeze(1)
        grasped_corners_x = corners[:, [0, 1], 0] - self.scene.env_origins[:,0].unsqueeze(1)

        # ✅ New reward: Encourage free corners to move outward along X axis
        self._rewards["corner_x_reward"].value = torch.atan(free_corners_x.mean(-1) + 
            grasped_corners_x.mean(-1)) * self._rewards["corner_x_reward"].scale

        # Keep X of free corners higher
        self._rewards["direction_reward"].value = torch.atan(free_corners_x.mean(-1) - grasped_corners_x.mean(-1))
        self._rewards["direction_reward"].value *= self._rewards["direction_reward"].scale

        # Alignment reward (encourage the gripper to point down)
        self._rewards["gripper_downward_reward"].value = torch.zeros(self.num_envs, device=self.device)
        for i, robot_key in enumerate(self.robots.keys()):
            gripper_quat = self.robots[robot_key].data.body_quat_w[:, self.hand_link_idx[robot_key]]
            gripper_rot = matrix_from_quat(gripper_quat)  # Assuming matrix_from_quat is correctly implemented

            gripper_z_axis = gripper_rot[:, :, 2]  # Z axis direction of the gripper (num_envs, 3)
            world_down = torch.tensor([0, 0, -1], device=self.device, dtype=torch.float32).repeat(self.num_envs, 1)
            alignment_reward = torch.sum(gripper_z_axis * world_down, dim=-1)  # Dot product to align downward

            self._rewards["gripper_downward_reward"].value += alignment_reward * self._rewards["gripper_downward_reward"].scale

        ### Penalize joint changes
        self._rewards["action_penalty"].value = torch.sum(
            self.last_joint_change**2, dim=-1) * self._rewards["action_penalty"].scale

        ### Or penalize joint velocities
        # self._rewards["action_penalty"].value = torch.zeros(self.num_envs, device=self.device)
        # for robot_key in self.robots.keys():
        #     self._rewards["action_penalty"].value += torch.sum(
        #         self.robots[robot_key].data.joint_vel**2, dim=(-1,-2)) * self.cfg.action_penalty_scale

        # ✅ Compute Total Reward
        total_reward = torch.zeros(self.num_envs, device=self.device)

        for reward_key in self._rewards.keys():
            if self._rewards[reward_key].use: # If the reward is enabled
                if 'penalty' in reward_key:
                    total_reward -= self._rewards[reward_key].value / 1
                else:
                    total_reward += self._rewards[reward_key].value / 1

        ### If EE distances are bad in any time step, set that env's reward to -1
        invalid_env = (torch.bitwise_or(self.ee_distances > 1.0, self.ee_distances < 0.4)).any(0)
        total_reward[invalid_env] = 0

        # ✅ Log for Debugging
        self.extras["log"] = {n: v.value.mean() for n, v in self._rewards.items()}

        return total_reward

    def _reset_idx(self, env_ids: torch.Tensor | None):
        """Reset the state of multiple robots and environment objects properly."""
        super()._reset_idx(env_ids)

        # if len(self.filtered_trajectories['robot_1']) > 0:
        #     self._plot_comparison_dof_targets()
        #     breakpoint()

        self.action_steps[env_ids] = 0

        for robot_key in self.robots.keys():  # Loop through all robots dynamically
            robot = self.robots[robot_key]
            joint_pos = robot.data.default_joint_pos[env_ids].clone()
            joint_vel = robot.data.default_joint_vel[env_ids].clone()
            robot.set_joint_position_target(joint_pos, env_ids=env_ids)
            robot.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)

        if self.default_states is None:
            for key in self.robots.keys():
                self.robot_entity_cfg[key].resolve(self.scene)
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

        # Create joints
        if not self.joints_created:
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

        # ✅ Refresh intermediate values so that `_get_observations()` has updated values
        # self._compute_intermediate_values(env_ids)

    def _get_observations(self) -> torch.Tensor:
        """Compute observations for all robots and return a single tensor."""

        observations = []  # List to store tensors for all robots

        for robot_key in self.robots.keys(): # Loop through all robots dynamically
            robot = self.robots[robot_key]

            # ✅ Normalize DOF positions separately for each robot
            dof_pos_scaled = (
                2.0
                * (robot.data.joint_pos - self.robot_dof_lower_limits[robot_key])
                / (self.robot_dof_upper_limits[robot_key] - self.robot_dof_lower_limits[robot_key])
                - 1.0
            )

            # ✅ Concatenate observations for the current robot
            obs = torch.cat(
                (
                    dof_pos_scaled,
                    robot.data.joint_vel * self.cfg.dof_velocity_scale,
                ),
                dim=-1,
            )

            # ✅ Append the observation tensor to the list
            observations.append(obs)

        observations = torch.cat(observations, axis=-1)

        corners = self._get_corners() - self.scene.env_origins.unsqueeze(1)
        corners = corners.reshape(self.num_envs, -1)

        # observations = torch.cat((observations, corners), axis=-1)
        observations = corners

        # Append absolute pose
        abs_pose = self._get_absolute_pose()[:,0:3]
        abs_pose[:,0:3] -= self.scene.env_origins
        observations = torch.cat((observations, abs_pose), axis=-1)

        return {"policy": observations}
