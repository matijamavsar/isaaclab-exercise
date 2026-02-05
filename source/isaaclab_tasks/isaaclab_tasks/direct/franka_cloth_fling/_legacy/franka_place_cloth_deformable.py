# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch

from isaacsim.core.utils.stage import get_current_stage
from isaacsim.core.utils.torch.transformations import tf_combine, tf_inverse, tf_vector
from pxr import UsdGeom, UsdShade, Sdf, Gf, PhysxSchema

import isaaclab.sim as sim_utils
from isaaclab.actuators.actuator_cfg import ImplicitActuatorCfg
from isaaclab.assets import Articulation, ArticulationCfg, AssetBaseCfg
from isaaclab.assets import ParticleClothObject, ParticleClothObjectCfg
from isaaclab.assets import DeformableObject, DeformableObjectCfg, RigidObjectCfg
from isaaclab.envs import DirectRLEnv, DirectRLEnvCfg
from isaaclab.managers import SceneEntityCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg, PhysxCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab.utils.math import sample_uniform, subtract_frame_transforms, matrix_from_quat
from isaaclab.sensors import ContactSensorCfg
from pxr import UsdPhysics, UsdGeom, Gf, Sdf
import omni.usd
import omni.kit.commands
from omni.physx.scripts import physicsUtils, particleUtils
from isaaclab.controllers import DifferentialIKController, DifferentialIKControllerCfg
from isaacsim.core.utils.prims import get_prim_at_path

from isaaclab_assets import FRANKA_PANDA_HIGH_PD_CFG
import torch.nn.functional as F

@configclass
class FrankaClothEnvCfg(DirectRLEnvCfg):
    # env
    episode_length_s = 6
    max_episode_length = episode_length_s*120
    decimation = 4
    action_space = 16
    observation_space = 36
    state_space = 0
    use_dynamic_rewards = False

    # simulation
    sim: SimulationCfg = SimulationCfg(
        dt=1 / 120,
        render_interval=decimation,
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
        )
    )

    # scene
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=1, env_spacing=2.5, replicate_physics=False)

    cloth_plain = DeformableObjectCfg(
        prim_path="/World/envs/env_.*/cuboid",
        init_state=DeformableObjectCfg.InitialStateCfg(pos=(0.30993, -0.00316, 0.39438), rot=(0.71663, 0.0, 0.69745, 0.0)),
        # init_state=DeformableObjectCfg.InitialStateCfg(pos=(0.8, 0.0, 0.0), rot=(1, 0, 0, 0)),
        spawn=sim_utils.UsdFileCfg(
            usd_path="/workspace/isaaclab/source/isaaclab_tasks/isaaclab_tasks/sim/deformable_for_joint_lowRes.usd",
            scale=(1, 1, 0.5),
        ),
            )

    cloth = DeformableObjectCfg(
        prim_path="/World/envs/env_.*/cuboid/cuboid",
        init_state=DeformableObjectCfg.InitialStateCfg(pos=(0, 0, 0), rot=(0, 0, 0, 1)),
        spawn=None,
    )

    ### Robots
    robot_1 = FRANKA_PANDA_HIGH_PD_CFG.replace(
        prim_path="/World/envs/env_.*/Robot1",
        init_state=ArticulationCfg.InitialStateCfg(
            joint_pos={
                "panda_joint1": 0.0385,
                "panda_joint2": -0.4821,
                "panda_joint3": 0.1572, 
                "panda_joint4": -1.5468,
                "panda_joint5": 0.0822,
                "panda_joint6": 1.0760,
                "panda_joint7": 1.0027,
                "panda_finger_joint.*": 0.005,
            },
            pos=(0.0, -0.43, 0.0),
            rot=(1.0, 0.0, 0.0, 0.0),
        ),)
    
    robot_2 = FRANKA_PANDA_HIGH_PD_CFG.replace(
        prim_path="/World/envs/env_.*/Robot2",
        init_state=ArticulationCfg.InitialStateCfg(
            joint_pos={  
                "panda_joint1": -0.0741,
                "panda_joint2": -0.4798,
                "panda_joint3": -0.1377,
                "panda_joint4": -1.5472,
                "panda_joint5": -0.0717,
                "panda_joint6": 1.0772,
                "panda_joint7": 0.5548,
                "panda_finger_joint.*": 0.005,
            },
            pos=(0.0, 0.43, 0.0),
            rot=(1.0, 0.0, 0.0, 0.0),
        ),)

    # # ➕ **Add Contact Sensor to Track Forces on the Hand**
    # contact_sensor1 = ContactSensorCfg(
    #     prim_path="/World/envs/env_.*/Robot1/panda_leftfinger",  # ✅ End-effector contact
    #     update_period=0.0,  # Update every step
    #     history_length=6,
    #     debug_vis=True,  # Visualize contacts (can be disabled)
    # )

    # contact_sensor2 = ContactSensorCfg(
    #     prim_path="/World/envs/env_.*/Robot2/panda_leftfinger",  # ✅ End-effector contact
    #     update_period=0.0,  # Update every step
    #     history_length=6,
    #     debug_vis=True,  # Visualize contacts (can be disabled)
    # )

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

    action_scale = 2.5 # default is 7.5
    dof_velocity_scale = 0.1

    # reward scales
    spread_reward_scale = 20.0
    height_reward_scale = 2.0
    gripper_distance_reward_scale = 5.0
    corner_x_reward_scale = 15.0
    action_penalty_scale = 1e-4
    downward_reward_scale = 1e-3
    robot_centering_reward_scale = 1.0

    JOINTS_PLEASE = True # You want cartesian of joint control? this is where it happens

    # Reward selection flags
    use_rewards = {
        "spread":                   True,
        "height":                   True,
        "corner_x":                 True,
        "downward":                 False,
        "action_penalty":           False,
        "gripper_distance":         True,
        "robot_centering":          True,
    }

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
        

class FrankaClothEnv(DirectRLEnv):
    # pre-physics step calls
    #   |-- _pre_physics_step(action)
    #   |-- _apply_action()
    # post-physics step calls
    #   |-- _get_dones()
    #   |-- _get_rewards()
    #   |-- _reset_idx(env_ids)
    #   |-- _get_observations()

    cfg: FrankaClothEnvCfg

    def __init__(self, cfg: FrankaClothEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        def get_env_local_pose(env_pos: torch.Tensor, xformable: UsdGeom.Xformable, device: torch.device):
            """Compute pose in env-local coordinates"""
            world_transform = xformable.ComputeLocalToWorldTransform(0)
            world_pos = world_transform.ExtractTranslation()
            world_quat = world_transform.ExtractRotationQuat()

            px = world_pos[0] - env_pos[0]
            py = world_pos[1] - env_pos[1]
            pz = world_pos[2] - env_pos[2]
            qx = world_quat.imaginary[0]
            qy = world_quat.imaginary[1]
            qz = world_quat.imaginary[2]
            qw = world_quat.real

            return torch.tensor([px, py, pz, qw, qx, qy, qz], device=device)

        self.dt = self.cfg.sim.dt * self.cfg.decimation
        self.iteration_step = torch.tensor(0.0, device=self.device)
        self.default_states = None
        self._prev_actions = torch.zeros((self.num_envs, 18), device=self.device)
        self.joints_created = False

        self.robot_entity_cfg = {}
        self.robot_entity_cfg['robot_1'] = SceneEntityCfg("robot_1", joint_names=["panda_joint.*"], body_names=["panda_hand"])
        self.robot_entity_cfg['robot_2'] = SceneEntityCfg("robot_2", joint_names=["panda_joint.*"], body_names=["panda_hand"])
        
        diff_ik_cfg = DifferentialIKControllerCfg(command_type="pose", use_relative_mode=False, ik_method="dls")
        self._ik_controller = {}
        self._ik_controller['robot_1'] = DifferentialIKController(diff_ik_cfg, num_envs=self.num_envs, device=self.device)
        self._ik_controller['robot_2'] = DifferentialIKController(diff_ik_cfg, num_envs=self.num_envs, device=self.device)

        # Initialize robot dictionaries
        self.robots = {}  # Store robot objects
        self.prev_joint_vel = {}  # Store previous joint velocities
        self.filtered_dof_targets = {}  # Store smoothed action buffers
        self.robot_dof_targets = {}  # Store DOF targets
        self.rewards = {}

        self.robot_dof_lower_limits = {}  # Store lower joint limits
        self.robot_dof_upper_limits = {}  # Store upper joint limits
        self.robot_dof_speed_scales = {}  # Store joint speed scales

        stage = get_current_stage()

        # Define the number of robots dynamically
        self.num_robots = 2  # Change this number to create more robots dynamically

        for i in range(1, self.num_robots + 1):
            robot_key = f"robot_{i}"
            robot_prim_path = f"/World/envs/env_0/Robot{i}"  # Adjust if necessary

            # Instantiate robot and store in dictionary
            self.robots[robot_key] = getattr(self, "_" + robot_key)

            # Initialize buffers
            num_joints = self.robots[robot_key].num_joints

            self.prev_joint_vel[robot_key] = torch.zeros((self.num_envs, num_joints), device=self.device)
            self.filtered_dof_targets[robot_key] = torch.zeros((self.num_envs, num_joints), device=self.device)
            self.robot_dof_targets[robot_key] = torch.zeros((self.num_envs, num_joints), device=self.device)

            # Joint limits and speed scales
            self.robot_dof_lower_limits[robot_key] = self.robots[robot_key].data.soft_joint_pos_limits[0, :, 0].to(device=self.device)
            self.robot_dof_upper_limits[robot_key] = self.robots[robot_key].data.soft_joint_pos_limits[0, :, 1].to(device=self.device)
            self.robot_dof_speed_scales[robot_key] = torch.ones_like(self.robot_dof_lower_limits[robot_key])

            # Adjust finger joint speed scale
            finger_joints = ["panda_finger_joint1", "panda_finger_joint2"]
            for finger_joint in finger_joints:
                joint_idx = self.robots[robot_key].find_joints(finger_joint)[0]
                self.robot_dof_speed_scales[robot_key][joint_idx] = 0.1

        # One-Euro filter for smoothing actions
        self.one_euro_filter = OneEuroFilter(beta=0.002, min_cutoff=1.5, d_cutoff=1.0, dt=self.dt)

        # Initialize dictionaries for robot-specific data
        self.robot_local_grasp_pos = {}  # Grasp position for each robot
        self.robot_local_grasp_rot = {}  # Grasp rotation for each robot
        self.hand_link_idx = {}  # Hand link indices
        self.left_finger_link_idx = {}  # Left finger indices
        self.right_finger_link_idx = {}  # Right finger indices
        self.gripper_forward_axis = {}  # Forward axis for each robot
        self.gripper_up_axis = {}  # Up axis for each robot

        stage = get_current_stage()

        for i in range(1, self.num_robots + 1):
            robot_key = f"robot_{i}"
            robot_prim_path = f"/World/envs/env_0/Robot{i}"  # Adjust if necessary

            # ✅ Read hand and finger poses separately for each robot
            hand_pose = get_env_local_pose(
                self.scene.env_origins[0],
                UsdGeom.Xformable(stage.GetPrimAtPath(f"{robot_prim_path}/panda_link7")),
                self.device,
            )

            lfinger_pose = get_env_local_pose(
                self.scene.env_origins[0],
                UsdGeom.Xformable(stage.GetPrimAtPath(f"{robot_prim_path}/panda_leftfinger")),
                self.device,
            )

            rfinger_pose = get_env_local_pose(
                self.scene.env_origins[0],
                UsdGeom.Xformable(stage.GetPrimAtPath(f"{robot_prim_path}/panda_rightfinger")),
                self.device,
            )

            # ✅ Compute grasp pose separately for each robot
            finger_pose = torch.zeros(7, device=self.device)
            finger_pose[0:3] = (lfinger_pose[0:3] + rfinger_pose[0:3]) / 2.0
            finger_pose[3:7] = lfinger_pose[3:7]

            hand_pose_inv_rot, hand_pose_inv_pos = tf_inverse(hand_pose[3:7], hand_pose[0:3])
            robot_local_grasp_pose_rot, robot_local_pose_pos = tf_combine(
                hand_pose_inv_rot, hand_pose_inv_pos, finger_pose[3:7], finger_pose[0:3]
            )
            robot_local_pose_pos += torch.tensor([0, 0.04, 0], device=self.device)

            # ✅ Store robot-specific grasp positions and rotations
            self.robot_local_grasp_pos[robot_key] = robot_local_pose_pos.repeat((self.num_envs, 1))
            self.robot_local_grasp_rot[robot_key] = robot_local_grasp_pose_rot.repeat((self.num_envs, 1))

            # ✅ Store robot-specific transformation axes
            self.gripper_forward_axis[robot_key] = torch.tensor([0, 0, 1], device=self.device, dtype=torch.float32).repeat((self.num_envs, 1))
            self.gripper_up_axis[robot_key] = torch.tensor([0, 1, 0], device=self.device, dtype=torch.float32).repeat((self.num_envs, 1))

            # ✅ Store hand and finger link indices for each robot
            self.hand_link_idx[robot_key] = self.robots[robot_key].find_bodies("panda_link7")[0][0]
            self.left_finger_link_idx[robot_key] = self.robots[robot_key].find_bodies("panda_leftfinger")[0][0]
            self.right_finger_link_idx[robot_key] = self.robots[robot_key].find_bodies("panda_rightfinger")[0][0]

        # ✅ Store grasp pose separately for each robot
        self.robot_grasp_rot = {robot_key: torch.zeros((self.num_envs, 4), device=self.device) for robot_key in self.robots}
        self.robot_grasp_pos = {robot_key: torch.zeros((self.num_envs, 3), device=self.device) for robot_key in self.robots}

    def _create_cloth(self, stage, env_idx):
        cloth_mesh_path = Sdf.Path(f"/World/envs/env_{env_idx}/Cloth")
        particle_material_path = Sdf.Path("/World/particleMaterial")

        # ✅ Create a mesh that will be turned into cloth
        plane_resolution = 100
        plane_width = 100

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

    def _process_multiple_arrays(self, points_list):
        """
        Processes multiple 3D arrays of points and finds the corners for each one.
        
        :param points_list: A list of tensors, where each tensor has shape (N, 3).
        :return: A list of dictionaries with the corners of each array.
        """
        corners_list = []
        
        for points in points_list:
            corners = self._find_corners(points)
            corners_list.append(corners)
        
        return corners_list

    def _find_corners(self, points):
        """
        Finds the corners of a rectangular mesh in 3D space from a set of points.
        
        :param points: A tensor of shape (N, 3) where N is the number of points, and each point has (x, y, z) coordinates.
        :return: A dictionary with keys 'bottom_left', 'top_left', 'top_right', 'bottom_right' corresponding to the 4 corners.
        """
        # Extract the minimum and maximum X, Y, and Z values
        min_x = torch.min(points[:, 0])
        max_x = torch.max(points[:, 0])
        min_y = torch.min(points[:, 1])
        max_y = torch.max(points[:, 1])
        max_z = torch.max(points[:, 2])
        
        # Find the corner points by matching these min/max values with the points in the tensor
        bottom_left_idx = torch.argmin(torch.abs(points[:, 0] - min_x) + torch.abs(points[:, 1] - min_y) + torch.abs(points[:, 2] - max_z))
        top_left_idx = torch.argmin(torch.abs(points[:, 0] - min_x) + torch.abs(points[:, 1] - max_y) + torch.abs(points[:, 2] - max_z))
        top_right_idx = torch.argmin(torch.abs(points[:, 0] - max_x) + torch.abs(points[:, 1] - max_y) + torch.abs(points[:, 2] - max_z))
        bottom_right_idx = torch.argmin(torch.abs(points[:, 0] - max_x) + torch.abs(points[:, 1] - min_y) + torch.abs(points[:, 2] - max_z))

        # Get the corresponding corner points from the indices
        bottom_left = points[bottom_left_idx]
        top_left = points[top_left_idx]
        top_right = points[top_right_idx]
        bottom_right = points[bottom_right_idx]

        print(bottom_left_idx, top_left_idx, top_right_idx, bottom_right_idx)
        # Return the corners as a dictionary
        corners = torch.stack((
            bottom_left,
            top_left,
            top_right,
            bottom_right
        ))
        
        return corners

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

        # ✅ Setup contact sensors
        # self._contact_sensors = {}
        # self._contact_sensors['robot_1'] = self.cfg.contact_sensor1.class_type(self.cfg.contact_sensor1)
        # self._contact_sensors['robot_2'] = self.cfg.contact_sensor2.class_type(self.cfg.contact_sensor2)
        self._cloth_plain = self.cfg.cloth_plain.class_type(self.cfg.cloth_plain)
        self._cloth = self.cfg.cloth.class_type(self.cfg.cloth)

        # ✅ Clone environments
        self.scene.clone_environments(copy_from_source=True)
        
        # # ----------------------------
        # # ✅ Create Particle System ONCE
        # # ----------------------------
        # stage = self.scene.stage
        # particle_system_path = Sdf.Path("/World/particleSystem")

        # restOffset = 0.01
        # contactOffset = 0.02

        # particleUtils.add_physx_particle_system(
        #     stage=stage,
        #     particle_system_path=particle_system_path,
        #     contact_offset=contactOffset,
        #     rest_offset=restOffset,
        #     particle_contact_offset=contactOffset,
        #     solid_rest_offset=restOffset,
        #     fluid_rest_offset=0.01,
        #     solver_position_iterations=16,
        #     simulation_owner="physicsScene",
        # )

        # # ✅ Create Particle Material ONCE
        # particle_material_path = Sdf.Path("/World/particleMaterial")
        # particleUtils.add_pbd_particle_material(stage, particle_material_path)
        # particleUtils.add_pbd_particle_material(stage, particle_material_path, drag=0.01, lift=0.01, friction=0.6)

        # physicsUtils.add_physics_material_to_prim(
        #     stage, stage.GetPrimAtPath(particle_system_path), particle_material_path
        # )

        # # ----------------------------
        # # ✅ Create a Single Material for Cloth
        # # ----------------------------
        # material_path = particle_material_path
        # # material_path = Sdf.Path("/World/Materials/ClothMaterial")

        # # ✅ Create a Shader for the material
        # shader_path = material_path.AppendPath("Shader")
        # shader = UsdShade.Shader.Define(stage, shader_path)
        # shader.CreateIdAttr("UsdPreviewSurface")  # Use USD's Preview Surface Shader

        # # ✅ Set shader properties (e.g., color, roughness, metallic)
        # shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(1.0, 0.2, 0.2))
        # shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.8)
        # shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(0.0)

        # # ✅ Create cloth for each environment
        # self._cloth_objects = []
        # for env_idx in range(self.scene.cfg.num_envs):
        #     cloth_prim_path = self._create_cloth(self.scene.stage, env_idx)  # Create cloth for each env
        #     cloth_cfg = ParticleClothObjectCfg(
        #         prim_path=cloth_prim_path.pathString, 
        #         spawn=None
        #         )
        #     self._cloth_objects.append(ParticleClothObject(cloth_cfg))

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


    # pre-physics step calls
    def _pre_physics_step(self, actions: torch.Tensor):
        """Apply the One-Euro filter for smoother, dynamic motion for multiple robots."""
        ee_jacobi_idx = 7

        num_robots = len(self.robots)
        # Ensure actions are clamped between -1 and 1
        self.actions = actions.clone().clamp(-1.5, 1.5)
        self.actions = self.actions.reshape(self.num_envs, num_robots, -1)

        # ✅ Get gripper values for fixing gripper actions
        gripper_actions = self.actions[:,:,-1].clone() * 0.0

        for i, robot_key in enumerate(self.robots.keys()):  # Loop through all robots dynamically
            if self.cfg.JOINTS_PLEASE is False:
                self.robot_dof_targets[robot_key] = torch.zeros((self.num_envs, 9), device=self.device)
                self.robot_dof_targets[robot_key][:,-2:] = gripper_actions[:,i].unsqueeze(1).repeat(1,2)
                # ✅ Compute new target positions for each robot separately
                robot = self.robots[robot_key]
                robot_entity_cfg = self.robot_entity_cfg[robot_key]

                # Extract desired end-effector position and orientation
                ee_pose_desired = self.actions[:,i,0:7]
                self._ik_controller[robot_key].set_command(ee_pose_desired)

                ee_pose_w = robot.data.body_state_w[
                    :, robot_entity_cfg.body_ids[0], 0:7
                ]

                # Obtain robot's Jacobian matrix
                jacobian = robot.root_physx_view.get_jacobians()[
                    :, ee_jacobi_idx, :, 
                    robot_entity_cfg.joint_ids]

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
                new_targets = (
                    padded_joint_pos
                    + self.robot_dof_speed_scales[robot_key] * self.dt
                    * self.robot_dof_targets[robot_key] * self.cfg.action_scale
                )
            else:
                padded_actions = F.pad(input=self.actions[:,i,0:7], pad=(0,2), mode='constant', value=0)
                new_targets = (
                    self.robot_dof_targets[robot_key]
                    + self.robot_dof_speed_scales[robot_key] * self.dt
                    * padded_actions * self.cfg.action_scale
                )
            # ✅ Clamp values within each robot's DOF limits
            new_targets = torch.clamp(new_targets, self.robot_dof_lower_limits[robot_key], self.robot_dof_upper_limits[robot_key])

            # ✅ Apply One-Euro filter for smoother motion
            # self.filtered_dof_targets[robot_key] = self.one_euro_filter.filter(new_targets)
            self.filtered_dof_targets[robot_key] = new_targets

            # ✅ Use the smoothed target instead of the raw target
            self.robot_dof_targets[robot_key] = self.filtered_dof_targets[robot_key]

    def _apply_action(self):
        """Apply joint position targets for multiple robots dynamically."""
        
        for robot_key in self.robots.keys():  # Loop through all robots dynamically
            # ✅ Set joint position target separately for each robot
            self.robots[robot_key].set_joint_position_target(self.robot_dof_targets[robot_key])

    # post-physics step calls
    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
         # Initialize a boolean tensor for termination flags
        terminated = 0
        truncated = self.episode_length_buf >= self.max_episode_length - 1
        return terminated, truncated

    def _get_rewards(self) -> torch.Tensor:
        """Compute rewards for lifting the cloth with stable grasping."""
        
        # Refresh intermediate values
        self._compute_intermediate_values()

        # Prepare arrays for rewards, corresponding to all environments
        cloth_positions = self._cloth_plain.root_physx_view.get_nodal_positions().reshape(self.num_envs, -1, 3)

        # corners = torch.stack(self._process_multiple_arrays(cloth_positions))
        corners = cloth_positions[:, [2016, 2066, 4566, 4616]].clone()

        # Try to keep height low
        mean_cloth_height = cloth_positions[:, :, 2].mean(dim=1)
        height_reward = 1/(0.1 + mean_cloth_height) * self.cfg.height_reward_scale

        pairwise_sum = torch.zeros(self.num_envs, device=self.device)
        count = 0.0
        # Loop over each pair (there are 6 pairs for 4 corners)
        for i in range(4):
            for j in range(i + 1, 4):
                pairwise_sum += torch.norm(corners[:, i] - corners[:, j], dim=-1)
                count += 1.0
        r_spread = pairwise_sum / count  # larger is better

        spread_reward = torch.zeros(self.num_envs, device=self.device)
        max_allowed_distance = 0.75
        close_enough = torch.norm(corners[:, 2] - corners[:, 3], dim=-1) <= max_allowed_distance
        spread_reward[close_enough] = (self.cfg.spread_reward_scale * r_spread[close_enough])

        # ✅ New reward: Encourage free corners to move outward along X axis
        corner_x_vals = corners[:, 0, 0] + corners[:, 1, 0]
        corner_x_reward = corner_x_vals * self.cfg.corner_x_reward_scale

        # Gripper distance reward and center grippers
        robot_centering_reward = torch.zeros(self.num_envs, device=self.device)
        grippers = {}
        for robot_key in self.robots.keys():
            # Get root pose (position + rotation quaternion)
            root_pos = self.robots[robot_key].data.root_state_w[:, 0:3]
            root_quat = self.robots[robot_key].data.root_state_w[:, 3:7]
            lfinger_pos = self.robots[robot_key].data.body_pos_w[:, self.left_finger_link_idx[robot_key]]
            rfinger_pos = self.robots[robot_key].data.body_pos_w[:, self.right_finger_link_idx[robot_key]]
            grippers[robot_key] = 0.5 * (lfinger_pos + rfinger_pos)

            # Use dummy quaternion for gripper (only care about position)
            dummy_quat = torch.tensor([1.0, 0.0, 0.0, 0.0], device=self.device).repeat(self.num_envs, 1)

            # Compute gripper position in base frame
            local_pos, _ = subtract_frame_transforms(
                root_pos, root_quat,
                grippers[robot_key], dummy_quat
            )

            # Reward closeness to Y=0 using exp(-|y|), always positive and highest at 0
            y_dev = torch.abs(local_pos[:, 1])  # Y component
            robot_centering_reward += torch.exp(-y_dev) * self.cfg.robot_centering_reward_scale

        # Reward so that distance betweens grippers is in some range
        min_dist = 0.5
        max_dist = 0.7
        gripper_distance = torch.norm(grippers['robot_1'] - grippers['robot_2'], dim=-1)

        # Reward peaks in middle of range (like a Gaussian)
        center = (min_dist + max_dist) / 2
        width = (max_dist - min_dist) / 2

        gripper_distance_reward = torch.exp(-((gripper_distance - center) ** 2) / (2 * width ** 2))
        gripper_distance_reward *= self.cfg.gripper_distance_reward_scale

        # Alignment reward (encourage the gripper to point down)
        gripper_downward_reward = torch.zeros(self.num_envs, device=self.device)
        for i, robot_key in enumerate(self.robots.keys()):
            gripper_quat = self.robots[robot_key].data.body_quat_w[:, self.hand_link_idx[robot_key]]
            gripper_rot = matrix_from_quat(gripper_quat)  # Assuming matrix_from_quat is correctly implemented

            gripper_z_axis = gripper_rot[:, :, 2]  # Z axis direction of the gripper (num_envs, 3)
            world_down = torch.tensor([0, 0, -1], device=self.device, dtype=torch.float32).repeat(self.num_envs, 1)
            alignment_reward = torch.sum(gripper_z_axis * world_down, dim=-1)  # Dot product to align downward

            gripper_downward_reward += alignment_reward * self.cfg.downward_reward_scale  # Add to final reward

        # Penalize actions
        # action_penalty = torch.sum(self.actions**2, dim=(-1,-2)) * self.cfg.action_penalty_scale

        # Penalize joint velocities
        action_penalty = torch.zeros(self.num_envs, device=self.device)
        for robot_key in self.robots.keys():
            action_penalty += torch.sum(self.robots[robot_key].data.joint_vel**2, dim=(-1,-2)) * self.cfg.action_penalty_scale

        self.rewards["spread"] = spread_reward
        self.rewards["height"] = height_reward
        self.rewards["corner_x"] = corner_x_reward
        self.rewards["downward"] = gripper_downward_reward
        self.rewards["robot_centering"] = robot_centering_reward
        self.rewards["action_penalty"] = action_penalty
        self.rewards["gripper_distance"] = gripper_distance_reward

        # ✅ Compute Total Reward
        total_reward = torch.zeros(self.num_envs, device=self.device)

        for reward_name, is_enabled in self.cfg.use_rewards.items():
            if is_enabled:  # If the reward is enabled
                if 'penalty' in reward_name:
                    total_reward -= self.rewards[reward_name]
                else:
                    total_reward += self.rewards[reward_name]

        # ✅ Log for Debugging
        self.extras["log"] = {
            "spread_reward": spread_reward.mean(),
            "height_reward": height_reward.mean(),
            "corner_x_reward": corner_x_reward.mean(),
            "gripper_downward_reward": gripper_downward_reward.mean(),
            "robot_centering_reward": robot_centering_reward.mean(),
            "action_penalty": action_penalty.mean(),
            "gripper_distance_reward": gripper_distance_reward.mean(),
        }

        return total_reward

    def _reset_idx(self, env_ids: torch.Tensor | None):
        """Reset the state of multiple robots and environment objects properly."""
        super()._reset_idx(env_ids)
        for robot_key in self.robots.keys():  # Loop through all robots dynamically
            robot = self.robots[robot_key]
            joint_pos = robot.data.default_joint_pos[env_ids].clone()
            joint_vel = robot.data.default_joint_vel[env_ids].clone()
            robot.set_joint_position_target(joint_pos, env_ids=env_ids)
            robot.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)

        ### Use Cartesian controller to pick up the handles of the cloth so you can then do RL
        if self.default_states is None:
            for key in self.robots.keys():
                self.robot_entity_cfg[key].resolve(self.scene)
            self.default_states = {}
            self.default_states['cloth'] = self._cloth.data.nodal_state_w.clone()

        self._cloth.write_nodal_state_to_sim(
            self.default_states['cloth'][env_ids], env_ids=env_ids)

        # Create joints
        if not self.joints_created:
            for idx in env_ids:
                cube_1_path = Sdf.Path("/World/envs/env_" + str(int(idx)) + "/cuboid/Cube_01")
                cube_2_path = Sdf.Path("/World/envs/env_" + str(int(idx)) + "/cuboid/Cube")
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

        # # Create attachments
        # for idx in env_ids:
        #     cloth_path = Sdf.Path("/World/envs/env_" + str(int(idx)) + "/cuboid/cuboid")
        #     finger_paths = []
        #     finger_paths.append(Sdf.Path("/World/envs/env_" + str(int(idx)) + "/Robot1/panda_leftfinger"))
        #     finger_paths.append(Sdf.Path("/World/envs/env_" + str(int(idx)) + "/Robot1/panda_rightfinger"))
        #     finger_paths.append(Sdf.Path("/World/envs/env_" + str(int(idx)) + "/Robot2/panda_leftfinger"))
        #     finger_paths.append(Sdf.Path("/World/envs/env_" + str(int(idx)) + "/Robot2/panda_rightfinger"))
        #     for k, finger_path in enumerate(finger_paths):
        #         attachment_path = cloth_path.AppendElementString("attachment" + str(k))
        #         attachment = PhysxSchema.PhysxPhysicsAttachment.Define(self.scene.stage, attachment_path)
        #         attachment.GetActor0Rel().SetTargets([cloth_path])
        #         attachment.GetActor1Rel().SetTargets([finger_path])
        #         auto_attach = PhysxSchema.PhysxAutoAttachmentAPI
        #         auto_attach.Apply(attachment.GetPrim())
        
        # ✅ Refresh intermediate values so that `_get_observations()` has updated values
        self._compute_intermediate_values(env_ids)

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

        # ✅ Add cloth corners to the observation space
        cloth_positions = self._cloth_plain.root_physx_view.get_nodal_positions().reshape(self.num_envs, -1, 3)

        # corners = torch.stack(self._process_multiple_arrays(cloth_positions))
        corners = cloth_positions[:, [2016, 4566, 4616, 2066]]
        corners = corners.view(corners.shape[0], -1)

        observations = torch.cat((observations, corners), axis=-1)

        # Append previous actions to the observations
        # observations = torch.cat((observations, self._prev_actions), axis=-1)

        return {"policy": observations}

    # auxiliary methods

    def _compute_intermediate_values(self, env_ids: torch.Tensor | None = None):
        """Compute intermediate values separately for each robot."""
        
        for robot_key in self.robots.keys():  # Loop through all robots dynamically
            robot = self.robots[robot_key]

            if env_ids is None:
                env_ids = robot._ALL_INDICES  # Ensure correct environment indices per robot

            # ✅ Extract hand position and rotation for the current robot
            hand_pos = robot.data.body_pos_w[env_ids, self.hand_link_idx[robot_key]]
            hand_rot = robot.data.body_quat_w[env_ids, self.hand_link_idx[robot_key]]

            # ✅ Compute grasp transform separately for each robot
            (
                self.robot_grasp_rot[robot_key][env_ids],
                self.robot_grasp_pos[robot_key][env_ids],
            ) = self._compute_grasp_transforms(
                hand_rot,
                hand_pos,
                self.robot_local_grasp_rot[robot_key][env_ids],
                self.robot_local_grasp_pos[robot_key][env_ids],
            )

    def _compute_grasp_transforms(
        self,
        hand_rot,
        hand_pos,
        franka_local_grasp_rot,
        franka_local_grasp_pos,
    ):
        global_franka_rot, global_franka_pos = tf_combine(
            hand_rot, hand_pos, franka_local_grasp_rot, franka_local_grasp_pos
        )

        return global_franka_rot, global_franka_pos
