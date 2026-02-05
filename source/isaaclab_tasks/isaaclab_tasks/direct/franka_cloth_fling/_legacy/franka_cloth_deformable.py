# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch

from isaacsim.core.utils.stage import get_current_stage
from isaacsim.core.utils.torch.transformations import tf_combine, tf_inverse, tf_vector
from pxr import UsdGeom, UsdShade

import isaaclab.sim as sim_utils
from isaaclab.actuators.actuator_cfg import ImplicitActuatorCfg
from isaaclab.assets import Articulation, ArticulationCfg, AssetBase, AssetBaseCfg, DeformableObject, DeformableObjectCfg
from isaaclab.envs import DirectRLEnv, DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg, PhysxCfg
from isaaclab.terrains import TerrainImporterCfg, TerrainGeneratorCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab.utils.math import sample_uniform
from isaaclab.sensors import ContactSensorCfg
from pxr import UsdPhysics, UsdGeom, PhysxSchema, Gf, Sdf
import omni.usd
import omni.kit.commands
from omni.physx.scripts import physicsUtils, deformableUtils
from isaacsim.core.utils.prims import get_prim_at_path


@configclass
class FrankaClothEnvCfg(DirectRLEnvCfg):
    # env
    episode_length_s = 8.3333
    max_episode_length = 800
    decimation = 2
    action_space = 18
    observation_space = 336
    state_space = 0

    # simulation
    sim: SimulationCfg = SimulationCfg(
        dt=1 / 120,
        render_interval=decimation,
        disable_contact_processing=True,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
        physx=PhysxCfg(
            gpu_max_particle_contacts = 2**20 # Default is 2**20
        )
    )

    # scene
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=1, env_spacing=5.0, replicate_physics=False)

    # robots

    robot_1 = ArticulationCfg(
        prim_path="/World/envs/env_.*/Robot1",
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"{ISAAC_NUCLEUS_DIR}/Robots/Franka/franka_instanceable.usd",
            activate_contact_sensors=True,
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
                "panda_joint1": 1.157,
                "panda_joint2": -1.066,
                "panda_joint3": -0.155,
                "panda_joint4": -2.239,
                "panda_joint5": -1.841,
                "panda_joint6": 1.003,
                "panda_joint7": 0.469,
                "panda_finger_joint.*": 0.035,
            },
            pos=(0.0, -0.750, 0.0),
            rot=(0.0, 0.0, 0.0, 1.0),
        ),
        actuators={
            "panda_shoulder": ImplicitActuatorCfg(
                joint_names_expr=["panda_joint[1-4]"],
                effort_limit=87.0,
                velocity_limit=2.175,
                stiffness=80.0,
                damping=4.0,
            ),
            "panda_forearm": ImplicitActuatorCfg(
                joint_names_expr=["panda_joint[5-7]"],
                effort_limit=12.0,
                velocity_limit=2.61,
                stiffness=80.0,
                damping=4.0,
            ),
            "panda_hand": ImplicitActuatorCfg(
                joint_names_expr=["panda_finger_joint.*"],
                effort_limit=200.0,
                velocity_limit=0.2,
                stiffness=2e3,
                damping=1e2,
            ),
        },
    )

    robot_2 = ArticulationCfg(
        prim_path="/World/envs/env_.*/Robot2",
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"{ISAAC_NUCLEUS_DIR}/Robots/Franka/franka_instanceable.usd",
            activate_contact_sensors=True,
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
                "panda_joint1": 1.157,
                "panda_joint2": -1.066,
                "panda_joint3": -0.155,
                "panda_joint4": -2.239,
                "panda_joint5": -1.841,
                "panda_joint6": 1.003,
                "panda_joint7": 0.469,
                "panda_finger_joint.*": 0.035,
            },
            pos=(0.0, 0.750, 0.0),
            rot=(0.0, 0.0, 0.0, 1.0),
        ),
        actuators={
            "panda_shoulder": ImplicitActuatorCfg(
                joint_names_expr=["panda_joint[1-4]"],
                effort_limit=87.0,
                velocity_limit=2.175,
                stiffness=80.0,
                damping=4.0,
            ),
            "panda_forearm": ImplicitActuatorCfg(
                joint_names_expr=["panda_joint[5-7]"],
                effort_limit=12.0,
                velocity_limit=2.61,
                stiffness=80.0,
                damping=4.0,
            ),
            "panda_hand": ImplicitActuatorCfg(
                joint_names_expr=["panda_finger_joint.*"],
                effort_limit=200.0,
                velocity_limit=0.2,
                stiffness=2e3,
                damping=1e2,
            ),
        },
    )

    # ➕ **Add Contact Sensor to Track Forces on the Hand**
    contact_sensor1 = ContactSensorCfg(
        prim_path="/World/envs/env_.*/Robot1/panda_link7",  # ✅ End-effector contact
        update_period=0.0,  # Update every step
        history_length=6,
        debug_vis=True,  # Visualize contacts (can be disabled)
    )

    contact_sensor2 = ContactSensorCfg(
        prim_path="/World/envs/env_.*/Robot2/panda_link7",  # ✅ End-effector contact
        update_period=0.0,  # Update every step
        history_length=6,
        debug_vis=True,  # Visualize contacts (can be disabled)
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

    action_scale = 7.5
    dof_velocity_scale = 0.1

    # reward scales
    height_reward_scale = 2.0
    separation_reward_scale = 0.2
    facing_reward_scale = 0.2


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
        physics_context = self.sim.get_physics_context()
        physics_context.set_physx_update_transformations_settings(update_to_usd=True)
        physics_context.enable_fabric = False

        # Initialize robot dictionaries
        self.robots = {}  # Store robot objects
        self.prev_joint_vel = {}  # Store previous joint velocities
        self.filtered_dof_targets = {}  # Store smoothed action buffers
        self.robot_dof_targets = {}  # Store DOF targets

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

    def create_cloth_mesh(self, stage, target_path, mesh_size, mesh_resolution):
        _, tmp_path = omni.kit.commands.execute("CreateMeshPrim", 
                                                prim_type="Cube", 
                                                select_new_prim=False,
                                                u_patches=mesh_resolution,
                                                v_patches=mesh_resolution,
                                                w_patches=2,
                                                half_scale=mesh_size / 2)
        omni.kit.commands.execute("MovePrim", path_from=tmp_path, path_to=target_path)
        omni.usd.get_context().get_selection().set_selected_prim_paths([], False)
        return UsdGeom.Mesh.Get(stage, target_path)

    def create_cloth(self, stage, env_idx):
        """Creates a high-resolution thin cube to act as a deformable cloth."""
        
        # ✅ Define the deformable object path
        cloth_prim_path = f"/World/envs/env_{env_idx}/Cloth"
        print("***************************************")
        print("Creating cloth at path", cloth_prim_path)
        print("***************************************")

        # Create sphere mesh used as the 'skin mesh' for the deformable body
        mesh_size = 100 # in cm
        mesh_resolution = 50
        skin_mesh = self.create_cloth_mesh(stage, cloth_prim_path, mesh_size, mesh_resolution)
        skin_mesh.GetPrim().GetAttribute("xformOp:translate").Set(Gf.Vec3f(0.4, 0.0, 0.01))
        skin_mesh.GetPrim().GetAttribute("xformOp:scale").Set(Gf.Vec3f(1.0, 1.0, 0.005))

        skin_mesh.CreateDisplayColorAttr().Set([Gf.Vec3f(0.0, 0.0, 0.5)])

        # Create tet meshes for simulation and collision based on the skin mesh
        simulation_resolution = 50

        # Apply PhysxDeformableBodyAPI and PhysxCollisionAPI to skin mesh and set parameter to default values
        _ = deformableUtils.add_physx_deformable_body(
            stage,
            skin_mesh.GetPath(),
            collision_simplification=True,
            simulation_hexahedral_resolution=simulation_resolution,
            self_collision=True,
        )

        # ✅ Set Rest Offset and Contact Offset using PhysxCollisionAPI
        deformable_body_prim = stage.GetPrimAtPath(skin_mesh.GetPath())

        if deformable_body_prim:
            collision_api = PhysxSchema.PhysxCollisionAPI.Apply(deformable_body_prim)

            # ✅ Set the Rest Offset and Contact Offset (THIS WORKS!)
            collision_api.GetRestOffsetAttr().Set(0.0)  # Controls surface penetration before contact
            collision_api.GetContactOffsetAttr().Set(0.0001)  # Determines early collision detection

        # Create a deformable body material and set it on the deformable body
        deformable_material_path = omni.usd.get_stage_next_free_path(stage, cloth_prim_path + "/deformableBodyMaterial", True)
        deformableUtils.add_deformable_body_material(
            stage,
            deformable_material_path,
            youngs_modulus=1000000.0,
            poissons_ratio=0.49,
            damping_scale=0.5,
            dynamic_friction=1.5,
            density=10,
        )
        physicsUtils.add_physics_material_to_prim(stage, skin_mesh.GetPrim(), deformable_material_path)

        return cloth_prim_path

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
        self._contact_sensor1 = self.cfg.contact_sensor1.class_type(self.cfg.contact_sensor1)
        self._contact_sensor2 = self.cfg.contact_sensor2.class_type(self.cfg.contact_sensor2)

        # ✅ Clone environments
        self.scene.clone_environments(copy_from_source=True)

        # ✅ Create deformable cloth dynamically
        self._cloth_objects = []
        for env_idx in range(self.scene.cfg.num_envs):
            cloth_prim_path = self.create_cloth(self.scene.stage, env_idx)
            cloth_cfg = DeformableObjectCfg(
                prim_path=cloth_prim_path, 
                spawn=None, 
                init_state=DeformableObjectCfg.InitialStateCfg(
                    pos=(0, 0, 0), rot=(0, 0, 0, 1))
                )
            self._cloth_objects.append(DeformableObject(cloth_cfg))

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
            visual_plane.GetDisplayColorAttr().Set([Gf.Vec3f(0.0, 0.05, 0.0)])  # Green

        # ✅ Create a new material for the ground
        material_path = "/World/Materials/GroundMaterial"
        material_prim = stage.DefinePrim(material_path, "Material")
        material = UsdShade.Material(material_prim)

        # ✅ Create a shader
        shader_path = material_path + "/Shader"
        shader = UsdShade.Shader.Define(stage, shader_path)
        shader.CreateIdAttr("UsdPreviewSurface")  # Use USD's default shader

        # ✅ Set ground color to dark gray
        shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(0.2, 0.2, 0.2))  # Dark gray
        shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.8)  # Rough texture
        shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(0.0)  # Non-metallic

        # ✅ Bind the material to the ground plane
        UsdShade.MaterialBindingAPI(visual_plane).Bind(material)

    # pre-physics step calls

    def _pre_physics_step(self, actions: torch.Tensor):
        """Apply the One-Euro filter for smoother, dynamic motion for multiple robots."""

        # Ensure actions are clamped between -1 and 1
        self.actions = actions.clone().clamp(-1.0, 1.0)

        # Split the action tensor for each robot
        num_robots = len(self.robots)
        split_actions = torch.chunk(self.actions, num_robots, dim=-1)  # Split actions across robots

        for i, robot_key in enumerate(self.robots.keys()):  # Loop through all robots dynamically
            # ✅ Compute new target positions for each robot separately
            new_targets = (
                self.robot_dof_targets[robot_key]
                + self.robot_dof_speed_scales[robot_key] * self.dt * split_actions[i] * self.cfg.action_scale
            )
            
            # ✅ Clamp values within each robot's DOF limits
            new_targets = torch.clamp(new_targets, self.robot_dof_lower_limits[robot_key], self.robot_dof_upper_limits[robot_key])

            # ✅ Apply One-Euro filter for smoother motion
            self.filtered_dof_targets[robot_key] = self.one_euro_filter.filter(new_targets)

            # ✅ Use the smoothed target instead of the raw target
            self.robot_dof_targets[robot_key] = self.filtered_dof_targets[robot_key]

    def _apply_action(self):
        """Apply joint position targets for multiple robots dynamically."""
        
        for robot_key in self.robots.keys():  # Loop through all robots dynamically
            robot = self.robots[robot_key]

            # ✅ Set joint position target separately for each robot
            robot.set_joint_position_target(self.robot_dof_targets[robot_key])

    # post-physics step calls

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        terminated = 0 # TODO
        truncated = self.episode_length_buf >= self.max_episode_length - 1
        return terminated, truncated

    def _get_rewards(self) -> torch.Tensor:
        """Compute rewards based on lifting cloth, keeping end-effectors apart, and facing each other."""
        
        # Refresh the intermediate values for all robots
        self._compute_intermediate_values()

        # ✅ 1. Height Reward - Encourage robots to lift the cloth as high as possible
        height_reward = torch.zeros(self.num_envs, device=self.device)  # Initialize height reward per environment
        proximity_reward = torch.zeros(self.num_envs, device=self.device)  # Initialize proximity reward per environment

        for env_idx in range(self.num_envs):
            self._cloth_objects[env_idx].data.update(self.dt)
            cloth_positions = self._cloth_objects[env_idx].data.nodal_pos_w

            mean_cloth_height = cloth_positions[0][:,2].mean()
            height_reward[env_idx] = mean_cloth_height * self.cfg.height_reward_scale  # Scale reward
        
            # ✅ Compute proximity reward for grippers
            min_distance = float('inf')
            for robot_key in self.robots.keys():
                ee_pos = self.robots[robot_key].data.body_pos_w[:, self.hand_link_idx[robot_key]]  # (num_envs, 3)
                ee_cloth_distances = torch.norm(cloth_positions.unsqueeze(0) - ee_pos[env_idx].unsqueeze(0), dim=-1)  # (N,)
                min_distance = min(min_distance, ee_cloth_distances.min())

            proximity_reward[env_idx] = (1.0 / (1.0 + min_distance)) * 5.0  # Inverse distance scaling (closer is better)

        # ✅ 2. Smooth Motion Penalty - Penalize sudden velocity changes
        smooth_motion_penalty = torch.zeros(self.num_envs, device=self.device)
        for robot_key in self.robots.keys():
            joint_vel = self.robots[robot_key].data.joint_vel
            delta_vel = torch.abs(joint_vel - self.prev_joint_vel[robot_key])  # Compute velocity change
            smooth_motion_penalty += torch.where(delta_vel > 1.0, -0.1 * (delta_vel - 1.0), torch.tensor(0.0, device=self.device)).sum(dim=-1)
            self.prev_joint_vel[robot_key] = joint_vel.clone()  # Update previous velocity

        # ✅ 3. Separation Reward - Encourage end-effectors to stay apart
        robot_keys = list(self.robots.keys())  # Get robot keys dynamically
        ee_1_pos = self.robots[robot_keys[0]].data.body_pos_w[:, self.hand_link_idx[robot_keys[0]]]  # End-effector 1
        ee_2_pos = self.robots[robot_keys[1]].data.body_pos_w[:, self.hand_link_idx[robot_keys[1]]]  # End-effector 2

        ee_distance = torch.norm(ee_1_pos - ee_2_pos, p=2, dim=-1)  # Euclidean distance
        target_distance = 0.5  # Desired separation in meters
        separation_reward = -torch.abs(ee_distance - target_distance) * self.cfg.separation_reward_scale  # Penalize deviation

        # ✅ 4. Facing Reward - Encourage end-effectors to face each other
        ee_1_quat = self.robots[robot_keys[0]].data.body_quat_w[:, self.hand_link_idx[robot_keys[0]]]  # Shape: (num_envs, 4)
        ee_1_forward_vec = torch.tensor([0., 0., 1.], device=self.device).unsqueeze(0).expand(ee_1_quat.shape[0], -1)  # Shape: (num_envs, 3)
        ee_1_forward = tf_vector(ee_1_quat, ee_1_forward_vec)  # Apply quaternion rotation

        ee_2_quat = self.robots[robot_keys[1]].data.body_quat_w[:, self.hand_link_idx[robot_keys[1]]]  # Shape: (num_envs, 4)
        ee_2_forward_vec = torch.tensor([0., 0., -1.], device=self.device).unsqueeze(0).expand(ee_2_quat.shape[0], -1)  # Shape: (num_envs, 3)
        ee_2_forward = tf_vector(ee_2_quat, ee_2_forward_vec)  # Apply quaternion rotation

        facing_alignment = torch.sum(ee_1_forward * ee_2_forward, dim=-1)  # Dot product to measure alignment
        facing_reward = facing_alignment * self.cfg.facing_reward_scale  # Scale reward

        # ✅ Final Reward Computation
        total_reward = (
            height_reward  # Lift the cloth
            + proximity_reward
            # + separation_reward  # Keep end-effectors apart
            # + facing_reward  # Face each other
            # + smooth_motion_penalty  # Avoid jerky movements
        )

        # ✅ Log Debugging Information
        self.extras["log"] = {
            "height_reward": height_reward.mean(),
            "proximity_reward": proximity_reward.mean(),
            "separation_reward": separation_reward.mean(),
            "facing_reward": facing_reward.mean(),
            "smooth_motion_penalty": smooth_motion_penalty.mean(),
        }

        return total_reward


    def _reset_idx(self, env_ids: torch.Tensor | None):
        """Reset the state of multiple robots and environment objects properly."""
        super()._reset_idx(env_ids)

        for robot_key in self.robots.keys():  # Loop through all robots dynamically
            robot = self.robots[robot_key]

            # Generate randomized initial joint positions
            joint_pos = robot.data.default_joint_pos[env_ids] + sample_uniform(
                -0.125, 0.125, (len(env_ids), robot.num_joints), self.device
            )

            # Clamp within the robot's DOF limits
            joint_pos = torch.clamp(joint_pos, self.robot_dof_lower_limits[robot_key], self.robot_dof_upper_limits[robot_key])
            joint_vel = torch.zeros_like(joint_pos)

            # ✅ Reset the robot's joint positions correctly
            robot.set_joint_position_target(joint_pos, env_ids=env_ids)
            robot.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)

        # ✅ Reset Cloth
        for env_idx in range(self.num_envs):
            default_nodal_state = self._cloth_objects[env_idx].data.default_nodal_state_w.clone()
            self._cloth_objects[env_idx].write_nodal_state_to_sim(default_nodal_state)

        # ✅ Refresh intermediate values so that `_get_observations()` has updated values
        self._compute_intermediate_values(env_ids)

    def _get_observations(self) -> torch.Tensor:
        """Compute observations for all robots and return a single tensor."""

        observations = []  # List to store tensors for all robots

        for robot_key in self.robots.keys():  # Loop through all robots dynamically
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

        # ✅ Add cloth mesh points (100 points, equidistantly spaced) to the observation space
        cloth_points_obs = torch.zeros((self.num_envs, 100, 3), device=self.device)

        for env_idx in range(self.num_envs):
            cloth_positions = self._cloth_objects[env_idx].data.nodal_pos_w[0]  # (N, 3) tensor
            num_points = cloth_positions.shape[0]

            if num_points >= 100:
                # ✅ Select 100 equidistant indices
                indices = torch.linspace(0, num_points - 1, steps=100).long()
            else:
                # ✅ If fewer than 100 points, take all and pad with zeros
                indices = torch.arange(num_points)

            cloth_points_obs[env_idx, :len(indices)] = cloth_positions[indices]

        # ✅ Flatten the cloth points to add them to the observation tensor
        cloth_points_obs = cloth_points_obs.view(self.num_envs, -1)  # Shape (num_envs, 300)

        observations = torch.cat(observations, axis=-1)  # Shape (num_envs, 36)
        observations = torch.cat((observations, cloth_points_obs), axis=-1)  # Shape (num_envs, 336)

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
