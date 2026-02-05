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
from isaaclab.assets import Articulation, ArticulationCfg
from isaaclab.assets import ParticleClothObject, ParticleClothObjectCfg
from isaaclab.envs import DirectRLEnv, DirectRLEnvCfg
from isaaclab.managers import SceneEntityCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg, PhysxCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab.utils.math import sample_uniform, subtract_frame_transforms
from isaaclab.sensors import ContactSensorCfg
from pxr import UsdPhysics, UsdGeom, Gf, Sdf
import omni.usd
import omni.kit.commands
from omni.physx.scripts import physicsUtils, particleUtils
from isaaclab.controllers import DifferentialIKController, DifferentialIKControllerCfg


@configclass
class FrankaClothEnvCfg(DirectRLEnvCfg):
    # env
    episode_length_s = 7
    max_episode_length = episode_length_s*120
    decimation = 4
    action_space = 9
    observation_space = 18
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
            gpu_max_particle_contacts = 2**22 # Default is 2**20
        )
    )

    # scene
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=1, env_spacing=4.0, replicate_physics=False)

    cloth = ParticleClothObjectCfg(
                prim_path="/World/envs/env_.*/Cloth",
                init_state=ParticleClothObjectCfg.InitialStateCfg(pos=(0.0, 0, 0.1), rot=(1, 0, 0, 0)),
                spawn=sim_utils.UsdFileCfg(
                    usd_path="/workspace/isaaclab/source/isaaclab_tasks/isaaclab_tasks/sim/cloth_particle.usd",
                    scale=(1.0, 1.0, 1.0),
                ),
            )

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
            pos=(-1.0, 0.0, 0.0),
            rot=(1.0, 0.0, 0.0, 0.0),
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
            pos=(1.0, 0.0, 0.0),
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

    action_scale = 7.5 # default is 7.5
    dof_velocity_scale = 0.1

    # reward scales
    height_reward_scale = 30.0
    proximity_scale = 0.01
    smooth_penalty_scale = 0.1
    separation_reward_scale = 0.1
    facing_reward_scale = 0.1
    action_penalty_scale = 0.05


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
        self.default_cloth_positions = []
        self._prev_actions = torch.zeros((self.num_envs, 18), device=self.device)

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
        self.num_robots = 1  # Change this number to create more robots dynamically

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


    def create_cloth(self, stage, env_idx):
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

    def _setup_scene(self):
        # ✅ Create robots
        self._robot_1 = Articulation(self.cfg.robot_1)
        # self._robot_2 = Articulation(self.cfg.robot_2)

        self.scene.articulations["robot_1"] = self._robot_1
        # self.scene.articulations["robot_2"] = self._robot_2

        # ✅ Configure terrain
        self.cfg.terrain.num_envs = self.scene.cfg.num_envs
        self.cfg.terrain.env_spacing = self.scene.cfg.env_spacing
        self._terrain = self.cfg.terrain.class_type(self.cfg.terrain)

        # ✅ Setup contact sensors
        # self._contact_sensor1 = self.cfg.contact_sensor1.class_type(self.cfg.contact_sensor1)
        # self._contact_sensor2 = self.cfg.contact_sensor2.class_type(self.cfg.contact_sensor2)

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
        #     cloth_prim_path = self.create_cloth(self.scene.stage, env_idx)  # Create cloth for each env
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

        num_robots = len(self.robots)
        # Ensure actions are clamped between -1 and 1
        self.actions = actions.clone().clamp(-1.0, 1.0)
        self.actions = self.actions.reshape(self.num_envs, num_robots, -1)

        # Split the action tensor for each robot
        gripper_actions = self.actions[:,:,-1].clone()

        # ✅ Check proximity and modify gripper action
        cloth_positions = self._cloth.root_physx_view.get_positions().reshape(self.num_envs, -1, 3)
        for i, robot_key in enumerate(self.robots.keys()):  # Loop through all robots dynamically
            ee_pos_left = self.robots[robot_key].data.body_pos_w[:, self.left_finger_link_idx[robot_key]]
            ee_pos_right = self.robots[robot_key].data.body_pos_w[:, self.right_finger_link_idx[robot_key]]
            ee_cloth_distances_left = torch.norm(cloth_positions - ee_pos_left.unsqueeze(1), dim=-1)
            ee_cloth_distances_right = torch.norm(cloth_positions - ee_pos_right.unsqueeze(1), dim=-1)
            min_distance_left = ee_cloth_distances_left.min(1)[0]
            min_distance_right = ee_cloth_distances_right.min(1)[0]

            # ✅ If close enough to the cloth, start closing the gripper automatically
            close_mask = torch.bitwise_and(min_distance_left < 0.01, min_distance_right < 0.01)
            gripper_actions[close_mask, i] = 0.0  # Fully close gripper
            far_mask = torch.bitwise_and(min_distance_left > 0.03, min_distance_right > 0.03)
            gripper_actions[far_mask, i] = 1.0  # Fully open gripper

            self.actions[:,:,-1] = gripper_actions

            # ✅ Compute new target positions for each robot separately
            new_targets = (
                self.robot_dof_targets[robot_key]
                + self.robot_dof_speed_scales[robot_key] * self.dt
                * self.actions[:,i] * self.cfg.action_scale
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

        # self._prev_actions[:,0:9] = self.robot_dof_targets['robot_1']
        # self._prev_actions[:,9:] = self.robot_dof_targets['robot_2']

    # post-physics step calls

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        terminated = 0 # TODO
        truncated = self.episode_length_buf >= self.max_episode_length - 1
        return terminated, truncated

    def _get_rewards(self) -> torch.Tensor:
        """Compute rewards for lifting the cloth with stable grasping."""
        
        # Refresh intermediate values
        self._compute_intermediate_values()

        # ✅ Initialize Reward Components
        height_reward = torch.zeros(self.num_envs, device=self.device)
        proximity_reward = torch.zeros(self.num_envs, device=self.device)
        grasp_reward = torch.zeros(self.num_envs, device=self.device)
        lift_reward = torch.zeros(self.num_envs, device=self.device)

        for env_idx in range(self.num_envs):
            cloth_positions = self._cloth.root_physx_view.get_positions()[env_idx].reshape(-1, 3).clone()
            mean_cloth_height = cloth_positions[:, 2].mean()

            # ✅ 1. Proximity Reward - Encourage approaching the cloth
            min_distance = torch.tensor(500.0, device=self.device)
            for i, robot_key in enumerate(self.robots.keys()):
                ee_pos_l = self.robots[robot_key].data.body_pos_w[:, self.left_finger_link_idx[robot_key]]
                ee_pos_r = self.robots[robot_key].data.body_pos_w[:, self.right_finger_link_idx[robot_key]]
                ee_pos = (ee_pos_l + ee_pos_r) / 2
                ee_cloth_distances = torch.norm(cloth_positions.unsqueeze(0) - ee_pos[env_idx].unsqueeze(0), dim=-1)
                min_distance = min(min_distance, ee_cloth_distances.min())

                dist_reward = 1.0 / (1.0 + min_distance**2)
                dist_reward = dist_reward ** 2  # Strengthen effect
                proximity_reward[env_idx] += dist_reward * self.cfg.proximity_scale

                # ✅ 2. Grasping Reward - Encourage closing the gripper near the cloth
                gripper_action = self.actions[env_idx, i, -1]  # Gripper action
                if min_distance < 0.02 and gripper_action < 0.3:
                    grasp_reward[env_idx] = 10.0  # Strong reward for successful grasp

                # ✅ 3. Lift Reward - Encourage lifting **only after grasping**
                if min_distance < 0.02 and gripper_action < 0.3 and mean_cloth_height > 0.2:
                    lift_reward[env_idx] = mean_cloth_height * 20.0  # Scaled reward

                # ✅ 4. Height Reward - Keep lifting **stable over time**
                if mean_cloth_height > 0.05:
                    height_reward[env_idx] = mean_cloth_height * self.cfg.height_reward_scale
                    if mean_cloth_height > 0.1:
                        height_reward[env_idx] *= 10.0

        action_penalty = torch.sum(self.actions**2, dim=(-1,-2)) * self.cfg.action_penalty_scale

        # ✅ Compute Total Reward
        total_reward = (
            proximity_reward
            + grasp_reward
            + lift_reward
            + height_reward
            - action_penalty
        )

        # ✅ Log for Debugging
        self.extras["log"] = {
            "height_reward": height_reward.mean(),
            "proximity_reward": proximity_reward.mean(),
            "grasp_reward": grasp_reward.mean(),
            "lift_reward": lift_reward.mean(),
        }

        return total_reward

    def _reset_idx(self, env_ids: torch.Tensor | None):
        """Reset the state of multiple robots and environment objects properly."""
        super()._reset_idx(env_ids)

        if len(self.default_cloth_positions) == 0:
            self.default_cloth_positions = self._cloth.root_physx_view.get_positions().clone()

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

        self._cloth.root_physx_view.set_positions(
            self.default_cloth_positions.squeeze(), 
            indices=env_ids)

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

        # ✅ Add cloth mesh points (N points, equidistantly spaced) to the observation space
        N_points = 20
        cloth_points_obs = torch.zeros((self.num_envs, N_points, 3), device=self.device)

        for env_idx in range(self.num_envs):
            cloth_positions = self._cloth.root_physx_view.get_positions()[env_idx].reshape(-1, 3).clone()
            num_points = cloth_positions.shape[0]

            if num_points >= N_points:
                # ✅ Select 100 equidistant indices
                indices = torch.linspace(0, num_points - 1, steps=N_points).long()
            else:
                # ✅ If fewer than N_points points, take all and pad with zeros
                indices = torch.arange(num_points)

            cloth_points_obs[env_idx, :len(indices)] = cloth_positions[indices]

        # ✅ Flatten the cloth points to add them to the observation tensor
        cloth_points_obs = cloth_points_obs.view(self.num_envs, -1)
        observations = torch.cat(observations, axis=-1)
        # observations = torch.cat((observations, cloth_points_obs), axis=-1)

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
