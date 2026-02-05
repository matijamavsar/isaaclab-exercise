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
from isaaclab.assets import Articulation, ArticulationCfg
from isaaclab.assets import DeformableObjectCfg, RigidObjectCfg
from isaaclab.envs import DirectRLEnv, DirectRLEnvCfg
from isaaclab.managers import SceneEntityCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg, PhysxCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab.utils.math import sample_uniform, subtract_frame_transforms, matrix_from_quat
from isaaclab.utils.math import quat_slerp, quat_mul, quat_inv
from isaaclab.sensors import ContactSensorCfg
from pxr import UsdPhysics, UsdGeom, Gf, Sdf
import omni.usd
import omni.kit.commands
from omni.physx.scripts import physicsUtils, particleUtils
from isaaclab.controllers import DifferentialIKController, DifferentialIKControllerCfg
from isaacsim.core.utils.prims import get_prim_at_path


from isaaclab_assets import FRANKA_PANDA_HIGH_PD_CFG
import torch.nn.functional as F
import matplotlib.pyplot as plt

""" Run this training using
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py --task Bimanual-Franka-Place-Cloth-Direct-v0 --num_envs 128 --headless --video --video_interval 6000
"""

@configclass
class FrankaClothEnvCfg(DirectRLEnvCfg):
    # env
    episode_length_s = 6
    max_episode_length = episode_length_s*120
    decimation = 4
    action_space = 3
    observation_space = 55
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

    cloth = DeformableObjectCfg(
        prim_path="/World/envs/env_.*/cuboid",
        init_state=DeformableObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 0.356), rot=(0.5, 0.5, 0.5, 0.5)),
        spawn=sim_utils.UsdFileCfg(
            usd_path="/workspace/isaaclab/source/isaaclab_tasks/isaaclab_tasks/sim/deformable_for_joint_noCCD_freeCorners.usd",
            scale=(1, 1, 0.5),
        ),
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
    # FRANKA_PANDA_HIGH_PD_CFG.actuators['panda_shoulder'].stiffness = 100.0
    # FRANKA_PANDA_HIGH_PD_CFG.actuators['panda_forearm'].stiffness = 100.0
    robot_1 = FRANKA_PANDA_HIGH_PD_CFG.replace(
        prim_path="/World/envs/env_.*/Robot1",
        init_state=ArticulationCfg.InitialStateCfg(
            joint_pos={
                "panda_joint1": 2.2308e-03,
                "panda_joint2": -8.5887e-01,
                "panda_joint3": -5.7424e-04,
                "panda_joint4": -1.7517e+00,
                "panda_joint5": -5.5968e-04,
                "panda_joint6": 8.9285e-01,
                "panda_joint7": 7.8692e-01,
                "panda_finger_joint.*": 0.005,
            },
            pos=(0.0, -0.65, 0.0),
            rot=(0.7071, 0, 0, 0.7071),
            # rot=(1.0, 0.0, 0.0, 0.0),
        ),)
    
    robot_2 = FRANKA_PANDA_HIGH_PD_CFG.replace(
        prim_path="/World/envs/env_.*/Robot2",
        init_state=ArticulationCfg.InitialStateCfg(
            joint_pos={
                "panda_joint1": -3.4342e-02,
                "panda_joint2": -8.5897e-01,
                "panda_joint3": 8.7878e-03,
                "panda_joint4": -1.7517e+00,
                "panda_joint5": 8.5473e-03,
                "panda_joint6": 8.9276e-01,
                "panda_joint7": 7.6214e-01,
                "panda_finger_joint.*": 0.005,
            },
            pos=(0.0, 0.65, 0.0),
            rot=(0.7071, 0, 0, -0.7071),
            # rot=(1.0, 0.0, 0.0, 0.0),
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

    action_scale = 20 # default is 7.5
    dof_velocity_scale = 0.1
    filter_kernel_size = 7

    # reward scales
    spread_reward_scale = 5.0
    height_reward_scale = 1.0
    corner_x_reward_scale = 15.0
    downward_reward_scale = 1e-3
    direction_reward_scale = 30.0
    action_penalty_scale = 1e-4

    # Reward selection flags
    use_rewards = {
        "spread_reward":            True,
        "height_reward":            True,
        "corner_x_reward":          True,
        "gripper_downward_reward":  False,
        "direction_reward":         True,
        "action_penalty":           False,
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


class GaussianSmoother:
    """
    Fixed 1-D Gaussian convolution applied along the *time* axis of a tensor
    shaped (B, C, T) = (envs, joints, time_steps).
    """
    def __init__(self, kernel_size: int = 7, sigma: float = 2.0, device="cpu"):
        assert kernel_size % 2 == 1, "kernel_size must be odd to keep phase-alignment"
        self.kernel_size = kernel_size

        # build a 1-D gaussian → shape (1, 1, K)
        half = (kernel_size - 1) / 2
        t = torch.arange(kernel_size, device=device) - half
        kernel = torch.exp(-0.5 * (t / sigma) ** 2)
        kernel = kernel / kernel.sum()
        self.weight = kernel.view(1, 1, -1)               # (out_ch, in_ch/groups, K)

    @torch.no_grad()
    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """
        x : (envs, joints, time_steps)

        returns the *latest* smoothed value for every env & joint,
        shape (envs, joints)
        """
        x = x.permute(1,2,0)
        b, c, t = x.shape
        # (envs*joints, 1, time) so we can use grouped conv
        x_ = x.reshape(b * c, 1, t)
        y_ = F.conv1d(
            F.pad(x_, (self.kernel_size // 2, ) * 2, mode="replicate"),
            self.weight,
            groups=1,
        )
        y = y_.view(b, c, t)           # back to (envs, joints, time)
        return y[..., -1]              # only the most recent filtered value


class GaussianFilter:
    def __init__(self, device, kernel_size=15, sigma=None, channels=7):
        if kernel_size % 2 == 0:
            raise ValueError("kernel_size must be odd")
        self.kernel_size = kernel_size
        self.sigma = sigma or kernel_size / 3          # wider bell by default
        self.kernel = self._create_kernel(device, channels)

    def _create_kernel(self, device, channels):
        """Returns (channels,1,K) so each channel is filtered independently."""
        k = torch.arange(self.kernel_size, device=device).float()
        k = k - (self.kernel_size - 1) / 2
        kernel = torch.exp(-0.5 * (k / self.sigma) ** 2)
        kernel = kernel / kernel.sum()                 # L1‐normalise
        # shape (channels, 1, K) ⇒ depthwise conv on every joint
        return kernel.view(1, 1, -1).repeat(channels, 1, 1)

    @torch.no_grad()
    def filter(self, x):
        """
        x : (batch, channels, time)  OR  (batch, time, channels)
        Returns the *same* shape as input.
        """
        # ensure (B,C,T)
        if x.dim() != 3:
            raise ValueError("expected 3‑D tensor")
        if x.shape[2] < x.shape[1]:        # (B,T,C) → (B,C,T)
            x = x.permute(0, 2, 1)

        pad = self.kernel_size // 2
        y = F.conv1d(
            F.pad(x, (pad, pad), mode="replicate"),
            self.kernel,
            groups=x.shape[1],
        )
        return y                            # same (B,C,T)

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
        self.joints_created = False

        self.robot_entity_cfg = {}
        
        diff_ik_cfg = DifferentialIKControllerCfg(
            command_type="pose", use_relative_mode=False, ik_method="dls")
        self._ik_controller = {}

        # Initialize robot dictionaries
        self.robots = {}  # Store robot objects
        self.prev_joints = {}  # Store previous joint velocities
        self.filtered_dof_targets = {}  # Store smoothed action buffers
        self.robot_dof_targets = {}  # Store DOF targets
        self.rewards = {}
        self.filtered_trajectories = {}
        self.unfiltered_trajectories = {}

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

            self.filtered_trajectories[robot_key] = []
            self.unfiltered_trajectories[robot_key] = []

            # Adjust finger joint speed scale
            finger_joints = ["panda_finger_joint1", "panda_finger_joint2"]
            for finger_joint in finger_joints:
                joint_idx = self.robots[robot_key].find_joints(finger_joint)[0]
                self.robot_dof_speed_scales[robot_key][joint_idx] = 0.1

        # One-Euro filter for smoothing actions
        # self.filter = GaussianFilter(device=self.device, kernel_size=self.cfg.filter_kernel_size)
        self.smoother = GaussianSmoother(
            kernel_size=self.cfg.filter_kernel_size,
            sigma=self.cfg.filter_kernel_size / 3,   # rule-of-thumb
            device=self.device,
        )

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

    def _plot_comparison_dof_targets(self):
            """
            This function will plot a comparison of the filtered and unfiltered DOF targets
            for each robot, without blocking the training process.
            """
            # Create a new figure for plotting
            plt.figure(figsize=(10, 6))

            for robot_key in ['robot_1']: # self.robots.keys():
                # Extract the unfiltered and filtered DOF targets
                unfiltered_trajectories = torch.stack(self.unfiltered_trajectories[robot_key])[:,:4]
                filtered_trajectories = torch.stack(self.filtered_trajectories[robot_key])[:,:4]

                # Plot unfiltered DOF targets
                plt.plot(unfiltered_trajectories.cpu().numpy(), label=f'{robot_key} - Unfiltered', marker='o')

                plt.gca().set_prop_cycle(None)
                # Plot filtered DOF targets
                plt.plot(filtered_trajectories.cpu().numpy(), label=f'{robot_key} - Filtered', marker='x')

            # Set plot details
            plt.title('Comparison of Filtered and Unfiltered DOF Targets')
            plt.xlabel('Simulation step')
            plt.ylabel('Joint Position')
            plt.legend()
            plt.grid(True)
            plt.savefig('/workspace/isaaclab/source/isaaclab_tasks/isaaclab_tasks/filter_nofilter.jpg')
            print("Plotting!")

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

    def _abs_to_arm_poses(self, abs_pose: torch.Tensor, rel_pose: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Convert absolute + relative pose to individual poses for two arms.
        
        Args:
            abs_pose (torch.Tensor): (N, 7) absolute pose [x, y, z, qw, qx, qy, qz]
            rel_pose (torch.Tensor): (N, 7) relative pose [dx, dy, dz, qw, qx, qy, qz]
                                    dx, dy, dz are relative positions (from mid to left/right).
                                    The quaternion is the relative rotation from left to right.

        Returns:
            left_pose, right_pose: each (N, 7), pose for each gripper
        """
        assert abs_pose.shape[-1] == 7 and rel_pose.shape[-1] == 7, "Poses must be in (N, 7) format"

        # Extract absolute position and quaternion
        abs_pos = abs_pose[:, :3]
        abs_quat = abs_pose[:, 3:]

        # Get half of the relative offset vector
        rel_offset = rel_pose[:, :3]  # (N, 3)
        offset = 0.5 * rel_offset     # (N, 3)

        # Compute left and right positions
        left_pos = abs_pos - offset
        right_pos = abs_pos + offset

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
        left_quat = quat_mul(half_rel_quat, abs_quat)
        right_quat = quat_mul(quat_inv(half_rel_quat), abs_quat)

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

        # ✅ Setup contact sensors
        # self._contact_sensors = {}
        # self._contact_sensors['robot_1'] = self.cfg.contact_sensor1.class_type(self.cfg.contact_sensor1)
        # self._contact_sensors['robot_2'] = self.cfg.contact_sensor2.class_type(self.cfg.contact_sensor2)
        self._cloth = self.cfg.cloth.class_type(self.cfg.cloth)
        self._cloth_plain = self.cfg.cloth_plain.class_type(self.cfg.cloth_plain)
        self._handle_1 = self.cfg.handle_1.class_type(self.cfg.handle_1)
        self._handle_2 = self.cfg.handle_2.class_type(self.cfg.handle_2)
        self._free_corner_1 = self.cfg.free_corner_1.class_type(self.cfg.free_corner_1)
        self._free_corner_2 = self.cfg.free_corner_2.class_type(self.cfg.free_corner_2)

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
        self.actions = actions.clone().clamp(-0.1, 0.1)

        gripper_actions = torch.zeros((self.num_envs, 2), device=self.device)
        ee_jacobi_idx = 7

        absolute_pose = self._get_absolute_pose()
        
        # Add Cartesian change from self.actions
        absolute_pose[:,0] += self.actions[:,0]
        absolute_pose[:,2] += self.actions[:,1]
        
        absolute_pose[:, 0:3] = absolute_pose[:, 0:3] - self.scene.env_origins
        
        # Fixed Y-distance between grippers
        fixed_rel_pos = torch.tensor([0.0, 0.66, 0.0], device=self.device).repeat(self.num_envs, 1)
        fixed_rel_quat = torch.tensor([0, 0.0, 0.0, 1], device=self.device).repeat(self.num_envs, 1)
        relative_pose = torch.cat((fixed_rel_pos, fixed_rel_quat), dim=-1)

        # Transform from global into each robot coordinate system
        larm_pose, rarm_pose = self._abs_to_arm_poses(absolute_pose, relative_pose)
        larm_pos_b, larm_quat_b = larm_pose[:,0:3], larm_pose[:,3:]
        rarm_pos_b, rarm_quat_b = rarm_pose[:,0:3], rarm_pose[:,3:]

        rot_around_x = self.actions[:, 2]
        yaw_angles = torch.tanh(rot_around_x) * torch.pi  # scale input
        half_angles = yaw_angles / 2.0
        cos_half = torch.cos(half_angles)
        sin_half = torch.sin(half_angles)

        # Construct quaternion representing rotation around X axis
        # [qw, qx, qy, qz] = [cos(θ/2), sin(θ/2), 0, 0]
        rot_around_x_quat_1 = torch.stack([cos_half, sin_half, 
                                         torch.zeros_like(cos_half), torch.zeros_like(cos_half)], dim=-1)
        rot_around_x_quat_2 = torch.stack([cos_half, -sin_half, 
                                         torch.zeros_like(cos_half), torch.zeros_like(cos_half)], dim=-1)

        # rarm_goal_quat_b = quat_mul(rot_around_x_quat_1, rarm_quat_b)
        # larm_goal_quat_b = quat_mul(rot_around_x_quat_2, larm_quat_b)
        rarm_goal_quat_b = rarm_quat_b
        larm_goal_quat_b = larm_quat_b

        # for i in range(self.num_envs):
        #     rarm_quat_b[i] = quat_slerp(rarm_quat_b[i], rarm_goal_quat_b[i], self.dt*self.cfg.action_scale)
        #     larm_quat_b[i] = quat_slerp(larm_quat_b[i], larm_goal_quat_b[i], self.dt*self.cfg.action_scale)

        rarm_pose_b = torch.cat((rarm_pos_b, rarm_goal_quat_b), dim=-1)
        larm_pose_b = torch.cat((larm_pos_b, larm_goal_quat_b), dim=-1)

        self._ik_controller['robot_1'].set_command(larm_pose_b)
        self._ik_controller['robot_2'].set_command(rarm_pose_b)

        for i, robot_key in enumerate(self.robots.keys()):  # Loop through all robots dynamically
            self.robot_dof_targets[robot_key][:,-2:] = gripper_actions[:,i].unsqueeze(1).repeat(1,2)
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
            new_targets = (
                padded_joint_pos
                + self.robot_dof_speed_scales[robot_key] * self.dt
                * (self.robot_dof_targets[robot_key] - padded_joint_pos) * self.cfg.action_scale
            )
            # new_targets = self.robot_dof_targets[robot_key]

            # ✅ Clamp values within each robot's DOF limits
            new_targets = torch.clamp(new_targets, self.robot_dof_lower_limits[robot_key], self.robot_dof_upper_limits[robot_key])
            self.prev_joints[robot_key] = self.prev_joints[robot_key].roll(-1, dims=0)
            # self.prev_joints[robot_key][-2] = current_joint_pos
            self.prev_joints[robot_key][-1] = new_targets[:, 0:7]

            # self.filtered_trajectories[robot_key].append(self.smoother(self.prev_joints[robot_key])[0])
            # self.unfiltered_trajectories[robot_key].append(new_targets[0])

            # ✅ Apply One-Euro filter for smoother motion
            smoothed = self.smoother(self.prev_joints[robot_key])
            self.filtered_dof_targets[robot_key][..., :7] = smoothed
            self.filtered_dof_targets[robot_key][..., 7:] = new_targets[..., 7:]
            # ✅ OR DON'T
            # self.filtered_dof_targets[robot_key] = new_targets

            # ✅ Use the smoothed target instead of the raw target
            self.robot_dof_targets[robot_key] = self.filtered_dof_targets[robot_key]

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

    def _apply_action(self):
        """Apply joint position targets for multiple robots dynamically."""

        for robot_key in self.robots.keys():
            # ✅ Set joint position target separately for each robot
            self.robots[robot_key].set_joint_position_target(self.robot_dof_targets[robot_key])

    # post-physics step calls
    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        # Calculate distance between grippers
        robot_keys = list(self.robots.keys())  # Get robot keys dynamically
        ee_1_pos = self.robots[robot_keys[0]].data.body_pos_w[:, self.left_finger_link_idx[robot_keys[0]]]
        ee_2_pos = self.robots[robot_keys[1]].data.body_pos_w[:, self.left_finger_link_idx[robot_keys[1]]]

        ee_distance = torch.norm(ee_1_pos - ee_2_pos, dim=-1)

        is_not_first_step = self.episode_length_buf > 0
        terminated = torch.zeros_like(is_not_first_step)
        terminated[is_not_first_step] = torch.bitwise_or(
            ee_distance[is_not_first_step] > 1.2, ee_distance[is_not_first_step] < 0.4)
        truncated = self.episode_length_buf >= self.max_episode_length - 1
        return terminated, truncated

    def _get_corners(self):
        corners = [self._handle_1.root_physx_view.get_transforms()[:,0:3],
                   self._handle_2.root_physx_view.get_transforms()[:,0:3],
                   self._free_corner_1.root_physx_view.get_transforms()[:,0:3],
                   self._free_corner_2.root_physx_view.get_transforms()[:,0:3]]
        corners = torch.stack(corners).permute(1,0,2)
        return corners

    def _get_rewards(self) -> torch.Tensor:
        """Compute rewards for lifting the cloth with stable grasping."""
        
        # Refresh intermediate values
        # self._compute_intermediate_values()

        corners = self._get_corners()

        # Try to keep height low
        cloth_positions = self._cloth_plain.root_physx_view.get_nodal_positions().reshape(self.num_envs, -1, 3)
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
        spread_reward = (self.cfg.spread_reward_scale * r_spread)

        # ✅ New reward: Encourage free corners to move outward along X axis
        corners_in_env_origins = corners - self.scene.env_origins.unsqueeze(1)
        corner_x_vals = corners_in_env_origins[:, 2, 0] + corners_in_env_origins[:, 3, 0]
        corner_x_reward = corner_x_vals * self.cfg.corner_x_reward_scale

        # Define grasped and free corners (in this example, grasped corners are 0 and 1, free are 2 and 3)
        free_corners_x = corners[:, [2, 3], 0]
        grasped_corners_x = corners[:, [0, 1], 0]

        # Keep X of free corners higher
        direction_reward = free_corners_x.mean(-1) - grasped_corners_x.mean(-1)
        direction_reward *= self.cfg.direction_reward_scale

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
        action_penalty = torch.sum(self.actions**2, dim=(-1,-2)) * self.cfg.action_penalty_scale

        # Penalize joint velocities
        # action_penalty = torch.zeros(self.num_envs, device=self.device)
        # for robot_key in self.robots.keys():
        #     action_penalty += torch.sum(self.robots[robot_key].data.joint_vel**2, dim=(-1,-2)) * self.cfg.action_penalty_scale

        self.rewards["spread_reward"] = spread_reward
        self.rewards["height_reward"] = height_reward
        self.rewards["corner_x_reward"] = corner_x_reward
        self.rewards["downward_reward"] = gripper_downward_reward
        self.rewards["direction_reward"] = direction_reward
        self.rewards["action_penalty"] = action_penalty

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
            "direction_reward": direction_reward.mean(),
            "action_penalty": action_penalty.mean(),
        }

        return total_reward

    def _reset_idx(self, env_ids: torch.Tensor | None):
        """Reset the state of multiple robots and environment objects properly."""
        super()._reset_idx(env_ids)

        # if len(self.filtered_trajectories['robot_1']) > 0:
        #     self._plot_comparison_dof_targets()
        #     breakpoint()

        for robot_key in self.robots.keys():  # Loop through all robots dynamically
            robot = self.robots[robot_key]
            joint_pos = robot.data.default_joint_pos[env_ids].clone()
            joint_vel = robot.data.default_joint_vel[env_ids].clone()
            robot.set_joint_position_target(joint_pos, env_ids=env_ids)
            robot.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)
            self.filtered_trajectories[robot_key] = []
            self.unfiltered_trajectories[robot_key] = []

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

        # ✅ Add cloth corners to the observation space
        cloth_positions = self._cloth_plain.root_physx_view.get_nodal_positions().reshape(self.num_envs, -1, 3)

        corners = self._get_corners()
        corners = corners.reshape(self.num_envs, -1)

        observations = torch.cat((observations, corners), axis=-1)

        # Append absolute pose
        observations = torch.cat((observations, self._get_absolute_pose()), axis=-1)

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
