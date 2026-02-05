# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
from __future__ import annotations

import os
import time
import random
from datetime import datetime

import cv2
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from torchvision.transforms.functional import resize
from isaaclab.assets import Articulation, ArticulationCfg
from isaaclab.assets import ParticleClothObject, ParticleClothObjectCfg, RigidObjectCfg
from isaaclab.controllers import DifferentialIKController, DifferentialIKControllerCfg
import isaaclab.sim as sim_utils
from isaaclab.envs import DirectRLEnv, DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import PhysxCfg, RenderCfg, SimulationCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.utils.math import subtract_frame_transforms
from isaacsim.core.utils.prims import get_prim_at_path
from pxr import UsdGeom, Gf, Sdf, UsdGeom, UsdShade, UsdLux, UsdPhysics
from omni.physx.scripts import physicsUtils, particleUtils
import omni.usd
import omni.kit.commands
from isaaclab_assets import FRANKA_PANDA_HIGH_PD_CFG
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab.actuators.actuator_cfg import ImplicitActuatorCfg
import carb

from curobo.types.base import TensorDeviceType
from curobo.types.math import Pose
from curobo.types.robot import RobotConfig
from curobo.util_file import get_robot_configs_path, join_path, load_yaml
from curobo.wrap.reacher.ik_solver import IKSolver, IKSolverConfig

import curobo
curobo.util.logger.setup_logger('error', 'curobo')

from torchvision.utils import save_image  # kept because original code used it in logs
from scipy.spatial import ConvexHull      # kept; convex hull code gated behind ifs
from torchvision.models import resnet18, ResNet18_Weights, resnet50, ResNet50_Weights

# Local utilities
from .dmp_integrator import BatchDMPIntegrator
from .min_jerk_traj import generate_minimum_jerk

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
    sim_steps = 120
    episode_length_s = 8
    max_episode_length = episode_length_s * sim_steps
    num_updates_per_episode = 1
    num_steps_to_settle = sim_steps * 2
    num_tries = 3
    decimation = max_episode_length // num_updates_per_episode
    action_space = 28
    observation_space = 100
    state_space = 0
    enable_camera_recording = False
    plot_trajectories = False
    nice_render = True
    use_resnet = False
    use_cloth_positions = False
    seed = 42

    torch.manual_seed(seed)

    # simulation
    sim: SimulationCfg = SimulationCfg(
        dt=1 / sim_steps,
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

    if enable_camera_recording:
        from isaaclab.sensors import CameraCfg
        camera = CameraCfg(
            prim_path="/World/Camera",
            offset=CameraCfg.OffsetCfg(
                pos=(-6.0, -6.0, 2.0),
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

        realsense_mesh = sim_utils.UsdFileCfg(
            usd_path=f"{ISAAC_NUCLEUS_DIR}/Sensors/Intel/RealSense/rsd455.usd",
            scale=(1.0, 1.0, 1.0)
        )

        realsense_cfg = RigidObjectCfg(
            prim_path="/World/envs/env_.*/RealsenseD455",
            spawn=realsense_mesh,
            init_state=RigidObjectCfg.InitialStateCfg(
                pos=(0.0, 0.0, 1.0),
                rot=(0.7071, 0, 0.7071, 0),
            ),
        )

        ee_camera = CameraCfg(
            prim_path="/World/envs/env_.*/RealsenseD455/RSD455/Camera_Pseudo_Depth",
            spawn=None,
            data_types=["depth"],
            width=256,
            height=256
        )

        # ee_camera = CameraCfg(
        #     # mount under the Franka hand so it inherits the link pose
        #     prim_path="/World/envs/env_.*/ee_camera",
        #     offset=CameraCfg.OffsetCfg(
        #         pos=(0.0, 0.0, 1.2),
        #         rot=(0.7071,0,0.7071,0),
        #         convention="world",
        #     ),
        #     data_types=["depth"],
        #     spawn=sim_utils.PinholeCameraCfg(
        #         focal_length=16.0, focus_distance=100.0,
        #         horizontal_aperture=12.0, clipping_range=(0.05, 20.0)
        #     ),
        #     width=256,    # smaller to keep obs light-weight
        #     height=256,
        # )

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
            pos=(0.0, -0.5, 0.0),
            rot=(0.7071, 0, 0, 0.7071),
        ),
        # actuators={
        #     "panda_shoulder": ImplicitActuatorCfg(
        #         joint_names_expr=["panda_joint[1-4]"],
        #         effort_limit=87.0,
        #         velocity_limit=2.175,
        #         stiffness=50.0,
        #         damping=10.0,
        #     ),
        #     "panda_forearm": ImplicitActuatorCfg(
        #         joint_names_expr=["panda_joint[5-7]"],
        #         effort_limit=12.0,
        #         velocity_limit=2.61,
        #         stiffness=50.0,
        #         damping=10.0,
        #     ),
        #     "panda_hand": ImplicitActuatorCfg(
        #         joint_names_expr=["panda_finger_joint.*"],
        #         effort_limit=200.0,
        #         velocity_limit=0.2,
        #         stiffness=50,
        #         damping=10,
        #     ),
        # },
    )

    action_scale = 15

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
        "spread_reward": Reward(True, 5.0),
        "height_reward": Reward(True, 1e-1),
        "corner_x_reward": Reward(False, 10.0),
        "direction_reward": Reward(False, 15.0),
        "action_penalty": Reward(False, 1e-2),
        "target_area_reward": Reward(True, 1.0),
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
        self.num_steps_to_settle = self.cfg.num_steps_to_settle

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

        # DMPs
        self.dmp_initialized = False
        self.dmp_integrator = BatchDMPIntegrator(N_basis=7, dof=3, device=self.device)
        self.prev_y = torch.zeros((self.num_envs, 3), device=self.device)

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

        if self.cfg.use_resnet:
            weights = ResNet18_Weights.DEFAULT
            self.resnet_preprocess = weights.transforms()
            self.resnet_model = resnet18(weights=weights).to(self.device)
            self.resnet_model.eval()

        tensor_args = TensorDeviceType()

        config_file = load_yaml(join_path(get_robot_configs_path(), "franka.yml"))
        urdf_file = config_file["robot_cfg"]["kinematics"]["urdf_path"]
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

        self.dof_lower_limits = self._robot.data.soft_joint_pos_limits[0, :, 0].to(self.device)
        self.dof_upper_limits = self._robot.data.soft_joint_pos_limits[0, :, 1].to(self.device)

        # simple speed scaling (slow fingers)
        self.dof_speed_scales = torch.ones_like(self.dof_lower_limits)
        for fname in ["panda_finger_joint1", "panda_finger_joint2"]:
            j_idx = self._robot.find_joints(fname)[0]
            self.dof_speed_scales[j_idx] = 0.1

        # low-pass filter over all robot DOFs
        self.lp = BiquadLP2Batch(
            self.scene.cfg.num_envs,
            self._robot.num_joints,
            fc=1.0,
            dt=self.physics_dt,
            device=self.device,
        )

        self.robot_dof_targets = torch.zeros((self.num_envs, self._robot.num_joints), device=self.device)
        self.gripper_actions = torch.zeros((self.num_envs, 2), device=self.device)
        self.left_finger_link_idx = self._robot.find_bodies("panda_leftfinger")[0][0]
        self.right_finger_link_idx = self._robot.find_bodies("panda_rightfinger")[0][0]

        # store which body is EE (hand link)
        self.robot_entity_cfg = type("Tmp", (), {})()  # simple struct
        self.robot_entity_cfg.body_ids = [self._robot.find_bodies("panda_hand")[0]]
        self.robot_entity_cfg.joint_ids = torch.arange(self._robot.num_joints, device=self.device)

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


    def create_cloth(self, stage, env_idx):
        cloth_mesh_path = Sdf.Path(f"/World/envs/env_{env_idx}/Cloth")
        particle_material_path = Sdf.Path("/World/particleMaterial")

        # ✅ Create a mesh that will be turned into cloth
        # plane_resolution = 40
        plane_resolution = 50
        # plane_width = float(torch.rand(1).item() * (70 - 45) + 45)
        plane_width = 40

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

        # Randomize parameters
        # stretchStiffness = float(torch.rand(1).item() * (1_000_000 - 500_000) + 500_000)
        # bendStiffness    = float(torch.rand(1).item() * (1000 - 100) + 100)
        # shearStiffness   = float(torch.rand(1).item() * (1000 - 500) + 500)
        # damping          = float(torch.rand(1).item() * (0.9 - 0.5) + 0.5)

        # stretchStiffness = float(torch.rand(1).item() * (1_000_000 - 50_000) + 50_000)
        # bendStiffness    = float(torch.rand(1).item() * (100 - 10) + 10)
        # shearStiffness   = float(torch.rand(1).item() * (500 - 50) + 50)
        # damping          = float(torch.rand(1).item() * (0.9 - 0.25) + 0.25)

        # stretchStiffness = 1000000.0
        # bendStiffness = 20.0
        # shearStiffness = 100.0
        # damping = 0.8

        stretchStiffness = 1000000.0
        bendStiffness = 1000.0
        shearStiffness = 1000.0
        damping = 0.8

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
        particle_mass = 0.8 / (plane_resolution**2)
        num_verts = len(cloth_mesh.GetPointsAttr().Get())
        mass = particle_mass * num_verts
        massApi = UsdPhysics.MassAPI.Apply(cloth_mesh.GetPrim())
        massApi.GetMassAttr().Set(mass)

        # Set color and access to positions
        cloth_prim = self.scene.stage.GetPrimAtPath(cloth_mesh_path)
        cloth_color_attr = cloth_prim.GetAttribute("primvars:displayColor")
        cloth_color_attr.Set([Gf.Vec3f(0.2, 0.0, 0.220)])

        return cloth_mesh_path, plane_width

    def _setup_scene(self):
        # Terrain
        self.cfg.terrain.num_envs = self.scene.cfg.num_envs
        self.cfg.terrain.env_spacing = self.scene.cfg.env_spacing
        self._terrain = self.cfg.terrain.class_type(self.cfg.terrain)
        self._robot = Articulation(self.cfg.robot)
        self.scene.articulations["robot"] = self._robot

        # Clone envs
        self.scene.clone_environments(copy_from_source=True)

        # ----------------------------
        # ✅ Create Particle System ONCE
        # ----------------------------
        stage = self.scene.stage
        particle_system_path = Sdf.Path("/World/particleSystem")

        restOffset = 0.003
        contactOffset = restOffset * 2
        particleContactOffset = contactOffset
        solidRestOffset = 0.9 * particleContactOffset
        fluidRestOffset = 0.9 * particleContactOffset

        particleUtils.add_physx_particle_system(
            stage=stage,
            particle_system_path=particle_system_path,
            contact_offset=contactOffset,
            rest_offset=restOffset,
            particle_contact_offset=particleContactOffset,
            solid_rest_offset=solidRestOffset,
            fluid_rest_offset=fluidRestOffset,
            solver_position_iterations=16,
            simulation_owner="physicsScene",
        )

        # ✅ Create Particle Material ONCE
        particle_material_path = Sdf.Path("/World/particleMaterial")
        particleUtils.add_pbd_particle_material(stage, particle_material_path)
        particleUtils.add_pbd_particle_material(stage, particle_material_path, drag=0.01, lift=0.01, friction=0.6)

        physicsUtils.add_physics_material_to_prim(
            stage, stage.GetPrimAtPath(particle_system_path), particle_material_path
        )

        # ----------------------------
        # ✅ Create a Single Material for Cloth
        # ----------------------------
        material_path = particle_material_path
        # material_path = Sdf.Path("/World/Materials/ClothMaterial")

        # ✅ Create a Shader for the material
        shader_path = material_path.AppendPath("Shader")
        shader = UsdShade.Shader.Define(stage, shader_path)
        shader.CreateIdAttr("UsdPreviewSurface")  # Use USD's Preview Surface Shader

        # ✅ Set shader properties (e.g., color, roughness, metallic)
        shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(1.0, 0.2, 0.2))
        shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.8)
        shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(0.0)

        # ✅ Create cloth for each environment
        self._cloth_objects = []
        self.cloth_lengths = torch.zeros(self.scene.cfg.num_envs, device=self.device)
        for env_idx in range(self.scene.num_envs):
            cam_prim = get_prim_at_path(f"/World/envs/env_{env_idx}/RealsenseD455/RSD455")
            # if not cam_prim.HasAPI(UsdPhysics.RigidBodyAPI):
            #     UsdPhysics.RigidBodyAPI.Apply(cam_prim)
            # UsdPhysics.RigidBodyAPI(cam_prim).CreateKinematicEnabledAttr(True)

            cloth_prim_path, plane_width = self.create_cloth(self.scene.stage, env_idx)  # Create cloth for each env
            cloth_cfg = ParticleClothObjectCfg(
                prim_path=f"/World/envs/env_{env_idx}", 
                spawn=None
                )
            self._cloth_objects.append(ParticleClothObject(cloth_cfg))

            # Randomize cloth length (scale Y) per env
            cloth_prim_path = f"/World/envs/env_{env_idx}/Cloth"
            cloth_prim = get_prim_at_path(cloth_prim_path)

            # # Uncomment this for scale randomization
            # random_scale = torch.tensor([torch.rand(1).item() * (1.0 - 0.5) + 0.5, 
            #                              torch.rand(1).item() * (1.0 - 0.5) + 0.5, 
            #                              1.0])
            # cloth_prim.GetAttribute("xformOp:scale").Set(Gf.Vec3f(*random_scale.tolist()))

            cloth_prim.GetAttribute("xformOp:translate").Set(
                Gf.Vec3f(0.2, 0.0, 0.5))
            # You can comment this to prevent rotation randomization
            cloth_prim.GetAttribute("xformOp:orient").Set(
                Gf.Quatd(0.5 - 0.01 + 0.02*torch.rand(1).item(), 
                         0.5 - 0.01 + 0.02*torch.rand(1).item(), 
                         0.5 - 0.01 + 0.02*torch.rand(1).item(), 
                         0.5 - 0.01 + 0.02*torch.rand(1).item())
            )
            # And uncomment this
            # cloth_prim.GetAttribute("xformOp:orient").Set(
            #     Gf.Quatd(0.55, 0.5, 0.5, 0.5))
            # self.cloth_lengths[env_idx] = plane_width / 100 # because it's in cm

        # Camera
        if self.cfg.enable_camera_recording:
            self._camera = self.cfg.camera.class_type(self.cfg.camera)
            self._realsense_camera = self.cfg.realsense_cfg.class_type(self.cfg.realsense_cfg)
            self._ee_camera = self.cfg.ee_camera.class_type(self.cfg.ee_camera)

             # --- Randomize camera positions (±10 cm in each axis) ---
            random_range = 0.10  # 10 cm in meters

            for env_idx in range(self.scene.num_envs):
                # Example paths for your cameras
                cam_paths = [
                    f"/World/envs/env_{env_idx}/RealsenseD455/RSD455",
                    f"/World/envs/env_{env_idx}/RealsenseD455/RSD455/Camera_Pseudo_Depth",
                    f"/World/Camera"
                ]

                for path in cam_paths:
                    prim = get_prim_at_path(path)
                    if not prim or not prim.IsValid():
                        continue

                    # Get current translation
                    try:
                        pos = prim.GetAttribute("xformOp:translate").Get()
                        if pos is None:
                            continue

                        # Add random offset ±0.1 m per axis
                        dx = (random.random() - 0.5) * 2 * random_range
                        dy = (random.random() - 0.5) * 2 * random_range
                        dz = (random.random() - 0.5) * 2 * random_range

                        new_pos = Gf.Vec3f(pos[0] + dx, pos[1] + dy, pos[2] + dz)
                        prim.GetAttribute("xformOp:translate").Set(new_pos)
                        print(f"[Camera Randomized] {path}: Δ=({dx:+.2f}, {dy:+.2f}, {dz:+.2f})")
                    except Exception as e:
                        print(f"Could not randomize camera {path}: {e}")
        
            ### TODO: Randomize camera properties

            for env_idx in range(self.scene.num_envs):
                cam_prim = get_prim_at_path(f"/World/envs/env_{env_idx}/RealsenseD455/RSD455")
                if not cam_prim.HasAPI(UsdPhysics.RigidBodyAPI):
                    UsdPhysics.RigidBodyAPI.Apply(cam_prim)
                UsdPhysics.RigidBodyAPI(cam_prim).CreateKinematicEnabledAttr(True)

        self.set_cloth_texture("/workspace/isaaclab/source/isaaclab_tasks/isaaclab_tasks/sim/cloth_texture/cloth_texture_1.jpeg",
            roughness=0.6)

        # Light
        if not self.cfg.nice_render:
            light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
            light_cfg.func("/World/Light", light_cfg)

            # Hide world environment (optional)
            stage = omni.usd.get_context().get_stage()
            env_prim = stage.GetPrimAtPath("/World/ground/Environment")
            if env_prim:
                UsdGeom.Imageable(env_prim).GetVisibilityAttr().Set(UsdGeom.Tokens.invisible)
        else:
            self.setup_clean_stage(renderer="HydraStorm")  # simplest, no noise/ghosting

    def _get_ee_pose_w(self):
        """End-effector pose (world) of the Franka hand."""
        ee_pose_w = self._robot.data.body_state_w[:, self.left_finger_link_idx, 0:7]
        return ee_pose_w.squeeze(1)

    def _get_corners(self):
        cloth_positions = self._get_cloth_positions_shaped()
        side_len = int(torch.sqrt(torch.tensor(cloth_positions.shape[1])).to(self.device))
        corners = cloth_positions[:, [-side_len, -1, 0, side_len - 1], :]
        return corners

    def _get_node_idx_from_start_pos(self, dmp_parameters, radius_m: float = 0.02):
        """
        For each env, return the index of the cloth node that:
        1) lies within 'radius_m' (default 2 cm) of start_xy in the XY plane, and
        2) has the highest Z among those nodes.
        If no nodes are within the circle, fall back to the nearest node in XY.

        dmp_parameters: [E, ...], with start_xy in [:, 0:2] (x,y) in meters.
        Returns:
            idx: LongTensor [E] of node indices
            cloth_positions: Tensor [E, N, 3] of node positions
        """
        # [E*N, 3] -> [E, N, 3]
        cloth_positions = self._get_cloth_positions_shaped()

        start_xy = dmp_parameters[:, 0:2].to(cloth_positions.device)  # [E, 2]
        cloth_xy = cloth_positions[..., :2]                            # [E, N, 2]
        z = cloth_positions[..., 2]                                    # [E, N]

        # Distances in XY
        diff = cloth_xy - start_xy[:, None, :]                         # [E, N, 2]
        dist2 = (diff * diff).sum(dim=-1)                              # [E, N]

        # Nodes inside the circle of radius_m
        r2 = radius_m * radius_m
        inside = dist2 <= r2                                           # [E, N] bool

        # Among inside nodes, pick the highest Z
        neg_inf = torch.tensor(float("-inf"), device=cloth_positions.device, dtype=z.dtype)
        z_masked = torch.where(inside, z, neg_inf)                     # [E, N]
        idx_top_inside = z_masked.argmax(dim=1)                        # [E]

        # Fallback: nearest node if no inside candidates
        has_inside = inside.any(dim=1)                                 # [E] bool
        idx_nearest = dist2.argmin(dim=1)                              # [E]

        idx = torch.where(has_inside, idx_top_inside, idx_nearest)     # [E]
        return idx, cloth_positions

    # -------------------------
    # RL hooks
    # -------------------------

    def _pre_physics_step(self, actions: torch.Tensor):
        self.actions = actions
        self.action_steps *= 0

    def _apply_action(self):
        """
        PHASE STRUCTURE:
        Phase 1: Cloth settling                      [0 .. S-1]
        Phase 2: Robot approaches cloth node         [S .. 2S-1]
        Phase 3: Gripper closing                     [2S .. 2S + S/2 - 1]
        Phase 4: DMP integration + robot tracking    [2S + S/2 .. end_of_dmp]
        Phase 5: Robot opens gripper + final settle  [end_of_dmp .. end_of_dmp + S]
        """

        S = self.num_steps_to_settle
        step = self.action_steps[0]
        clamp_loc = 0.2

        # ------------------------
        # Phase 1: Cloth settling
        # ------------------------
        if step < S:
            # Do NOT step DMP, do NOT move robot.
            # Keep gripper open.
            self.gripper_actions[:, 0] = 0.04
            self.gripper_actions[:, 1] = 0.04
            self.action_steps += 1
            self.iteration_step += 1
            return

        # ---------------------------------------------------------
        # Phase 2 START: at exactly step == S
        #   - detect cloth node
        #   - initialize DMP
        #   - take ONE DMP step → initial pose
        #   - robot begins moving toward that one pose (open gripper)
        # ---------------------------------------------------------
        if step == S:
            dmp_parameters = self.actions.clone()

            # Clamp goal XY + Z
            dmp_parameters[:, 3] = dmp_parameters[:, 3].clamp(-clamp_loc, clamp_loc)
            dmp_parameters[:, 4] = dmp_parameters[:, 4].clamp(-clamp_loc, clamp_loc)
            dmp_parameters[:, 5] = dmp_parameters[:, 5].clamp(0.3, 0.5)

            # penalty buffer
            self.action_penalties[self.action_steps] = dmp_parameters[:, 6:-1].abs().mean()

            # Convert start+goal into world frame for node detection
            dmp_parameters[:, 0:3] += self.scene.env_origins
            dmp_parameters[:, 3:6] += self.scene.env_origins

            self.grab_node_idx, cloth_pos = self._get_node_idx_from_start_pos(dmp_parameters)

            # Overwrite DMP starting position with detected node world coords
            for i in range(self.num_envs):
                dmp_parameters[i, 0] = cloth_pos[i, self.grab_node_idx[i], 0]
                dmp_parameters[i, 1] = cloth_pos[i, self.grab_node_idx[i], 1]
                dmp_parameters[i, 2] = cloth_pos[i, self.grab_node_idx[i], 2] + 0.12

            # Back to env-local coords
            dmp_parameters[:, 0:3] -= self.scene.env_origins
            dmp_parameters[:, 3:6] -= self.scene.env_origins

            for i in range(self.num_envs):
                dmp_parameters[i, 0] = cloth_pos[i, self.grab_node_idx[i], 0] # * 0 + 0.25
                dmp_parameters[i, 1] = cloth_pos[i, self.grab_node_idx[i], 1] # * 0 + 0.0

            # DMP duration (minus approach+close time)
            # Full Episode = episode_length_s
            # Remove 2S + S/2 of time in seconds.
            dmp_tau = torch.zeros(self.num_envs, device=self.device) + self.cfg.episode_length_s
            dmp_tau -= (2.5 * S) / self.cfg.sim_steps
            self.actions[:, -1] = dmp_tau

            # Reset integrator
            reset_idx = torch.arange(self.num_envs, device=self.device)
            self.dmp_integrator.reset_indices(
                reset_idx, dmp_parameters, dmp_tau, dt=self.physics_dt, variant=1
            )
            self.dmp_initialized = True
            # self.actions = dmp_parameters

            # One DMP step to get initial target
            t, y, _, _ = self.dmp_integrator.step()
            self.dmp_integrator.x[t >= self.actions[:, -1]] = 0.1353

            y_world = y + self.scene.env_origins
            y_robot = y_world.clone() - self.scene.env_origins

            # Initialize low-pass at current joint state
            q0 = self._robot.data.joint_pos[:, : self._robot.num_joints]
            self.lp.reset(q0)

            # Save robot target for entire Phase 2
            self.initial_robot_target = y_robot.clone()

            # Gripper OPEN during approach
            self.gripper_actions[:, 0] = 0.04
            self.gripper_actions[:, 1] = 0.04

            # Move robot
            self.inverse_kinematics_curobo(y_robot)
            self.robot_dof_targets[:, -2:] = self.gripper_actions
            self.robot_dof_targets = self.lp.step(self.robot_dof_targets)

            self._robot.write_joint_position_to_sim(self.robot_dof_targets)
            self._robot.set_joint_position_target(self.robot_dof_targets)

            self.action_steps += 1
            self.iteration_step += 1
            return

        # -------------------------------------------------------------
        # Phase 2 continued: robot moves toward initial DMP pose
        # -------------------------------------------------------------
        if S < step < 2 * S:
            y_robot = self.initial_robot_target

            self.gripper_actions[:, 0] = 0.04
            self.gripper_actions[:, 1] = 0.04  # still OPEN

            self.inverse_kinematics_curobo(y_robot)
            self.robot_dof_targets[:, -2:] = self.gripper_actions
            self.robot_dof_targets = self.lp.step(self.robot_dof_targets)

            self._robot.write_joint_position_to_sim(self.robot_dof_targets)
            self._robot.set_joint_position_target(self.robot_dof_targets)

            self.action_steps += 1
            self.iteration_step += 1
            return

        # -----------------------------------------------
        # Phase 3: Gripper CLOSING (for S/2 steps)
        # -----------------------------------------------
        if 2 * S <= step < 2 * S + S // 2:
            # Keep robot frozen at initial pose
            y_robot = self.initial_robot_target

            self.gripper_actions[:, 0] = 0.0
            self.gripper_actions[:, 1] = 0.0

            self.inverse_kinematics_curobo(y_robot)
            self.robot_dof_targets[:, -2:] = self.gripper_actions
            self.robot_dof_targets = self.lp.step(self.robot_dof_targets)

            self._robot.write_joint_position_to_sim(self.robot_dof_targets)
            self._robot.set_joint_position_target(self.robot_dof_targets)

            self.action_steps += 1
            self.iteration_step += 1
            return

        # ------------------------------------------------------
        # Phase 4 — Normal DMP integration every step + robot IK
        # ------------------------------------------------------
        # Compute how many steps DMP should run
        dmp_total_steps = int(self.actions[0, -1] * self.cfg.sim_steps)

        if 2 * S + S // 2 <= step < 2 * S + S // 2 + dmp_total_steps:
            t, y, _, _ = self.dmp_integrator.step()
            self.dmp_integrator.x[t >= self.actions[:, -1]] = 0.1353

            y_world = y + self.scene.env_origins
            y_robot = y_world.clone() - self.scene.env_origins
            y_robot[:, 2] = y_robot[:, 2].clamp(0.3, 0.6)

            self.inverse_kinematics_curobo(y_robot)
            self.robot_dof_targets = self.lp.step(self.robot_dof_targets)

            self._robot.write_joint_position_to_sim(self.robot_dof_targets)
            self._robot.set_joint_position_target(self.robot_dof_targets)

            self.action_steps += 1
            self.iteration_step += 1
            return

        # ------------------------------------------------------
        # Phase 5 — Open gripper again + settle cloth for S steps
        # ------------------------------------------------------
        if step >= 2 * S + S // 2 + dmp_total_steps:
            # Keep robot steady, open fingers
            self.gripper_actions[:, 0] = 0.04
            self.gripper_actions[:, 1] = 0.04
            self.robot_dof_targets[:, -2:] = self.gripper_actions

            # Robot holds last pose (no more IK)
            self._robot.write_joint_position_to_sim(self.robot_dof_targets)
            self._robot.set_joint_position_target(self.robot_dof_targets)

            self.action_steps += 1
            self.iteration_step += 1
            return

        # If beyond Phase 5 duration → do nothing (episode likely ending)
        self.action_steps += 1
        self.iteration_step += 1

    def inverse_kinematics_curobo(self, y_3d: torch.Tensor):
        """
        y_2d: [E, 2] tensor with desired (x, z) in WORLD frame (per env).
        Robot EE y (world) is kept at env center Y.
        """
        current_ee = self._get_ee_pose_w()
        target_absolute_pose = current_ee.clone() * 0.0

        # target x,z from y_2d, y fixed at env center (0 in env-local)
        target_absolute_pose[:, 0] = y_3d[:, 0]              # x
        target_absolute_pose[:, 1] = y_3d[:, 1]              # x
        target_absolute_pose[:, 2] = y_3d[:, 2]              # z
        target_absolute_pose[:, 0:3] += self.scene.env_origins

        # orientation: point -Y (same as first script’s [0,1,0,0])
        target_absolute_pose[:, 3:] = torch.tensor([0, 1, 0, 0], device=self.device)

        root_pose_w = self._robot.data.root_state_w[:, 0:7]
        rarm_pos_b, _ = subtract_frame_transforms(
            root_pose_w[:, 0:3], root_pose_w[:, 3:7],
            target_absolute_pose[:, 0:3], target_absolute_pose[:, 3:7]
        )

        rarm_goal_quat_b = torch.zeros((self.num_envs, 4), device=self.device)
        rarm_goal_quat_b[:, 1] = 1.0

        with torch.inference_mode(False):
            current_joints = self._robot.data.joint_pos[:, 0:7].clone()
            # current_joints = self._robot.data.joint_pos[:, self.robot_entity_cfg.joint_ids].clone()
            goal = Pose(rarm_pos_b.clone(), rarm_goal_quat_b.clone())
            result = self.ik_solver.solve_batch(
                goal,
                retract_config=current_joints,
                seed_config=current_joints.unsqueeze(0).repeat(self.num_envs, 1, 1),
            )
            q_solution = result.solution.squeeze(1)

        # keep gripper as-is
        self.robot_dof_targets[:, -2:] = self.gripper_actions

        new_targets = q_solution
        # pad with finger joints
        new_targets = F.pad(input=new_targets, pad=(0, 2), mode="constant", value=0.0)
        new_targets = torch.clamp(new_targets, self.dof_lower_limits, self.dof_upper_limits)
        self.robot_dof_targets = new_targets

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        terminated = 0
        truncated = self.episode_length_buf >= self.max_episode_length * self.cfg.num_tries
        if truncated.any():
            truncated[:] = True
        return terminated, truncated

    def compute_rewards(self):
        corners = self._get_corners()
        cloth_positions = self._get_cloth_positions_shaped()
        mean_cloth_height = cloth_positions[:, :, 2].mean(dim=-1)

        rewards = {}

        # target box in env-local coordinates
        x_min, x_max = -0.35, 0.35
        y_min, y_max = -0.35, 0.35

        # positions: [E, N, 3] world; convert to env-local (subtract env origins)
        xy_local = cloth_positions[..., :2] - self.scene.env_origins[:, :2].unsqueeze(1)  # [E, N, 2]
        x_local = xy_local[..., 0]
        y_local = xy_local[..., 1]

        # boolean mask for nodes inside the box
        inside = (x_local >= x_min) & (x_local <= x_max) & (y_local >= y_min) & (y_local <= y_max)

        # reward: fraction of nodes inside (∈ [0,1]) times scale
        frac_inside = inside.float().mean(dim=1)  # [E]

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

        rewards["height_reward"] = (1.0 / (0.1 + mean_cloth_height)) * self._rewards["height_reward"].scale
        rewards["height_reward"][mean_cloth_height > 0.4] *= 0.0
        rewards["spread_reward"] = (pairwise_sum / count) * self._rewards["spread_reward"].scale
        rewards["corner_x_reward"] = (free_x.mean(-1)) * self._rewards["corner_x_reward"].scale
        rewards["direction_reward"] = (free_x.mean(-1) - grasped_x.mean(-1)) * self._rewards["direction_reward"].scale
        rewards["target_area_reward"] = frac_inside * self._rewards["target_area_reward"].scale

        # Simple action penalty buffer (kept structure)
        rewards["action_penalty"] = self.action_penalties.mean(0) * self._rewards["action_penalty"].scale

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

        # Only give reward at the last update step
        # if (self.episode_length_buf != self.cfg.num_updates_per_episode).any():
        #     total_reward[self.episode_length_buf != self.cfg.num_updates_per_episode] *= 0.0

        self.extras["log"] = {n: v.value.mean() for n, v in self._rewards.items()}
        return total_reward

    # -------------------------
    # Reset / Joints / Scale
    # -------------------------

    def _get_cloth_positions_shaped(self):
        cloth_positions = torch.zeros((
            self.num_envs, 
            self._cloth_objects[0].root_physx_view.max_particles_per_cloth, 3),
            device=self.device)
        for idx in range(self.num_envs):
            cloth_positions[idx] = self._cloth_objects[idx].root_physx_view.get_positions().reshape(-1,3)
        return cloth_positions

    def _get_cloth_positions_flat(self):
        cloth_positions = torch.zeros((
            self.num_envs, 
            self._cloth_objects[0].root_physx_view.max_particles_per_cloth * 3),
            device=self.device)
        for idx in range(self.num_envs):
            cloth_positions[idx] = self._cloth_objects[idx].root_physx_view.get_positions()
        return cloth_positions

    def _reset_idx(self, env_ids: torch.Tensor | None):
        super()._reset_idx(env_ids)

        self.action_steps[env_ids] = 0

        # Cache default cloth/handles state
        if not hasattr(self, "default_states") or self.default_states is None:
            self.default_states = {}
            for idx in range(self.num_envs):
                cloth_positions = self._get_cloth_positions_flat()
            self.default_states["_cloth_plain"] = cloth_positions.clone()

        # Reset cloth/handles
        zeros_vel = torch.zeros(
            (len(env_ids), self._cloth_objects[0].root_physx_view.max_particles_per_cloth * 3), device=self.device
        )

        zero_idx = torch.tensor([0], device=self.device)
        for idx in range(self.num_envs):
            self._cloth_objects[idx].root_physx_view.set_velocities(zeros_vel[idx], indices=zero_idx)
            self._cloth_objects[idx].root_physx_view.set_positions(self.default_states["_cloth_plain"][idx], indices=zero_idx)
            self._cloth_objects[idx].update(self.physics_dt)

        # Reset robot joints to default pose
        if hasattr(self, "_robot"):
            joint_pos = self._robot.data.default_joint_pos[env_ids].clone()
            joint_vel = self._robot.data.default_joint_vel[env_ids].clone()
            self._robot.set_joint_position_target(joint_pos, env_ids=env_ids)
            self._robot.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)

    # -------------------------
    # Observations
    # -------------------------

    def _get_observations(self) -> torch.Tensor:

        ##### Direct access ######
        if self.cfg.use_cloth_positions:
            cloth_pos = self._get_cloth_positions_flat()
            observations = cloth_pos[:,::10] # subsample

        elif not self.cfg.use_resnet:
            # Cloth corners (flattened)
            current_corner_obs = self._get_corners() - self.scene.env_origins.unsqueeze(1)
            current_corner_obs = current_corner_obs.reshape(self.num_envs, -1)

            observations = current_corner_obs

        ##### OR #####
        else:
            # Use depth image from EE camera
            if getattr(self, "_ee_camera", None) is not None:
                # advance and fetch depth
                self._ee_camera.update(self.physics_dt)
                # Try both keys depending on your build
                if "distance_to_image_plane" in self._ee_camera.data.output:
                    depth = self._ee_camera.data.output["distance_to_image_plane"]
                elif "depth" in self._ee_camera.data.output:
                    depth = self._ee_camera.data.output["depth"]
                else:
                    depth = self._ee_camera.data.output["rgb"]

                if depth.dim() == 4 and depth.size(-1) == 1:
                    depth = depth.squeeze(-1)

                # sanitize numeric issues
                # (clip to sensor range; convert NaN/Inf to max)
                # min_d = depth.min()
                # max_d = depth.max()
                min_d, max_d = 0.0, 1.0
                depth = torch.clamp(depth, min=min_d, max=max_d)

                # optional normalization to [0,1]
                depth_norm = (depth - min_d) / (max_d - min_d)
                # depth_norm = resize(depth_norm.permute(0,3,1,2), (256,256))

                img_transformed = self.resnet_preprocess(depth_norm.unsqueeze(1).repeat(1,3,1,1))
                img_transformed = self.resnet_model(img_transformed)

                observations = img_transformed

            # Optionally save image
            # plt.imshow(depth_norm[0].unsqueeze(1).repeat(1,3,1,1))
            # plt.savefig('/workspace/isaaclab/source/isaaclab_tasks/isaaclab_tasks/depth_env0_iteration_' + str(self.iteration_step) + '.jpg')

        return {"policy": observations}

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
        z = 0.0005  # offset to prevent coplanar overlap with PhysX ground
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