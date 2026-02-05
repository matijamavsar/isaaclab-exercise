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
from isaaclab.assets import ParticleClothObjectCfg, ParticleClothObject
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
from pxr import UsdPhysics, UsdGeom, Gf, Sdf, UsdLux
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
import carb

from .dmp_integrator import BatchDMPIntegrator
from .min_jerk_traj import generate_minimum_jerk

from torchvision.models import resnet18, ResNet18_Weights, resnet50, ResNet50_Weights

""" Run this training using
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py --task DMP-Based-Particle-Randomized-Init-Motion --num_envs 64 --max_iterations 16000 --headless --enable_cameras
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
    nice_render = True
    use_resnet = True
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
        render=RenderCfg(rendering_mode='performance')
    )

    # scene
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=1, env_spacing=2.0, replicate_physics=False)

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
        ee_camera = CameraCfg(
            # mount under the Franka hand so it inherits the link pose
            prim_path="/World/envs/env_.*/ee_camera",
            offset=CameraCfg.OffsetCfg(
                pos=(-1.0, 0.0, 0.35),
                rot=(1,0,0,0),
                convention="world",
            ),
            data_types=["depth"],
            spawn=sim_utils.PinholeCameraCfg(
                focal_length=16.0, focus_distance=100.0,
                horizontal_aperture=12.0, clipping_range=(0.05, 20.0)
            ),
            width=256,    # smaller to keep obs light-weight
            height=256,
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

    rewards = {
        "spread_reward": 
            Reward(True, 1.0),
        "height_reward": 
            Reward(True, 1.0),
        "corner_x_reward": 
            Reward(True, 10.0),
        "direction_reward": 
            Reward(True, 15.0),
        "action_penalty": 
            Reward(True, 1e-1),
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
        self.iteration_step = torch.tensor(0, device=self.device)
        self.default_states = None
        self.joints_created = False
        self.dmp_initialized = False
        self.reset_count = 0

        self._rewards = self.cfg.rewards

        self.sigmoid = torch.nn.Sigmoid()

        # Initialize robot dictionaries
        self.prev_rot_around_x = torch.zeros((self.num_envs), device=self.device)
        self.z_abs_pos = torch.zeros((self.cfg.max_episode_length, self.num_envs), device=self.device)
        self.action_penalties = torch.zeros((self.cfg.max_episode_length, self.num_envs), device=self.device)

        H  = self.cfg.decimation
        self.prev_corners_buf = torch.zeros((H, self.num_envs, 6), device=self.device)
        self.prev_abs_buf = torch.zeros((H, self.num_envs, 2), device=self.device)
        self.prev_depth_buf = torch.zeros((H, self.num_envs, 20), device=self.device)

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

        self.dmp_integrator = BatchDMPIntegrator(N_basis=25, dof=2, device=self.device)
        self.corner_dmp_integrator = BatchDMPIntegrator(N_basis=25, dof=12, device=self.device)
        self.corner_traj = torch.zeros((self.num_envs, self.cfg.decimation, 6), device=self.device)
        self.corner_dmp_weights = torch.zeros((self.num_envs, 12, 25), device=self.device)
        self.corner_y0 = torch.zeros((self.num_envs, 12), device=self.device)
        self.corner_goal = torch.zeros((self.num_envs, 12), device=self.device)

        if self.cfg.use_resnet:
            weights = ResNet18_Weights.DEFAULT
            self.resnet_preprocess = weights.transforms()
            self.resnet_model = resnet18(weights=weights).to(self.device)
            self.resnet_model.eval()

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
        abs_pose = self._get_absolute_pose().clone()  # (num_envs, 7)
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
        abs_pose = self._get_absolute_pose().clone()  # (num_envs, 7)
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

    def create_cloth(self, stage, env_idx):
        cloth_mesh_path = Sdf.Path(f"/World/envs/env_{env_idx}/Cloth")
        particle_material_path = Sdf.Path("/World/particleMaterial")

        # ✅ Create a mesh that will be turned into cloth
        # plane_resolution = 40
        plane_resolution = 20
        # plane_width = float(torch.rand(1).item() * (70 - 45) + 45)
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
        # stretchStiffness = float(torch.rand(1).item() * (1_000_000 - 50_000) + 50_000)
        # bendStiffness    = float(torch.rand(1).item() * (100 - 10) + 10)
        # shearStiffness   = float(torch.rand(1).item() * (500 - 50) + 50)
        # damping          = float(torch.rand(1).item() * (0.9 - 0.25) + 0.25)

        stretchStiffness = 1000000.0
        bendStiffness = 20.0
        shearStiffness = 100.0
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
        particle_mass = 0.002
        num_verts = len(cloth_mesh.GetPointsAttr().Get())
        mass = particle_mass * num_verts
        massApi = UsdPhysics.MassAPI.Apply(cloth_mesh.GetPrim())
        massApi.GetMassAttr().Set(mass)

        # Set color and access to positions
        cloth_prim = self.scene.stage.GetPrimAtPath(cloth_mesh_path)
        cloth_color_attr = cloth_prim.GetAttribute("primvars:displayColor")
        cloth_color_attr.Set([Gf.Vec3f(0.2, 0.0, 0.220)])

        return cloth_mesh_path, plane_width

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
        self.cloth_lengths = torch.zeros(self.scene.cfg.num_envs, device=self.device)

        # ✅ Configure terrain
        self.cfg.terrain.num_envs = self.scene.cfg.num_envs
        self.cfg.terrain.env_spacing = self.scene.cfg.env_spacing
        self._terrain = self.cfg.terrain.class_type(self.cfg.terrain)

        # ✅ Clone environments
        self.scene.clone_environments(copy_from_source=True)

        # ----------------------------
        # ✅ Create Particle System ONCE
        # ----------------------------
        stage = self.scene.stage
        particle_system_path = Sdf.Path("/World/particleSystem")

        restOffset = 0.01
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
            cloth_prim_path, plane_width = self.create_cloth(self.scene.stage, env_idx)  # Create cloth for each env
            cloth_cfg = ParticleClothObjectCfg(
                prim_path=f"/World/envs/env_{env_idx}", 
                spawn=None
                )
            self._cloth_objects.append(ParticleClothObject(cloth_cfg))

            # Randomize cloth length (scale Y) per env
            cloth_prim_path = f"/World/envs/env_{env_idx}/Cloth"
            cloth_prim = get_prim_at_path(cloth_prim_path)

            random_scale = torch.tensor([1.0, torch.rand(1).item() * (1.0 - 0.6) + 0.6, 1.0])
            cloth_prim.GetAttribute("xformOp:scale").Set(Gf.Vec3f(*random_scale.tolist()))
            cloth_prim.GetAttribute("xformOp:translate").Set(
                Gf.Vec3f(0.0, 0.0, 0.356 + ((1 - random_scale[1]) * 0.356).item())
            )
            cloth_prim.GetAttribute("xformOp:orient").Set(
                Gf.Quatd(0.5, 0.5, 0.5, 0.5))
            self.cloth_lengths[env_idx] = plane_width / 100 # because it's in cm

        if self.cfg.enable_camera_recording:
            self._camera = self.cfg.camera.class_type(self.cfg.camera)
            self._ee_camera = self.cfg.ee_camera.class_type(self.cfg.ee_camera)

        # ✅ Add lights (optional)
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

        # ✅ Get the stage
        stage = omni.usd.get_context().get_stage()

        self.set_cloth_texture("/workspace/isaaclab/source/isaaclab_tasks/isaaclab_tasks/sim/cloth_texture/cloth_texture_1.jpeg",
            roughness=0.6)

        if not self.cfg.nice_render:
            light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
            light_cfg.func("/World/Light", light_cfg)

            # Hide world environment (optional)
            stage = omni.usd.get_context().get_stage()
            env_prim = stage.GetPrimAtPath("/World/ground/Environment")
            if env_prim:
                UsdGeom.Imageable(env_prim).GetVisibilityAttr().Set(UsdGeom.Tokens.invisible)
        ##################################################################################

        ##################################################################################
        #### and enable this ####
        else:
            self.setup_clean_stage(renderer="HydraStorm")  # simplest, no noise/ghosting
        ##################################################################################

        # # ----------------------------
        # # ✅ CREATE A VISIBLE GROUND PLANE WITH MATERIAL
        # # ----------------------------
        # visual_plane_path = "/World/ground/GroundPlane/VisualPlane"
        # material_path = "/World/Materials/GroundMaterial"

        # # ✅ Create the ground plane if it doesn't exist
        # if not stage.GetPrimAtPath(visual_plane_path):
        #     visual_plane = UsdGeom.Mesh.Define(stage, visual_plane_path)

        #     # ✅ Set plane size and position
        #     visual_plane.GetPointsAttr().Set([
        #         Gf.Vec3f(-50, -50, 0), Gf.Vec3f(50, -50, 0),
        #         Gf.Vec3f(50, 50, 0), Gf.Vec3f(-50, 50, 0)
        #     ])
        #     visual_plane.GetFaceVertexIndicesAttr().Set([0, 1, 2, 2, 3, 0])
        #     visual_plane.GetFaceVertexCountsAttr().Set([3, 3])

        #     # ✅ Move it to the same height as the collision plane
        #     visual_plane.AddTranslateOp().Set(Gf.Vec3f(0, 0, 0))
        #     visual_plane.GetDisplayColorAttr().Set([Gf.Vec3f(0.00, 0.50, 0.00)])  # Green

        # # ✅ Create a new material for the ground
        # material_path = "/World/Materials/GroundMaterial"
        # material_prim = stage.DefinePrim(material_path, "Material")
        # material = UsdShade.Material(material_prim)

        # # ✅ Create a shader
        # shader_path = material_path + "/Shader"
        # shader = UsdShade.Shader.Define(stage, shader_path)
        # shader.CreateIdAttr("UsdPreviewSurface")  # Use USD's default shader

        # # ✅ Set ground color to dark gray
        # shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(0.1, 0.1, 0.1))  # Dark gray
        # shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.0)  # Rough texture
        # shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(0.0)  # Non-metallic

        # # ✅ Bind the material to the ground plane
        # UsdShade.MaterialBindingAPI(visual_plane).Bind(material)

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
        #     _ = self._create_cloth(self.scene.stage, env_idx)  # Create cloth for each env

    def _get_absolute_pose(self):
        absolute_pose = torch.zeros((self.num_envs, 7), device=self.device)
        absolute_pose[:,3] = 1 # Make quaternion (1,0,0,0)
        pose_1_w = self._get_corners()[:,0]
        pose_2_w = self._get_corners()[:,1]
        absolute_pose[:,0:3] = (pose_1_w + pose_2_w) / 2
        return absolute_pose

    # pre-physics step calls
    def _pre_physics_step(self, actions: torch.Tensor):

        print("Pre-physics step")
        dmp_parameters = actions.clone()
        dmp_parameters[:,2] = dmp_parameters[:,2].clamp(-0.5, 0.5)
        dmp_parameters[:,3] = dmp_parameters[:,3].clamp(0.15, 0.75)
        # dmp_parameters = dmp_parameters.clamp(-1.5, 1.5)

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

        y[:,0] = y[:,0].clamp(-0.5, 0.5)
        y[:,1] = y[:,1].clamp(0.15, 0.75)

        current_absolute_pose = self._get_absolute_pose()
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
                ### Good for top view of 4 envs
                # self._camera.set_world_poses_from_view(torch.tensor([0., 0., 6.5], device=self.device).unsqueeze(0),
                #                                        torch.tensor([0., 0., 0.], device=self.device))
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

        if not self.probing_done:
            y = torch.zeros((self.num_envs, 2), device=self.device)
            y[:,0] = self._probing_traj[self.action_steps][:,0]
            y[:,1] = self._probing_traj[self.action_steps][:,2]

            self.set_cloth_nodes(y)

            # get free current corners, flatten them
            corners = self._get_corners()[:, [2, 3]] - self.scene.env_origins.unsqueeze(1)  # (N,2,3)
            corners_flat = corners.reshape(self.num_envs, -1)
            self.prev_corners_buf = self.prev_corners_buf.roll(-1, dims=0)
            self.prev_corners_buf[-1] = corners_flat

            self.prev_abs_buf = self.prev_abs_buf.roll(-1, dims=0)
            self.prev_abs_buf[-1] = current_absolute_pose[:, [0, 2]] - self.scene.env_origins[:, [0, 2]]

            if self.cfg.use_resnet:
                self.prev_depth_buf = self.prev_depth_buf.roll(-1, dims=0)
                self.prev_depth_buf[-1] = F.avg_pool1d(self.get_depth_image(), 50)

            if (self.action_steps == self.cfg.max_episode_length - 1).all():
                self.probing_done = True

        elif (self.action_steps < (self.cfg.max_episode_length - 60)).all():
            self.set_cloth_nodes(y)
            self.action_penalties[self.action_steps] = dy.abs().mean(1)
        else:
            # Keep cloth in the same place right before the end of episode
            target_y = current_absolute_pose[:,0:3] - self.scene.env_origins
            target_y = target_y[:,[0,2]]
            self.set_cloth_nodes(y)

        if self.cfg.plot_trajectories:
            self.log_trajectory_panel()
            self.plot_final_poses()

        self.iteration_step += 1
        self.action_steps += 1

    def get_depth_image(self):
        self._ee_camera.update(self.physics_dt)
        # Try both keys depending on your build
        if "distance_to_image_plane" in self._ee_camera.data.output:
            depth = self._ee_camera.data.output["distance_to_image_plane"]
        else:
            depth = self._ee_camera.data.output["depth"]

        # ensure shape [B,H,W]
        if depth.dim() == 4 and depth.size(-1) == 1:
            depth = depth.squeeze(-1)

        # sanitize numeric issues
        # (clip to sensor range; convert NaN/Inf to max)
        # min_d = depth.min()
        # max_d = depth.max()
        min_d, max_d = 0.05, 2.0
        depth = torch.clamp(depth, min=min_d, max=max_d)

        # optional normalization to [0,1]
        depth_norm = (depth - min_d) / (max_d - min_d)

        img_transformed = self.resnet_preprocess(depth_norm.unsqueeze(1).repeat(1,3,1,1))
        img_transformed = self.resnet_model(img_transformed)

        return img_transformed

    def set_cloth_nodes(self, y):
        current_absolute_pose = self._get_absolute_pose()
        target_absolute_pose = current_absolute_pose.clone() * 0.0
        target_absolute_pose[:,0] = y[:,0]
        target_absolute_pose[:,2] = y[:,1]
        target_absolute_pose[:,0:3] += self.scene.env_origins
        
        target_absolute_pose[:, 3:] = torch.tensor([1, 0, 0, 0], device=self.device)
        
        # Fixed Y-distance between grippers
        fixed_rel_pos = torch.tensor([0.0, 0.7, 0.0], device=self.device).repeat(self.num_envs, 1)
        fixed_rel_quat = torch.tensor([1, 0, 0, 0], device=self.device).repeat(self.num_envs, 1)
        relative_pose = torch.cat((fixed_rel_pos, fixed_rel_quat), dim=-1)

        # Transform from global into each robot coordinate system
        larm_pose, rarm_pose = self._abs_to_arm_poses(target_absolute_pose, relative_pose)
        cloth_positions = self._get_cloth_positions_shaped()
        side_len = int(torch.sqrt(torch.tensor(cloth_positions.shape[1])).to(self.device))

        current_cloth_pos = self._get_cloth_positions_shaped()
        for i in range(self.num_envs):
            current_cloth_pos[i, -side_len] = rarm_pose[i,0:3]
            current_cloth_pos[i, -1] = larm_pose[i,0:3]
            self._cloth_objects[i].root_physx_view.set_positions(
                current_cloth_pos[i], 
                torch.tensor([0], device=self.device))

    # post-physics step calls
    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        # Calculate distance between grippers
        terminated = 0
        truncated = self.episode_length_buf >= self.max_episode_length

        if truncated.any():
            truncated[:] = True

        return terminated, truncated

    def _get_corners(self):
        cloth_positions = self._get_cloth_positions_shaped()
        side_len = int(torch.sqrt(torch.tensor(cloth_positions.shape[1])).to(self.device))
        corners = cloth_positions[:, [-side_len, -1, 0, side_len - 1], :]
        return corners

    def compute_rewards(self):
        """
        Compute individual reward term values and return a dict of reward_name -> value tensor.
        """
        # Gather data
        corners = self._get_corners()
        cloth_positions = self._get_cloth_positions_shaped()
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
            rewards["height_reward"] = (1.0 / (0.1 + mean_cloth_height)) * self._rewards["height_reward"].scale
            rewards["height_reward"][mean_cloth_height > 0.4] *= 0.0
            rewards["spread_reward"] = (pairwise_sum / count) * self._rewards["spread_reward"].scale
            rewards["corner_x_reward"] = (free_x.mean(-1)) * self._rewards["corner_x_reward"].scale
            rewards["direction_reward"] = (free_x.mean(-1) - grasped_x.mean(-1)) * self._rewards["direction_reward"].scale
        elif self.cfg.use_weighted_atan_rewards:
            rewards["height_reward"] = 0.5 + (1.0 / torch.pi) * torch.atan(1.0 / (0.1 + mean_cloth_height))
            rewards["spread_reward"] = 0.5 + (1.0 / torch.pi) * torch.atan(pairwise_sum / count)
            rewards["corner_x_reward"] = 0.5 + (1.0 / torch.pi) * torch.atan(free_x.mean(-1))
            rewards["direction_reward"] = 0.5 + (1.0 / torch.pi) * torch.atan(free_x.mean(-1) - grasped_x.mean(-1))
        elif self.cfg.use_weighted_exp_rewards:
            rewards["height_reward"] = torch.exp(1.0 / (0.1 + mean_cloth_height))
            rewards["spread_reward"] = torch.exp(pairwise_sum / count)
            rewards["corner_x_reward"] = torch.exp(free_x.mean(-1))
            rewards["direction_reward"] = torch.exp(free_x.mean(-1) - grasped_x.mean(-1))

        rewards["action_penalty"] = self.action_penalties.mean(0) * self._rewards["action_penalty"].scale
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

        invalid_env = (self.z_abs_pos < 0.15).any(0)
        total_reward[invalid_env] -= 10

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
            print(f"{reward_name} mean: \t {reward_value.mean().item():.4f}")
            print(f"{reward_name} std : \t {reward_value.std().item():.4f}")
        
        if total_reward is not None:
            print(f"\nTotal Reward: \t {total_reward.mean().item():.4f}")
        
        print("-------------------------")

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

    def change_cloth_scale(self,
                           scales: torch.Tensor,
                           env_ids: torch.Tensor | None = None):

        for idx in env_ids.tolist():
            prim_cloth_path = f"/World/envs/env_{idx}/Cloth"
            prim_cloth = get_prim_at_path(prim_cloth_path)
            sx, sy, sz = scales[idx].tolist()
            prim_cloth.GetAttribute("xformOp:scale").Set(Gf.Vec3f(sx, sy, sz))
            prim_cloth.GetAttribute('xformOp:translate').Set(
                Gf.Vec3f(0.0, 0.0, 0.356 + ((1 - scales[idx][1]) * 0.356).item()))

    def create_joints(self, env_ids):
        for idx in env_ids:
            cube_1_path = Sdf.Path("/World/envs/env_" + str(int(idx)) + "/Cloth/RightCube")
            cube_2_path = Sdf.Path("/World/envs/env_" + str(int(idx)) + "/Cloth/LeftCube")
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

        if not self.cfg.use_resnet:
            act_hist = self.prev_abs_buf.permute(1,0,2)
            act_hist = act_hist[:,::48].reshape(self.num_envs, -1)

            self.corner_traj = self.prev_corners_buf.permute(1,0,2)
            corner_traj_obs = self.corner_traj[:,::48].reshape(self.num_envs, -1)

            if self.cfg.disable_init_motion:
                corner_traj_obs *= 0.0
                act_hist *= 0.0
        
            observations = torch.cat((act_hist, corner_traj_obs), dim=-1)

        ##### OR #####
        else:
            act_hist = self.prev_abs_buf.permute(1,0,2)
            act_hist = act_hist[:,::48].reshape(self.num_envs, -1)

            img_hist = self.prev_depth_buf.permute(1,0,2)
            img_hist = img_hist[:,::48].reshape(self.num_envs, -1)

            if self.cfg.disable_init_motion:
                corner_traj_obs *= 0.0
                img_hist *= 0.0

            observations = torch.cat((act_hist, img_hist), dim=-1)

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