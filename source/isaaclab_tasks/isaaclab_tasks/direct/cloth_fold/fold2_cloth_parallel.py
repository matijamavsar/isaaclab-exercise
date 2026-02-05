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
from isaacsim.core.utils.prims import get_prim_at_path
from pxr import UsdGeom, Gf, Sdf, UsdGeom, UsdShade, UsdLux, UsdPhysics
from omni.physx.scripts import physicsUtils, particleUtils
import omni.usd
import omni.kit.commands
from isaaclab_assets import FRANKA_PANDA_HIGH_PD_CFG
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
import carb
import numpy as np

from torchvision.utils import save_image  # kept because original code used it in logs
from scipy.spatial import ConvexHull      # kept; convex hull code gated behind ifs
from torchvision.models import resnet18, ResNet18_Weights, resnet50, ResNet50_Weights

# Local utilities
from .dmp_integrator import BatchDMPIntegrator
from .min_jerk_traj import generate_minimum_jerk

from .implementation.Cloth import Cloth
from .implementation.utils import createRectangularMesh

from concurrent.futures import ProcessPoolExecutor

"""
TO USE INSIDE DOCKER FIRST DO apt-get update && apt-get install libsuitesparse-dev
"""

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
    episode_length_s = 3
    max_episode_length = episode_length_s * 120
    num_updates_per_episode = 1
    decimation = max_episode_length // num_updates_per_episode
    action_space = 55
    observation_space = 100
    state_space = 0
    enable_camera_recording = True
    plot_trajectories = False
    nice_render = True
    num_tries = 1
    use_resnet = False
    use_cloth_positions = True
    seed = 42

    torch.manual_seed(seed)

    # simulation
    sim: SimulationCfg = SimulationCfg(
        dt=1 / 120,
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
        "fold_reward": Reward(True, 1.0),
        "action_penalty": Reward(False, 1e-2),
        "target_area_reward": Reward(False, 1.0),
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

n = 20; na = n; nb = n
X, T = createRectangularMesh(a = 0.7, b = 0.7, na = na, nb = nb, h = 0.01)
# X[:,2] += 0.1; 
tf = 360; 
def _run_one_cloth(args):

    j, X, T, na, tf, u, env_origins = args

    XX = X.copy()
    
    XX[:,0] += env_origins[0].item()
    XX[:,1] += env_origins[1].item()
    cloth = Cloth(XX, T)
    cloth.setSimulatorParameters(dt=1/120, tol=0.005,
                                rho=0.1, delta=0.1, kappa=0.00005,
                                shr=12, str=0.03, alpha=0.3,
                                mu_f=0.5, mu_s=0.5)
    #simulation loop
    inds = [0, 19]
    for k in range(tf):
        cloth.simulate(u=np.array([u[k,0:3], u[k,3:6]]), control=inds)

    print('simulation finished for cloth', j)
    return j, cloth.history_pos, cloth.history_vel

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
        self.num_steps_to_settle = 240

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
        self.dmp_integrator = BatchDMPIntegrator(N_basis=7, dof=6, device=self.device)
        self.prev_y = torch.zeros((self.num_envs, 3), device=self.device)

        # Reward helpers
        self._rewards = self.cfg.rewards
        self.z_abs_pos = torch.zeros((self.cfg.max_episode_length, self.num_envs), device=self.device)
        self.action_penalty_buf = torch.zeros((self.cfg.max_episode_length, self.num_envs), device=self.device)
        self.cloth_pos_buffer = torch.zeros((self.cfg.max_episode_length, self.num_envs, 4, 3), device=self.device)
        self.y_buffer = torch.zeros((self.cfg.max_episode_length, self.num_envs, 6), device=self.device)

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

        # Caida libre
        # X[:,2] += 0.1; 

        self.simcloth_objects = []
        for i in range(self.num_envs):
            XX = X.copy()
            XX[:,0] += self.scene.env_origins[i,0].item()
            XX[:,1] += self.scene.env_origins[i,1].item()
            self.simcloth_objects.append(Cloth(XX, T)); 
            self.simcloth_objects[-1].setSimulatorParameters(
                dt=1/120,tol=0.005, rho=0.1,delta=0.1,kappa=0.000025,shr=20,
                str=0.05,alpha=0.3,mu_f=0.5,mu_s=0.5)

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
        plane_resolution = 19
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
        # omni.kit.commands.execute(
        #     "BindMaterialCommand",
        #     prim_path=cloth_mesh_path,  # Target cloth mesh
        #     material_path=material_path,  # Same material for all
        #     strength=None
        # )

        # ✅ Reuse existing particle system & material
        particle_system_path = Sdf.Path("/World/particleSystem")

        # ✅ Apply existing particle system to cloth (instead of creating a new one)

        # # Randomize parameters
        stretchStiffness = float(torch.rand(1).item() * (1_000_000 - 50_000) + 50_000)
        bendStiffness    = float(torch.rand(1).item() * (100 - 10) + 10)
        shearStiffness   = float(torch.rand(1).item() * (500 - 50) + 50)
        damping          = float(torch.rand(1).item() * (0.9 - 0.25) + 0.25)

        # stretchStiffness = 1000000.0
        # bendStiffness = 20.0
        # shearStiffness = 100.0
        # damping = 0.8

        self._particle_api = particleUtils.add_physx_particle_cloth(
            stage=stage,
            path=cloth_mesh_path,
            dynamic_mesh_path=None,
            particle_system_path=particle_system_path,  # Use the same system
            spring_stretch_stiffness=stretchStiffness,
            spring_bend_stiffness=bendStiffness,
            spring_shear_stiffness=shearStiffness,
            spring_damping=damping,
            self_collision=False,
            self_collision_filter=False,
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

        cloth_prim.GetAttribute('physxParticle:particleEnabled').Set(False)
        return cloth_mesh_path, plane_width

    def _setup_scene(self):
        # Terrain
        self.cfg.terrain.num_envs = self.scene.cfg.num_envs
        self.cfg.terrain.env_spacing = self.scene.cfg.env_spacing
        self._terrain = self.cfg.terrain.class_type(self.cfg.terrain)

        # Clone envs
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

            # # Uncomment this for scale randomization
            # random_scale = torch.tensor([torch.rand(1).item() * (1.0 - 0.5) + 0.5, 
            #                              torch.rand(1).item() * (1.0 - 0.5) + 0.5, 
            #                              1.0])
            # cloth_prim.GetAttribute("xformOp:scale").Set(Gf.Vec3f(*random_scale.tolist()))

            cloth_prim.GetAttribute("xformOp:translate").Set(
                Gf.Vec3f(0.0, 0.0, 1.5))
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

    def _get_corners(self):
        cloth_positions = self._get_cloth_positions_shaped()
        side_len = int(torch.sqrt(torch.tensor(cloth_positions.shape[1])).to(self.device))
        corners = cloth_positions[:, [-side_len, -1, 0, side_len - 1], :]
        return corners

    # -------------------------
    # RL hooks
    # -------------------------

    def _pre_physics_step(self, actions: torch.Tensor):
        clamp_loc = 0.6
        print('Setting DMP parameters')
        dmp_parameters = actions.clone()

        ### Set cloth node goal
        dmp_parameters[:, 6] = dmp_parameters[:, 6].clamp(-clamp_loc, clamp_loc)
        dmp_parameters[:, 7] = dmp_parameters[:, 7].clamp(-clamp_loc, clamp_loc)
        dmp_parameters[:, 8] = dmp_parameters[:, 8].clamp(0.01, 0.2)
        dmp_parameters[:, 9] = dmp_parameters[:, 9].clamp(-clamp_loc, clamp_loc)
        dmp_parameters[:, 10] = dmp_parameters[:, 10].clamp(-clamp_loc, clamp_loc)
        dmp_parameters[:, 11] = dmp_parameters[:, 11].clamp(0.01, 0.2)
        dmp_parameters[:,12:-1] = dmp_parameters[:,12:-1]# * clamp_loc / 1.6

        self.action_penalties[self.action_steps] = dmp_parameters[:,6:-1].abs().mean()

        # dmp_parameters = dmp_parameters.clamp(-1.0, 1.0)

        cloth_positions = self._get_cloth_positions_shaped()
        side_len = int(torch.sqrt(torch.tensor(cloth_positions.shape[1])).to(self.device))
        for i in range(self.num_envs):
            dmp_parameters[i, 0:3] = torch.tensor(self.simcloth_objects[i].positions[0], device=self.device)
            dmp_parameters[i, 3:6] = torch.tensor(self.simcloth_objects[i].positions[side_len-1], device=self.device)
        dmp_parameters[:,0:3] -= self.scene.env_origins
        dmp_parameters[:,3:6] -= self.scene.env_origins

        self.actions = dmp_parameters
        dmp_tau = torch.zeros(self.num_envs, device=self.device) + self.cfg.episode_length_s
        self.actions[:, -1] = dmp_tau

        reset_dmp_indices = torch.arange(0, dmp_parameters.shape[0])

        if len(reset_dmp_indices) > 0:
            self.dmp_integrator.reset_indices(
                reset_dmp_indices,
                dmp_parameters,
                dmp_tau,
                dt=self.physics_dt,
                variant=1,
            )
            self.dmp_initialized = True

        for i in range(self.cfg.max_episode_length):
            t, y, dy, ddy = self.dmp_integrator.step()
            self.dmp_integrator.x[t >= self.actions[:, -1]] = 0.1353
            yy = y.clone()
            yy[:,0:3] += self.scene.env_origins
            yy[:,3:6] += self.scene.env_origins
            self.y_buffer[i] = yy

        self.pos_history = [None] * self.num_envs
        self.vel_history = [None] * self.num_envs
        args_list = [(j, X, T, na, tf, self.y_buffer[:,j].cpu().numpy(), self.scene.env_origins[j].cpu().numpy()) for j in range(self.num_envs)]
        with ProcessPoolExecutor(max_workers=12) as ex:
            for m, pos, vel in ex.map(_run_one_cloth, args_list):
                self.pos_history[m] = pos
                self.vel_history[m] = vel

    def _apply_action(self):
        clamp_loc = 0.5

        if self.action_steps[0] == self.cfg.max_episode_length - 1:
            for i in range(self.num_envs):
                self._cloth_objects[i].root_physx_view.set_positions(
                    torch.tensor(np.stack(self.pos_history[i][1+self.action_steps[0]]), device=self.device), 
                    torch.tensor([0], device=self.device))
                self._cloth_objects[i].root_physx_view.set_velocities(
                    torch.tensor(np.stack(self.vel_history[i][1+self.action_steps[0]]), device=self.device), 
                    torch.tensor([0], device=self.device))
        else:
            pass

        # Camera recording window
        if self.cfg.enable_camera_recording:
            in_window = (self.iteration_step % 20000) < 1440
            if in_window and self.video_writer is None:
                self.render_count = 0
                self._setup_camera_writer(self.iteration_step)
            if in_window and self.video_writer is not None:
                if self.render_count % self.cfg.sim.render_interval == 0:
                    self._capture_and_write_frame()
                self.render_count += 1
            if (not in_window) and self.video_writer is not None:
                self.render_count = 0
                self._close_camera_writer()

        self.iteration_step += 1
        self.action_steps += 1

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        terminated = 0
        truncated = self.episode_length_buf >= self.max_episode_length
        if truncated.any():
            truncated[:] = True
        return terminated, truncated

    def compute_rewards(self):
        corners = self._get_corners()
        cloth_positions = self._get_cloth_positions_shaped()

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

        dist_02 = torch.norm(corners[:, 0] - corners[:, 2], dim=-1)
        dist_13 = torch.norm(corners[:, 1] - corners[:, 3], dim=-1)
        # folded = 1.0 / (1e-3 + dist_02) + 1.0 / (1e-3 + dist_13)
        folded = -dist_02 - dist_13

        rewards["fold_reward"] = folded * self._rewards["fold_reward"].scale
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
        # if not hasattr(self, "default_states") or self.default_states is None:
        #     self.default_states = {"_cloth_plain_pos": [], "_cloth_plain_vel": []}
        #     for i in range(self.num_envs):
        #         self.default_states["_cloth_plain_pos"].append(self.simcloth_objects[i].positions.copy())
        #         self.default_states["_cloth_plain_vel"].append(self.simcloth_objects[i].velocities.copy())

        # for i in range(self.num_envs):
        #     self.simcloth_objects[i].positions = self.default_states["_cloth_plain_pos"][i]
        #     self.simcloth_objects[i].velocities = self.default_states["_cloth_plain_vel"][i]
        #     # self.simcloth_objects[i].history_pos = []
        #     # self.simcloth_objects[i].history_vel = []
        #     self.simcloth_objects[i].restart()

        # plt.clf(); plt.cla()
        # plt.plot(self.cloth_pos_buffer[:,0,:,2].cpu()) # plot Z of corners for first env
        # plt.savefig('./logs/tmp_clothilde_parallel.pdf')

    # -------------------------
    # Observations
    # -------------------------

    def _get_observations(self) -> torch.Tensor:

        ##### Direct access ######
        if self.cfg.use_cloth_positions:
            cloth_pos = self._get_cloth_positions_flat()
            observations = cloth_pos[:,::25] # subsample

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

        observations = torch.zeros((self.num_envs, 1), device=self.device)

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
        z = -0.05  # offset to prevent coplanar overlap with PhysX ground
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