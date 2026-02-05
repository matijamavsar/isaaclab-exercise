# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
from __future__ import annotations

import os
import time
from datetime import datetime

import cv2
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from isaaclab.assets import Articulation, ArticulationCfg
from isaaclab.assets import ParticleClothObjectCfg, RigidObjectCfg
from isaaclab.controllers import DifferentialIKController, DifferentialIKControllerCfg
import isaaclab.sim as sim_utils
from isaaclab.envs import DirectRLEnv, DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import PhysxCfg, RenderCfg, SimulationCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaacsim.core.utils.prims import get_prim_at_path
from pxr import UsdGeom, Gf, Sdf, UsdGeom, UsdShade, UsdLux
import omni.usd
import omni.kit.commands
from isaaclab_assets import FRANKA_PANDA_HIGH_PD_CFG
import carb

from torchvision.utils import save_image  # kept because original code used it in logs
from scipy.spatial import ConvexHull      # kept; convex hull code gated behind ifs

# Local utilities
from .dmp_integrator import BatchDMPIntegrator
from .min_jerk_traj import generate_minimum_jerk


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
    episode_length_s = 5
    max_episode_length = episode_length_s * 120
    num_updates_per_episode = 1
    decimation = max_episode_length // num_updates_per_episode
    action_space = 28
    observation_space = 100
    state_space = 0
    enable_camera_recording = True
    plot_trajectories = False
    disable_init_motion = False
    use_weighted_atan_rewards = False
    use_weighted_exp_rewards = False
    nice_render = False
    num_tries = 3
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

    # cloth + handles
    cloth_plain = ParticleClothObjectCfg(
        prim_path="/World/envs/env_.*/Cloth",
        init_state=ParticleClothObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 0.356), rot=(0.55, 0.5, 0.5, 0.5)),
        spawn=sim_utils.UsdFileCfg(
            # usd_path="/workspace/isaaclab/source/isaaclab_tasks/isaaclab_tasks/sim/particle_cloth_new_noHandles.usd",
            usd_path="/workspace/isaaclab/source/isaaclab_tasks/isaaclab_tasks/sim/particle_cloth_two_pinch_half_width_box_bendy_noHandles.usd",
            scale=(1.0, 1.0, 1.0),
        ),
    )

    if enable_camera_recording:
        from isaaclab.sensors import CameraCfg
        camera = CameraCfg(
            prim_path="/World/Camera",
            offset=CameraCfg.OffsetCfg(
                pos=(-5.0, -5.0, 3.0),
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
        ee_camera = CameraCfg(
            # mount under the Franka hand so it inherits the link pose
            prim_path="/World/envs/env_.*/ee_camera",
            offset=CameraCfg.OffsetCfg(
                # small forward offset in the hand frame; tweak as needed
                pos=(0.0, 0.0, 1.5),
                # looking along the hand's -Z/+X depends on your asset; start with identity
                rot=(0.7071,0,0.7071,0),
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

    action_scale = 15
    dof_velocity_scale = 0.1
    filter_kernel_size = 7
    use_dynamic_rewards = False

    rewards = {
        "spread_reward": Reward(True, 3.0),
        "height_reward": Reward(True, 1e-1),
        "corner_x_reward": Reward(False, 10.0),
        "direction_reward": Reward(False, 15.0),
        "action_penalty": Reward(True, 1e-2),
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
        self.dmp_integrator = BatchDMPIntegrator(N_basis=7, dof=3, device=self.device)

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

    def _setup_scene(self):
        # Terrain
        self.cfg.terrain.num_envs = self.scene.cfg.num_envs
        self.cfg.terrain.env_spacing = self.scene.cfg.env_spacing
        self._terrain = self.cfg.terrain.class_type(self.cfg.terrain)

        # Cloth + handles
        self._cloth_plain = self.cfg.cloth_plain.class_type(self.cfg.cloth_plain)

        # Clone envs
        self.scene.clone_environments(copy_from_source=True)

        # Randomize cloth length (scale Y) per env
        self.cloth_lengths = torch.zeros(self.scene.cfg.num_envs, device=self.device)
        for env_idx in range(self.scene.cfg.num_envs):
            cloth_prim_path = f"/World/envs/env_{env_idx}/Cloth"
            cloth_prim = get_prim_at_path(cloth_prim_path)
            # random_scale = torch.tensor([1.0, torch.rand(1).item() * (1.0 - 0.6) + 0.6, 1.0])
            random_scale = torch.tensor([1.0, torch.rand(1).item() * (1.0 - 1.0) + 1.0, 1.0])
            cloth_prim.GetAttribute("xformOp:scale").Set(Gf.Vec3f(*random_scale.tolist()))
            cloth_prim.GetAttribute("xformOp:translate").Set(
                Gf.Vec3f(0.0, 0.0, 0.356 + ((1 - random_scale[1]) * 0.356).item())
            )
            self.cloth_lengths[env_idx] = random_scale[1].item() * 0.7

        # Camera
        if self.cfg.enable_camera_recording:
            self._camera = self.cfg.camera.class_type(self.cfg.camera)
            self._ee_camera = self.cfg.ee_camera.class_type(self.cfg.ee_camera)
        
        self.set_cloth_texture("/workspace/isaaclab/source/isaaclab_tasks/isaaclab_tasks/sim/cloth_texture/cloth_texture_1.jpeg",
            roughness=0.6)

        ##################################################################################
        #### You can disable this below ####
        # Light
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

    def _get_corners(self):
        cloth_positions = self._cloth_plain.root_physx_view.get_positions().reshape(self.num_envs, -1, 3)
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
        cloth_positions = self._cloth_plain.root_physx_view.get_positions()
        cloth_positions = cloth_positions.view(self.num_envs, -1, 3)

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
        # Step DMP
        clamp_loc = 0.5
        if self.action_steps[-1] == self.num_steps_to_settle:
            print('Setting DMP parameters')
            dmp_parameters = self.actions.clone()
            ### Set cloth node start
            dmp_parameters[:, 0] = dmp_parameters[:, 0].clamp(-clamp_loc, clamp_loc)
            dmp_parameters[:, 1] = dmp_parameters[:, 1].clamp(-clamp_loc, clamp_loc)

            ### Set cloth node goal
            dmp_parameters[:, 3] = dmp_parameters[:, 3].clamp(-clamp_loc, clamp_loc)
            dmp_parameters[:, 4] = dmp_parameters[:, 4].clamp(-clamp_loc, clamp_loc)
            dmp_parameters[:, 5] = dmp_parameters[:, 5] * 0 + 0.11

            self.action_penalties[self.action_steps] = dmp_parameters[:,6:-1].abs().mean()
            
            # dmp_parameters = dmp_parameters.clamp(-1.0, 1.0)

            dmp_parameters[:,0:3] += self.scene.env_origins
            dmp_parameters[:,3:6] += self.scene.env_origins

            self.actions = dmp_parameters
            dmp_tau = torch.zeros(self.num_envs, device=self.device) + self.cfg.episode_length_s
            dmp_tau -= self.num_steps_to_settle / 120
            self.actions[:, -1] = dmp_tau

            ### Get index of the node that's closest to start position
            self.grab_node_idx, cloth_pos = self._get_node_idx_from_start_pos(dmp_parameters)
            for i in range(self.num_envs):
                dmp_parameters[i, 0] = cloth_pos[i, self.grab_node_idx[i], 0]
                dmp_parameters[i, 1] = cloth_pos[i, self.grab_node_idx[i], 1]
                dmp_parameters[i, 2] = cloth_pos[i, self.grab_node_idx[i], 2]

            hard_reset = (self.episode_length_buf == 0).all()
            reset_dmp_indices = torch.arange(0, dmp_parameters.shape[0])

            if len(reset_dmp_indices) > 0:
                self.dmp_integrator.reset_indices(
                    reset_dmp_indices,
                    dmp_parameters,
                    dmp_tau,
                    dt=self.physics_dt,
                    variant=2,
                    soft_reset=not (hard_reset),
                )
                self.dmp_initialized = True

        if (self.action_steps[-1] > self.num_steps_to_settle).all():
        # if (self.episode_length_buf[-1] == 0 and 
        #     self.action_steps[-1] > self.num_steps_to_settle).all():
            t, y, dy, ddy = self.dmp_integrator.step()
            self.dmp_integrator.x[t >= self.actions[:, -1]] = 0.1353

            # TODO: clamp y to acceptable ranges and do not clamp DMP weights? e.g.
            y = y - self.scene.env_origins
            y[:,0] = y[:,0].clamp(-clamp_loc, clamp_loc)
            y[:,1] = y[:,1].clamp(-clamp_loc, clamp_loc)
            y[:,2] = y[:,2].clamp(0.11, 0.3)
            y = y + self.scene.env_origins

            current_cloth_pos = self._cloth_plain.root_physx_view.get_positions().view(self.num_envs, -1, 3) # [E,N,3]
            for i in range(self.num_envs):
                current_cloth_pos[i, self.grab_node_idx[i]] = y[i]
            self._cloth_plain.root_physx_view.set_positions(current_cloth_pos, torch.arange(0, self.num_envs, device=self.device))

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
        truncated = self.episode_length_buf >= self.max_episode_length * self.cfg.num_tries
        if truncated.any():
            truncated[:] = True
        return terminated, truncated

    def compute_rewards(self):
        corners = self._get_corners()
        cloth_positions = self._cloth_plain.root_physx_view.get_positions().reshape(self.num_envs, -1, 3)
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

    def _reset_idx(self, env_ids: torch.Tensor | None):
        super()._reset_idx(env_ids)

        self.action_steps[env_ids] = 0

        # Cache default cloth/handles state
        if not hasattr(self, "default_states") or self.default_states is None:
            self.default_states = {}
            cloth_positions = self._cloth_plain.root_physx_view.get_positions()
            self.default_states["_cloth_plain"] = cloth_positions.clone()

        # Reset cloth/handles
        zeros_vel = torch.zeros(
            (len(env_ids), self._cloth_plain.root_physx_view.max_particles_per_cloth * 3), device=self.device
        )
        self._cloth_plain.root_physx_view.set_velocities(zeros_vel, indices=env_ids)
        self._cloth_plain.root_physx_view.set_positions(self.default_states["_cloth_plain"][env_ids], indices=env_ids)
        self._cloth_plain.update(self.physics_dt)

    # -------------------------
    # Observations
    # -------------------------

    def _get_observations(self) -> torch.Tensor:

        ##### Direct access ######
        # Cloth corners (flattened)
        current_corner_obs = self._get_corners() - self.scene.env_origins.unsqueeze(1)
        current_corner_obs = current_corner_obs.reshape(self.num_envs, -1)

        observations = current_corner_obs

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