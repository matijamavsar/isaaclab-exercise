# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
from __future__ import annotations

import os
from datetime import datetime

import torch
import torch.nn.functional as F
import cv2
import matplotlib.pyplot as plt

from isaaclab.utils import configclass
import isaaclab.sim as sim_utils
from isaaclab.sim import SimulationCfg, PhysxCfg, RenderCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.envs import DirectRLEnv, DirectRLEnvCfg
from isaaclab.assets import Articulation, ArticulationCfg, RigidObjectCfg, ParticleClothObjectCfg
from isaaclab.managers import SceneEntityCfg
from isaaclab.controllers import DifferentialIKController, DifferentialIKControllerCfg
from isaaclab.utils.math import subtract_frame_transforms
from isaacsim.core.utils.prims import get_prim_at_path
from isaaclab.sensors import ContactSensorCfg, CameraCfg

from pxr import UsdShade, UsdGeom, UsdPhysics, Sdf, Gf
import omni.usd
import omni.kit.commands
from omni.physx.scripts import physicsUtils

from isaaclab_assets import FRANKA_PANDA_HIGH_PD_CFG

from .min_jerk_traj import generate_minimum_jerk
from .dmp_integrator import BatchDMPIntegrator

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
    """2-pole Butterworth low-pass, batched."""
    def __init__(self, batch_size: int, dof: int, fc: float, dt: float, device=None, dtype=torch.float32):
        self.B, self.D = batch_size, dof
        self.b0, self.b1, self.b2, self.a1, self.a2 = butter2_biquad_coeffs(fc, dt, device, dtype)
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
        self.value = None


@configclass
class FrankaDMPClothPlaceEnvCfg(DirectRLEnvCfg):
    # Env
    episode_length_s = 3
    max_episode_length = episode_length_s * 120
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
    dof_velocity_scale = 0.1
    filter_kernel_size = 7
    seed = 42
    torch.manual_seed(seed)

    # Simulation
    sim: SimulationCfg = SimulationCfg(
        dt=1 / 120,
        render_interval=8,
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

    # Scene
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=1, env_spacing=2.0, replicate_physics=False)

    # Cloth & handles
    cloth = ParticleClothObjectCfg(
        prim_path="/World/envs/env_.*/Cloth",
        init_state=ParticleClothObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 0.356), rot=(0.5, 0.5, 0.5, 0.5)),
        spawn=sim_utils.UsdFileCfg(
            usd_path="/workspace/isaaclab/source/isaaclab_tasks/isaaclab_tasks/sim/particle_cloth_small.usd",
            scale=(1.0, 1.0, 1.0),
        ),
    )
    cloth_plain = ParticleClothObjectCfg(
        prim_path="/World/envs/env_.*/Cloth",
        init_state=ParticleClothObjectCfg.InitialStateCfg(pos=(0, 0, 0), rot=(1, 0, 0, 0)),
        spawn=None,
    )
    handle_1 = RigidObjectCfg(
        prim_path="/World/envs/env_.*/Cloth/Cube",
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0, 0, 0), rot=(1, 0, 0, 0)),
        spawn=None,
    )

    # Single robot
    robot_1 = FRANKA_PANDA_HIGH_PD_CFG.replace(
        prim_path="/World/envs/env_.*/Robot1",
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
            pos=(0.0, -0.3, 0.0),
            rot=(0.7071, 0.0, 0.0, 0.7071),
        ),
    )

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

    # Ground
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
        "spread_reward": Reward(True, 1.0),
        "height_reward": Reward(True, 1.0),
        "corner_x_reward": Reward(True, 10.0),
        "direction_reward": Reward(True, 15.0),
        "action_penalty": Reward(True, 1e-2),
    }


def make_circle_path_torch(center=(0, 0, 3), radius=10, num_points=720, device="cpu"):
    cx, cy, cz = center
    angles = torch.linspace(torch.pi, torch.pi + 2 * torch.pi, num_points, device=device, requires_grad=False)
    xs = cx + radius * torch.cos(angles)
    ys = cy + radius * torch.sin(angles)
    zs = torch.full_like(xs, cz)
    return torch.stack([xs, ys, zs], dim=1)


class FrankaDMPClothPlaceEnv(DirectRLEnv):
    cfg: FrankaDMPClothPlaceEnvCfg

    def __init__(self, cfg: FrankaDMPClothPlaceEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        self.seed(self.cfg.seed)

        # Recording
        now = datetime.now().strftime("%Y%m%d_%H%M%S")
        if self.cfg.enable_camera_recording:
            self.video_folder = os.path.join("./logs/videos", now)
            os.makedirs(self.video_folder, exist_ok=True)
        self.video_writer = None
        self.render_count = 0

        # State buffers
        self.action_steps = torch.zeros(self.num_envs, device=self.device).int()
        self.iteration_step = torch.tensor(0, device=self.device)
        self.dt = self.cfg.sim.dt * self.cfg.decimation
        self.ee_jacobi_idx = 7
        self.default_states = None
        self.joints_created = False
        self.reset_count = 0

        # One robot only
        self.robots = {}
        self.robot_entity_cfg = {}
        self._rewards = self.cfg.rewards

        # IK
        diff_ik_cfg = DifferentialIKControllerCfg(command_type="pose", use_relative_mode=False, ik_method="dls")
        self._ik_controller = {}

        # Filters / buffers
        self.lp1 = {}
        self.prev_joints = {}
        self.filtered_dof_targets = {}
        self.robot_dof_targets = {}
        self.robot_dof_lower_limits = {}
        self.robot_dof_upper_limits = {}
        self.dmp_integrator = BatchDMPIntegrator(N_basis=25, dof=2, device=self.device)

        # History buffers
        H = self.cfg.decimation
        self.prev_corners_buf = torch.zeros((H, self.num_envs, 6), device=self.device)
        self.prev_abs_buf = torch.zeros((H, self.num_envs, 2), device=self.device)

        # Probing path (minimum jerk in x,z)
        abs_points = [
            [0.0, 0.0, 0.81],
            [-0.3, 0.0, 0.81],
            [0.2, 0.0, 0.81],
            [0.0, 0.0, 0.81],
        ]
        durations_relative = torch.tensor([1, 2, 1], device=self.device)
        durations = self.cfg.episode_length_s * durations_relative / durations_relative.sum()
        self._probing_traj = generate_minimum_jerk(waypoints=abs_points, durations=durations.cpu().tolist(),
                                                   num_points=self.cfg.decimation)
        self._probing_traj = torch.tensor(self._probing_traj, device=self.device)
        self._probing_traj = torch.cat(
            (self._probing_traj[0].unsqueeze(0), self._probing_traj, self._probing_traj[-1].unsqueeze(0)), dim=0
        )
        self.probing_done = False

        # Camera helpers
        self.init_camera_pose = torch.tensor([-10.0, -10.0, 0.3, 0.9238795, 0.0, 0.0, 0.3826834], device=self.device)
        self.circle_camera_path = make_circle_path_torch().to(self.device)

        # Per-step logs
        self.y_values = torch.zeros((self.cfg.max_episode_length, self.num_envs), device=self.device)
        self.z_abs_pos = torch.zeros((self.cfg.max_episode_length, self.num_envs), device=self.device)
        self.action_penalties = torch.zeros((self.cfg.max_episode_length, self.num_envs), device=self.device)

        # Init per-robot buffers/limits
        self.robots["robot_1"] = self._robot_1
        num_joints = self._robot_1.num_joints
        self.prev_joints["robot_1"] = torch.zeros((self.cfg.filter_kernel_size, self.num_envs, 7), device=self.device)
        self.filtered_dof_targets["robot_1"] = torch.zeros((self.num_envs, num_joints), device=self.device)
        self.robot_dof_targets["robot_1"] = torch.zeros((self.num_envs, num_joints), device=self.device)
        self.robot_dof_lower_limits["robot_1"] = self._robot_1.data.soft_joint_pos_limits[0, :, 0].to(device=self.device)
        self.robot_dof_upper_limits["robot_1"] = self._robot_1.data.soft_joint_pos_limits[0, :, 1].to(device=self.device)
        self.lp1["robot_1"] = BiquadLP2Batch(self.scene.cfg.num_envs, 7, fc=1.0, dt=self.physics_dt, device=self.device)
        self.robot_entity_cfg["robot_1"] = SceneEntityCfg("robot_1", joint_names=["panda_joint.*"],
                                                          body_names=["panda_hand"])
        self._ik_controller["robot_1"] = DifferentialIKController(
            DifferentialIKControllerCfg(command_type="pose", use_relative_mode=False, ik_method="dls"),
            num_envs=self.num_envs, device=self.device
        )

    # --- Scene ---
    def _setup_scene(self):
        # Robot
        self._robot_1 = Articulation(self.cfg.robot_1)
        self.scene.articulations["robot_1"] = self._robot_1

        # Terrain
        self.cfg.terrain.num_envs = self.scene.cfg.num_envs
        self.cfg.terrain.env_spacing = self.scene.cfg.env_spacing
        self._terrain = self.cfg.terrain.class_type(self.cfg.terrain)

        # Cloth and handles
        self._cloth = self.cfg.cloth.class_type(self.cfg.cloth)
        self._cloth_plain = self.cfg.cloth_plain.class_type(self.cfg.cloth_plain)
        self._handle_1 = self.cfg.handle_1.class_type(self.cfg.handle_1)

        self.scene.clone_environments(copy_from_source=True)

        # Slight randomization of cloth Y scale per env
        for env_idx in range(self.scene.cfg.num_envs):
            cloth_prim = get_prim_at_path(f"/World/envs/env_{env_idx}/Cloth")
            random_scale = torch.tensor([1.0, torch.rand(1).item() * (1.0 - 0.6) + 0.6, 1.0])
            cloth_prim.GetAttribute("xformOp:scale").Set(Gf.Vec3f(*random_scale.tolist()))
            cloth_prim.GetAttribute("xformOp:translate").Set(
                Gf.Vec3f(0.0, 0.0, 0.356 + ((1 - random_scale[1]) * 0.356).item())
            )

        # Optional camera
        if self.cfg.enable_camera_recording:
            self._camera = self.cfg.camera.class_type(self.cfg.camera)

        # Simple light
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

        # Visible ground plane material (dark gray)
        stage = omni.usd.get_context().get_stage()
        visual_plane_path = "/World/ground/GroundPlane/VisualPlane"
        if not stage.GetPrimAtPath(visual_plane_path):
            visual_plane = UsdGeom.Mesh.Define(stage, visual_plane_path)
            visual_plane.GetPointsAttr().Set(
                [Gf.Vec3f(-50, -50, 0), Gf.Vec3f(50, -50, 0), Gf.Vec3f(50, 50, 0), Gf.Vec3f(-50, 50, 0)]
            )
            visual_plane.GetFaceVertexIndicesAttr().Set([0, 1, 2, 2, 3, 0])
            visual_plane.GetFaceVertexCountsAttr().Set([3, 3])
            visual_plane.AddTranslateOp().Set(Gf.Vec3f(0, 0, 0))

            mat_path = "/World/Materials/GroundMaterial"
            mat_prim = stage.DefinePrim(mat_path, "Material")
            material = UsdShade.Material(mat_prim)
            shader = UsdShade.Shader.Define(stage, mat_path + "/Shader")
            shader.CreateIdAttr("UsdPreviewSurface")
            shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(0.1, 0.1, 0.1))
            shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.0)
            shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(0.0)
            UsdShade.MaterialBindingAPI(visual_plane).Bind(material)

    # --- Helpers ---
    def _get_absolute_pose(self):
        """For single robot: return panda_hand pose (world)."""
        ee_pose_w = self._robot_1.data.body_state_w[:, self.robot_entity_cfg["robot_1"].body_ids[0], 0:7]
        return ee_pose_w.clone()

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

    def _create_fixed_joint_to_handle(self, env_ids):
        """Attach right cloth cube to the robot's left finger (one fixed joint per env)."""
        stage = self.scene.stage
        for idx in env_ids:
            cube_path = Sdf.Path(f"/World/envs/env_{int(idx)}/Cloth/Cube")
            finger_path = Sdf.Path(f"/World/envs/env_{int(idx)}/Robot1/panda_leftfinger")
            joint_path = finger_path.AppendElementString("fixedJoint")
            j = UsdPhysics.FixedJoint.Define(stage, joint_path)
            j.CreateBody0Rel().SetTargets([finger_path])
            j.CreateBody1Rel().SetTargets([cube_path])
            j.CreateLocalPos0Attr().Set(Gf.Vec3f(0, 0, 0.05))
            j.CreateLocalRot0Attr().Set(Gf.Quatf(0.7071,-0.7071,0.0,0.0))
            j.CreateLocalPos1Attr().Set(Gf.Vec3f(0, 0, 0))
            j.CreateLocalRot1Attr().Set(Gf.Quatf(1, 0, 0, 0))

    # --- RL loop methods ---
    def _pre_physics_step(self, actions: torch.Tensor):
        dmp_parameters = actions.clone()
        dmp_parameters[:, 2] = dmp_parameters[:, 2].clamp(-0.7, 0.7)
        dmp_parameters[:, 3] = dmp_parameters[:, 3].clamp(0.15, 1.0)
        self.actions = dmp_parameters

        current_abs = self._get_absolute_pose()
        current_abs[:, 0:3] -= self.scene.env_origins
        dmp_parameters[:, 0] = current_abs[:, 0]  # x
        dmp_parameters[:, 1] = current_abs[:, 2]  # z

        # tau = episode length (one update)
        dmp_tau = (torch.zeros(self.num_envs, device=self.device) + self.cfg.episode_length_s // self.cfg.num_updates_per_episode) - 0.5
        self.actions[:, -1] = dmp_tau

        # (Optional) reset DMP indices each call — preserved behavior
        reset_dmp_indices = torch.arange(0, dmp_parameters.shape[0])

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
        # Init LP filter at episode start
        if self.action_steps[0] == 0:
            q0 = self._robot_1.data.joint_pos[:, :7]
            self.lp1["robot_1"].reset(q0)

        t, y, dy, ddy = self.dmp_integrator.step()
        self.dmp_integrator.x[t >= self.actions[:,-1]] = 0.1353
        # y[:,1] = y[:,1].clamp(0.15, 1.0)

        # if not hasattr(self, "prev_y"):
        #     self.prev_y = y

        # max_speed = 0.4  # m/s
        # dt = getattr(self, "physics_dt", 1.0/120.0)
        # max_step = max_speed * dt

        # delta = y - self.prev_y                           # (B, 3)
        # dist  = torch.linalg.norm(delta, dim=-1, keepdim=True)  # (B, 1)

        # # scale <= 1.0 when step is too large; =1.0 when already within limit
        # scale = torch.clamp(max_step / (dist + 1e-9), max=max_speed)  # (B, 1)
        # y = self.prev_y + delta * scale

        # self.prev_y = y

        # Probing phase uses minimum-jerk trajectory in XZ
        current_abs = self._get_absolute_pose()
        self.z_abs_pos[self.action_steps] = current_abs[:, 2].clone()

        if self.cfg.enable_camera_recording:
            in_window = (self.iteration_step % 20000) < 720
            just_entered = (in_window and self.video_writer is None)

            if just_entered:
                self.render_count = 0
                self._setup_camera_writer(self.iteration_step)

            # If we are in the window, capture & write one frame:
            if in_window and self.video_writer is not None:
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
            y[:, 0] = self._probing_traj[self.action_steps][:, 0]  # target x
            y[:, 1] = self._probing_traj[self.action_steps][:, 2]  # target z
            self.inverse_kinematics_single(y)

            # write targets
            if self.cfg.write_joint_state:
                self._robot_1.write_joint_position_to_sim(self.robot_dof_targets["robot_1"])
            else:
                self._robot_1.set_joint_position_target(self.robot_dof_targets["robot_1"])

            # cache obs history
            corners = self._get_corners()[:, [2, 3]] - self.scene.env_origins.unsqueeze(1)  # (N,2,3)
            self.prev_corners_buf = self.prev_corners_buf.roll(-1, dims=0)
            self.prev_corners_buf[-1] = corners.reshape(self.num_envs, -1)
            self.prev_abs_buf = self.prev_abs_buf.roll(-1, dims=0)
            self.prev_abs_buf[-1] = current_abs[:, [0, 2]] - self.scene.env_origins[:, [0, 2]]

            if (self.action_steps == self.cfg.max_episode_length - 1).all():
                self.probing_done = True

        elif (self.action_steps < (self.cfg.max_episode_length - 60)).all():
            # Hold current absolute x,z
            target_y = y
            self.inverse_kinematics_single(target_y)

            if self.cfg.write_joint_state:
                q_des = self.robot_dof_targets["robot_1"][:, :7]
                q_smooth = self.lp1["robot_1"].step(q_des)
                self.robot_dof_targets["robot_1"][:, :7] = q_smooth
                self._robot_1.write_joint_position_to_sim(self.robot_dof_targets["robot_1"])
            else:
                self._robot_1.set_joint_position_target(self.robot_dof_targets["robot_1"])

        self.iteration_step += 1
        self.action_steps += 1

    def inverse_kinematics_single(self, y_xz: torch.Tensor):
        """
        Move the single arm EE to absolute (x,z) while keeping world Y and a fixed down-facing orientation.
        y_xz: [N,2] -> target (x,z) in absolute/world frame relative to env origin.
        """
        current_abs = self._get_absolute_pose()

        target_abs = current_abs.clone() * 0.0
        target_abs[:, 0] = y_xz[:, 0]  # x
        target_abs[:, 2] = y_xz[:, 1]  # z
        target_abs[:, 0:3] += self.scene.env_origins
        target_abs[:, 3:] = torch.tensor([0, 0.7071, 0.7071, 0], device=self.device)  # fixed orientation (x-axis 180deg)

        # Transform to robot base
        root_pose_w = self._robot_1.data.root_state_w[:, 0:7]
        target_pos_b, _ = subtract_frame_transforms(
            root_pose_w[:, 0:3], root_pose_w[:, 3:7], target_abs[:, 0:3], target_abs[:, 3:7]
        )
        target_quat_b = torch.zeros((self.num_envs, 4), device=self.device)
        target_quat_b[:,1] = 1.0

        self._ik_controller["robot_1"].set_command(torch.cat([target_pos_b, target_quat_b], dim=-1))

        robot = self._robot_1
        robot_entity_cfg = self.robot_entity_cfg["robot_1"]

        ee_pose_w = robot.data.body_state_w[
            :, robot_entity_cfg.body_ids[0], 0:7]

        # Obtain robot's Jacobian matrix
        jacobian_w = robot.root_physx_view.get_jacobians()[
            :, self.ee_jacobi_idx, :, 
            robot_entity_cfg.joint_ids]
        base_rot = robot.data.root_quat_w
        jacobian = self._ik_controller["robot_1"].get_jacobian_in_root_frame(jacobian_w, base_rot)

        # Get root pose and joint positions
        root_pose_w = robot.data.root_state_w[:, 0:7]
        current_joint_pos = robot.data.joint_pos[
            :, robot_entity_cfg.joint_ids]

        ee_pos_b, ee_quat_b = subtract_frame_transforms(
            root_pose_w[:, 0:3], root_pose_w[:, 3:7], 
            ee_pose_w[:, 0:3], ee_pose_w[:, 3:7]
        )

        # Compute new joint positions using IK
        self.robot_dof_targets["robot_1"][:, 0:7] = self._ik_controller["robot_1"].compute(
            ee_pos_b, ee_quat_b, jacobian, current_joint_pos)

        padded_joint_pos = F.pad(input=current_joint_pos, pad=(0,2), mode='constant', value=0)
        self.joint_change = self.robot_dof_targets["robot_1"] - padded_joint_pos

        if self.cfg.write_joint_state:
            ### Limit change of joints if writing directly
            self.clamped_joint_change = self.joint_change # .clamp(-1e-1, 1e-1)
            new_targets = padded_joint_pos + self.clamped_joint_change
        else:
            ### Or just send it to target positions
            self.clamped_joint_change = self.joint_change # .clamp(-1e-1, 1e-1)
            new_targets = padded_joint_pos + self.clamped_joint_change

        self.robot_dof_targets["robot_1"] = new_targets

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        terminated = 0
        truncated = self.episode_length_buf >= self.max_episode_length
        if truncated.any():
            truncated[:] = True
        return terminated, truncated

    def _get_corners(self):
        cloth_positions = self._cloth.root_physx_view.get_positions().reshape(self.num_envs, -1, 3)
        side_len = int(torch.sqrt(torch.tensor(cloth_positions.shape[1])).to(self.device))
        corners = cloth_positions[:, [-side_len, -1, 0, side_len - 1], :]
        return corners

    def compute_rewards(self):
        corners = self._get_corners()
        cloth_positions = self._cloth_plain.root_physx_view.get_positions().reshape(self.num_envs, -1, 3)
        mean_cloth_height = cloth_positions[:, :, 2].mean(dim=-1)

        rewards = {}
        # Spread
        pairwise_sum = torch.zeros(self.num_envs, device=self.device)
        cnt = 0.0
        for i in range(4):
            for j in range(i + 1, 4):
                pairwise_sum += torch.norm(corners[:, i] - corners[:, j], dim=-1)
                cnt += 1.0

        free_x = corners[:, [2, 3], 0] - self.scene.env_origins[:, 0].unsqueeze(1)
        grasped_x = corners[:, [0, 1], 0] - self.scene.env_origins[:, 0].unsqueeze(1)

        if not self.cfg.use_weighted_atan_rewards and not self.cfg.use_weighted_exp_rewards:
            rewards["height_reward"] = (1.0 / (0.1 + mean_cloth_height)) * self._rewards["height_reward"].scale
            rewards["height_reward"][mean_cloth_height > 0.4] *= 0.0
            rewards["spread_reward"] = (pairwise_sum / cnt) * self._rewards["spread_reward"].scale
            rewards["corner_x_reward"] = (free_x.mean(-1)) * self._rewards["corner_x_reward"].scale
            rewards["direction_reward"] = (free_x.mean(-1) - grasped_x.mean(-1)) * self._rewards["direction_reward"].scale
        elif self.cfg.use_weighted_atan_rewards:
            rewards["height_reward"] = 0.5 + (1.0 / torch.pi) * torch.atan(1.0 / (0.1 + mean_cloth_height))
            rewards["spread_reward"] = 0.5 + (1.0 / torch.pi) * torch.atan(pairwise_sum / cnt)
            rewards["corner_x_reward"] = 0.5 + (1.0 / torch.pi) * torch.atan(free_x.mean(-1))
        else:  # weighted_exp
            rewards["height_reward"] = torch.exp(1.0 / (0.1 + mean_cloth_height))
            rewards["spread_reward"] = torch.exp(pairwise_sum / cnt)
            rewards["corner_x_reward"] = torch.exp(free_x.mean(-1))

        # Action penalty (single robot)
        rewards["action_penalty"] = self.action_penalties.mean(0) * self._rewards["action_penalty"].scale
        return rewards

    def _get_rewards(self) -> torch.Tensor:
        reward_dict = self.compute_rewards()
        for n, v in reward_dict.items():
            self._rewards[n].value = v

        total = torch.zeros(self.num_envs, device=self.device)
        for k, r in self._rewards.items():
            if r.use:
                if "penalty" in k:
                    total -= r.value
                else:
                    total += r.value

        # mask out intermediate steps (keep last-step reward)
        if (self.episode_length_buf != self.cfg.num_updates_per_episode).any():
            total[self.episode_length_buf != self.cfg.num_updates_per_episode] *= 0.0

        # simple invalidation example
        # total[self.z_abs_pos < 0.15].sub_(10.0)

        # Bad reward if deviates too much from zero Y
        total[(self.y_values.abs() > 0.1).any(0)].sub_(10.0)

        self.extras["log"] = {n: v.value.mean() for n, v in self._rewards.items()}
        return total

    def _reset_idx(self, env_ids: torch.Tensor | None):
        super()._reset_idx(env_ids)

        self.action_steps[env_ids] = 0

        if self.default_states is None:
            try:
                self.robot_entity_cfg["robot_1"].resolve(self.scene)
            except Exception:
                pass

        if self.default_states is not None:
            self._cloth_plain.root_physx_view.set_velocities(
                torch.zeros((self.num_envs, self._cloth_plain.root_physx_view.max_particles_per_cloth * 3),
                            device=self.device), indices=env_ids)
            self._cloth_plain.root_physx_view.set_positions(self.default_states["_cloth_plain"][env_ids],
                                                            indices=env_ids)
            self._cloth_plain.update(self.physics_dt)

            self._handle_1.write_root_state_to_sim(self.default_states["_handle_1"][env_ids], env_ids=env_ids)

        # self.sim.pause()

        if self.default_states is None:
            try:
                self.robot_entity_cfg["robot_1"].resolve(self.scene)
            except Exception:
                pass
            self.default_states = {}
            cloth_positions = self._cloth_plain.root_physx_view.get_positions()
            self.default_states["_cloth_plain"] = cloth_positions.clone()
            self.default_states["_handle_1"] = self._handle_1.data.root_state_w.clone(); self.default_states["_handle_1"][:, 7:] = 0.0

        if not self.joints_created:
            self._create_fixed_joint_to_handle(env_ids)
            self.joints_created = True

        # Reset robot state
        joint_pos = self._robot_1.data.default_joint_pos[env_ids].clone()
        joint_vel = self._robot_1.data.default_joint_vel[env_ids].clone()
        self._robot_1.set_joint_position_target(joint_pos, env_ids=env_ids)
        self._robot_1.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)

        # Reset cloth
        self._cloth_plain.root_physx_view.set_velocities(
            torch.zeros((len(env_ids), self._cloth_plain.root_physx_view.max_particles_per_cloth * 3),
                        device=self.device), indices=env_ids)
        self._cloth_plain.root_physx_view.set_positions(self.default_states["_cloth_plain"][env_ids], indices=env_ids)
        self._cloth_plain.update(self.physics_dt)

        # Reset handles
        self._handle_1.write_root_state_to_sim(self.default_states["_handle_1"][env_ids], env_ids=env_ids)

        # self.sim.play()
        self.reset_count += 1

    def _get_observations(self) -> torch.Tensor:
        # Joint state (normalized) + velocities
        robot = self._robot_1
        low = self.robot_dof_lower_limits["robot_1"]
        high = self.robot_dof_upper_limits["robot_1"]
        dof_pos_scaled = (2.0 * (robot.data.joint_pos - low) / (high - low) - 1.0)
        joint_obs = torch.cat((dof_pos_scaled, robot.data.joint_vel * self.cfg.dof_velocity_scale), dim=-1)

        # Action/corner history
        act_hist = self.prev_abs_buf.permute(1, 0, 2)
        act_hist = act_hist[:, ::48].reshape(self.num_envs, -1)

        corner_traj = self.prev_corners_buf.permute(1, 0, 2)
        corner_traj_obs = corner_traj[:, ::48].reshape(self.num_envs, -1)

        if self.cfg.disable_init_motion:
            corner_traj_obs *= 0.0
            act_hist *= 0.0

        # observations = torch.cat((joint_obs, act_hist, corner_traj_obs), dim=-1)
        observations = torch.cat((act_hist, corner_traj_obs), dim=-1)
        return {"policy": observations}
