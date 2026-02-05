# reach_pg_minimal.py
# Minimal REINFORCE that learns joint-position targets to reach a Cartesian pose.

import torch
import torch.nn as nn
import torch.optim as optim
import math
import time
import argparse

### THIS ###
from isaaclab.app import AppLauncher
app_launcher = AppLauncher({"headless": True})
simulation_app = app_launcher.app

### OR THIS ###
# from omni.isaac.kit import SimulationApp
# simulation_app = SimulationApp({"headless": True})

import isaaclab.sim as sim_utils
from isaaclab.sim import SimulationCfg, PhysxCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.envs import DirectRLEnv, DirectRLEnvCfg
from isaaclab.assets import Articulation, ArticulationCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.utils.math import subtract_frame_transforms
from isaaclab.controllers import DifferentialIKController, DifferentialIKControllerCfg
from isaaclab_assets import FRANKA_PANDA_HIGH_PD_CFG
# from source.isaaclab_assets.isaaclab_assets.robots.franka import FRANKA_PANDA_HIGH_PD_CFG
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

# ---------------------------
# 1) Tiny single-env DirectRLEnv
# ---------------------------
@configclass
class ReachEnvCfg(DirectRLEnvCfg):
    episode_length_s = 5.0
    max_episode_length = int(episode_length_s * 120)
    decimation = 1  # control every physics step
    observation_space = 0
    action_space = 0

    # single env, flat ground
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=64, env_spacing=2.5, replicate_physics=False)

    sim: SimulationCfg = SimulationCfg(
        dt=1/120, render_interval=1,
        physx=PhysxCfg()
    )
    terrain = TerrainImporterCfg(prim_path="/World/ground", terrain_type="plane")

    # Franka config
    robot = FRANKA_PANDA_HIGH_PD_CFG.replace(
        prim_path="/World/envs/env_.*/Robot",
        init_state=ArticulationCfg.InitialStateCfg(
            joint_pos={
                "panda_joint1": 0.0,
                "panda_joint2": -0.5,
                "panda_joint3": 0.0,
                "panda_joint4": -1.5,
                "panda_joint5": 0.0,
                "panda_joint6": 1.0,
                "panda_joint7": 0.7,
                "panda_finger_joint.*": 0.02,
            },
            pos=(0.0, 0.0, 0.0),
            rot=(1.0, 0.0, 0.0, 0.0),
        ),
    )

class ReachEnv(DirectRLEnv):
    cfg: ReachEnvCfg

    def __init__(self, cfg: ReachEnvCfg):
        super().__init__(cfg, render_mode=None)
        self.ee_body_name = "panda_hand"  # or "panda_link7" if you prefer
        self.ee_body_idx = self._robot.find_bodies(self.ee_body_name)[0][0]

        # joint limits (soft)
        self.lower = self._robot.data.soft_joint_pos_limits[0, :, 0]
        self.upper = self._robot.data.soft_joint_pos_limits[0, :, 1]

        # target pose in world (position only here; keep a simple reaching task)
        self.target_w = torch.tensor([0.45, 0.0, 0.35], device=self.device).float().unsqueeze(0)

        # convenience: action scale for delta-joints (rad per step)
        self.delta_scale = 0.2

    def _setup_scene(self):
        # add robot and terrain
        self._robot = Articulation(self.cfg.robot)
        self.scene.articulations["robot"] = self._robot

        self.cfg.terrain.num_envs = self.scene.cfg.num_envs
        self.cfg.terrain.env_spacing = self.scene.cfg.env_spacing
        self._terrain = self.cfg.terrain.class_type(self.cfg.terrain)

        # clone envs
        self.scene.clone_environments(copy_from_source=True)

    def _reset_idx(self, env_ids):
        print("Resetting environments")
        super()._reset_idx(env_ids)
        # reset joints to defaults
        q = self._robot.data.default_joint_pos[env_ids].clone()
        dq = self._robot.data.default_joint_vel[env_ids].clone()
        self._robot.set_joint_position_target(q, env_ids=env_ids)
        self._robot.write_joint_state_to_sim(q, dq, env_ids=env_ids)

    def _get_observations(self):
        # obs = [q(7), dq(7), ee_pos_rel(3)] -> shape [1, 17]
        q = self._robot.data.joint_pos
        dq = self._robot.data.joint_vel

        ee_pos_w = self._robot.data.body_pos_w[:, self.ee_body_idx, :]  # [1,3]
        ee_pos_rel = ee_pos_w - self.scene.env_origins - self.target_w

        obs = torch.cat([q, dq, ee_pos_rel], dim=-1)
        return {"policy": obs}

    def _pre_physics_step(self, actions):
        # action is delta-q for 7 joints (ignore fingers)
        dq_cmd = actions[:, :] * self.delta_scale
        q = self._robot.data.joint_pos
        q_target = torch.clamp(q + dq_cmd, self.lower, self.upper)
        # hold fingers constant
        # q_target = torch.cat([q_target, self._robot.data.joint_pos[:, 7:9]], dim=-1)
        self._robot.set_joint_position_target(q_target)
        # self._robot.write_joint_position_to_sim(q_target)

    def _apply_action(self):
        pass

    def _get_rewards(self):
        ee_pos_w = self._robot.data.body_pos_w[:, self.ee_body_idx, :] - self.scene.env_origins
        dist = torch.norm(ee_pos_w - self.target_w, dim=-1)  # [1]
        # shaping reward in [0, 1] roughly via 1/(1+alpha*dist)
        reward = 1.0 / (1.0 + 4.0 * dist)
        # tiny action penalty could be added; omitted for clarity
        return reward

    def _get_dones(self):
        # done if close enough or time limit
        ee_pos_w = self._robot.data.body_pos_w[:, self.ee_body_idx, :]
        dist = torch.norm(ee_pos_w - self.target_w, dim=-1)
        success = (dist < 0.02)
        truncated = self.episode_length_buf >= self.max_episode_length - 1
        terminated = success
        return terminated, truncated


# ---------------------------
# 2) Tiny policy (REINFORCE)
# ---------------------------
class Policy(nn.Module):
    def __init__(self, obs_dim, act_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 128), nn.Tanh(),
            nn.Linear(128, 128), nn.Tanh(),
            nn.Linear(128, act_dim),
        )
        # learnable log-std for Gaussian policy
        self.log_std = nn.Parameter(torch.full((act_dim,), math.log(0.3)))

    def forward(self, obs):
        mean = self.net(obs)
        std = self.log_std.exp().expand_as(mean)
        return mean, std

    def sample(self, obs):
        mean, std = self(obs)
        dist = torch.distributions.Normal(mean, std)
        a = dist.rsample()
        logp = dist.log_prob(a).sum(-1, keepdim=True)
        return a, logp

    def log_prob(self, obs, a):
        mean, std = self(obs)
        dist = torch.distributions.Normal(mean, std)
        return dist.log_prob(a).sum(-1, keepdim=True)


# ---------------------------
# 3) Train loop
# ---------------------------
def main():
    env = ReachEnv(ReachEnvCfg())

    # choose whether to control 7 or 9 joints:
    control_fingers = False
    act_dim = 7 if not control_fingers else 9
    obs_dim = 21

    pi = Policy(obs_dim, act_dim).to(env.device)
    opt = optim.Adam(pi.parameters(), lr=3e-4)
    entropy_coef = 1e-3
    gamma = 0.95
    episodes = 400

    print("Target (x,y,z):", env.target_w[0].tolist())

    for ep in range(episodes):
        logps, entropies, rewards = [], [], []
        obs_dict = env.reset()
        obs = obs_dict[0]["policy"]

        # rollout
        for t in range(env.max_episode_length):
            # if act_dim=7 but env expects 9, pad here:
            a, logp = pi.sample(obs)
            if act_dim == 7:
                pad = torch.zeros((a.shape[0], 2), device=a.device)
                a_env = torch.cat([a, pad], dim=-1)
            else:
                a_env = a

            obs_next, r, terminated, truncated, _ = env.step(a_env)
            done = bool(terminated[0].item()) or bool(truncated[0].item())
            if done:
                break

            # book-keeping
            with torch.no_grad():
                rewards.append(r.detach())
            logps.append(logp)

            # entropy (for exploration)
            mean, std = pi(obs)
            dist = torch.distributions.Normal(mean, std)
            entropies.append(dist.entropy().sum(-1, keepdim=True))

            obs = obs_next["policy"]

        # tensors
        logps = torch.stack(logps)          # [T, 1]
        rewards = torch.stack(rewards)      # [T, 1]
        entropies = torch.stack(entropies)  # [T, 1]

        # returns
        G = torch.zeros_like(rewards)
        running = torch.zeros(1, device=env.device)
        for i in reversed(range(len(rewards))):
            running = rewards[i] + gamma * running
            G[i] = running

        # normalize
        G = (G - G.mean()) / (G.std() + 1e-8)

        # REINFORCE loss with entropy bonus
        loss = -(G * logps.squeeze()).mean() - entropy_coef * entropies.mean()

        opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(pi.parameters(), 1.0)
        opt.step()

        ep_len = len(rewards)
        ep_ret = rewards.sum().item()
        # rough invert shaping to distance (same as your idea)
        last_r = rewards[-1].mean().item()
        last_dist = max(0.0, (1.0 / last_r - 1.0) / 4.0) if last_r > 0 else float('inf')
        print(f"Ep {ep:04d} | len {ep_len:4d} | ret {ep_ret:6.3f} | dist~ {last_dist:6.3f}")


if __name__ == "__main__":
    main()
