# run with: ./isaaclab.sh -p path/to/this_script.py

from isaaclab.app import AppLauncher
app = AppLauncher(headless=False).app  # set headless=True if you want no GUI

import torch
import isaaclab.sim as sim_utils
from isaaclab.sim import SimulationCfg
from isaaclab.envs import DirectRLEnv, DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.assets import RigidObjectCfg, RigidObject
from isaaclab.terrains import TerrainImporterCfg

from isaaclab.utils import configclass

# ---------- Configs ----------
@configclass
class FallingCubeEnvCfg(DirectRLEnvCfg):
    episode_length_s = 10
    max_episode_length = int(episode_length_s * 120)
    decimation = 1  # advance physics every step
    action_space = 0
    observation_space = 0
    state_space = 0

    sim: SimulationCfg = SimulationCfg(
        dt=1/120,
        render_interval=1,
        physics_material=sim_utils.RigidBodyMaterialCfg(),
        physx=sim_utils.PhysxCfg(),
    )

    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=1, env_spacing=2.0, replicate_physics=False
    )

    # Ground plane
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",
        debug_vis=False,
        physics_material=sim_utils.RigidBodyMaterialCfg(),
    )

    # Spawn a dynamic cube at z=1.0
    cube = RigidObjectCfg(
        prim_path="/World/envs/env_.*/Cube",
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 1.0), rot=(1,0,0,0)),
        spawn=sim_utils.CuboidCfg(
            size=(0.2, 0.2, 0.2),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=False),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            physics_material=sim_utils.RigidBodyMaterialCfg(),
            # color=(0.8, 0.1, 0.1),
        ),
    )

# ---------- Env ----------
class FallingCubeEnv(DirectRLEnv):
    cfg: FallingCubeEnvCfg

    def __init__(self, cfg: FallingCubeEnvCfg, render_mode=None):
        super().__init__(cfg, render_mode)

    def _setup_scene(self):
        # ground
        self.cfg.terrain.num_envs = self.scene.cfg.num_envs
        self.cfg.terrain.env_spacing = self.scene.cfg.env_spacing
        self._terrain = self.cfg.terrain.class_type(self.cfg.terrain)

        # cube
        self._cube: RigidObject = self.cfg.cube.class_type(self.cfg.cube)

        # register into scene and clone envs
        self.scene.rigid_objects["cube"] = self._cube
        self.scene.clone_environments(copy_from_source=True)

    def _pre_physics_step(self, actions: torch.Tensor):
        pass

    def _apply_action(self):
        pass

    def _get_dones(self):
        terminated = 0
        truncated = self.episode_length_buf >= self.max_episode_length - 1
        return terminated, truncated

    def _get_rewards(self):
        return torch.zeros(self.num_envs, device=self.device)

    def _reset_idx(self, env_ids):
        # reset cube to initial height
        self._cube.write_root_state_to_sim(self._cube.data.default_root_state[env_ids], env_ids)

    def _get_observations(self):
        return {}

# ---------- Run ----------
if __name__ == "__main__":
    env = FallingCubeEnv(FallingCubeEnvCfg())
    env.reset()

    # start physics
    env.sim.play()

    print("Simulating…")
    i = 0
    while i < 5:
        env.reset()
        for i in range(240):
            # one physics step
            env.step(torch.empty((env.num_envs, 0), device=env.device))

            # live pose from PhysX buffers (tensor on GPU usually)
            z = env._cube.data.root_pos_w[0, 2].item()
            print(f"step {i:03d}: cube z = {z:.4f}")

    env.sim.stop()
    app.close()
