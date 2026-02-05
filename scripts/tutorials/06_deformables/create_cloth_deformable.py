"""
Run this using script editor in Isaac Sim!
"""

from pxr import UsdGeom, Gf, PhysxSchema
import omni.kit.commands
import omni.usd
from omni.physx.scripts import physicsUtils, deformableUtils

stage = omni.usd.get_context().get_stage()

# -------------------------------
# Parameters
# -------------------------------
cloth_path = "/World/Cloth"
mesh_size = 100         # 1 meter cube
mesh_resolution = 10    # subdivisions
simulation_resolution = 10
spawn_pos = Gf.Vec3f(0.0, 0.0, 0.2)  # start a bit above ground
scale_flatten = Gf.Vec3f(0.2, 0.7, 0.005)  # thin cube

# -------------------------------
# Create Cube for Cloth
# -------------------------------
_, tmp_path = omni.kit.commands.execute("CreateMeshPrim",
                                        prim_type="Cube",
                                        select_new_prim=False,
                                        u_patches=mesh_resolution,
                                        v_patches=mesh_resolution,
                                        w_patches=2,
                                        half_scale=mesh_size/2)

omni.kit.commands.execute("MovePrim", path_from=tmp_path, path_to=cloth_path)
omni.usd.get_context().get_selection().set_selected_prim_paths([], False)

cloth_mesh = UsdGeom.Mesh.Get(stage, cloth_path)
cloth_mesh.GetPrim().GetAttribute("xformOp:translate").Set(spawn_pos)
cloth_mesh.GetPrim().GetAttribute("xformOp:scale").Set(scale_flatten)
cloth_mesh.CreateDisplayColorAttr().Set([Gf.Vec3f(0.0, 0.0, 0.8)])  # blue

# -------------------------------
# Make it deformable
# -------------------------------
deformableUtils.add_physx_deformable_body(
    stage,
    cloth_mesh.GetPath(),
    collision_simplification=True,
    simulation_hexahedral_resolution=simulation_resolution,
    self_collision=True,
)

# -------------------------------
# Collision settings
# -------------------------------
collision_api = PhysxSchema.PhysxCollisionAPI.Apply(cloth_mesh.GetPrim())
collision_api.GetRestOffsetAttr().Set(0.008)
collision_api.GetContactOffsetAttr().Set(0.010)

# -------------------------------
# Material
# -------------------------------
material_path = omni.usd.get_stage_next_free_path(stage, cloth_path + "/ClothMaterial", True)
deformableUtils.add_deformable_body_material(
    stage,
    material_path,
    youngs_modulus=1e6,
    poissons_ratio=0.49,
    damping_scale=0.5,
    dynamic_friction=1.5,
    density=10.0,
)
physicsUtils.add_physics_material_to_prim(stage, cloth_mesh.GetPrim(), material_path)

print(f"Deformable cloth created at {cloth_path}")


##### ISAAC LAB CODE ######
# from isaaclab.assets import DeformableObjectCfg, RigidObjectCfg
# from isaaclab.envs import DirectRLEnv, DirectRLEnvCfg
# from isaaclab.scene import InteractiveSceneCfg
# from isaaclab.utils import configclass

# import torch

# @configclass
# class ClothSimulationEnvCfg(DirectRLEnvCfg):

# 		decimation = 2
# 		episode_length_s = 2
# 		observation_space = 2
# 		action_space = 2
# 		scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=1, env_spacing=2.5, replicate_physics=False)


# 		cloth_plain = DeformableObjectCfg(
# 		    prim_path="/World/Cloth",
# 		    init_state=DeformableObjectCfg.InitialStateCfg(pos=(0, 0, 0), rot=(1, 0, 0, 0)),
# 		    spawn=None,
# 		)

# class ClothSimulationEnv(DirectRLEnv):
#     def __init__(self, cfg, **kwargs):
#         super().__init__(cfg, **kwargs)
#         self.cloth = cfg.cloth_plain.class_type(cfg.cloth_plain)
#         print(self.cloth.data)


# cfg = ClothSimulationEnvCfg()
# env = ClothSimulationEnv(cfg)
