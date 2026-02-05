from omni.isaac.kit import SimulationApp
from pxr import Usd, UsdGeom, Gf, Sdf, UsdPhysics
from pxr import UsdShade

import omni.kit.commands
from omni.physx.scripts import physicsUtils, particleUtils
import os

stage = omni.usd.get_context().get_stage()

particle_system_path = Sdf.Path("/World/particleSystem")

restOffset = 0.001
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

cloth_mesh_path = Sdf.Path(f"/World/Cloth")
particle_material_path = Sdf.Path("/World/particleMaterial")

# ✅ Create a mesh that will be turned into cloth
u_plane_resolution = 10
v_plane_resolution = 10
plane_width = 15

# ✅ Get the environment center from Isaac Lab
env_center = [0, 0, 0.2]  # This gives (x, y, z) of the env center

# ✅ Adjust the cloth spawn height relative to the environment center
cloth_position = Gf.Vec3f(float(env_center[0]), 
                          float(env_center[1]), 
                          float(env_center[2]))

success, tmp_path = omni.kit.commands.execute(
    "CreateMeshPrimWithDefaultXform",
    prim_type="Plane",
    u_patches=u_plane_resolution,
    v_patches=v_plane_resolution,
    u_verts_scale=1,
    v_verts_scale=1,
    half_scale=0.5 * plane_width,
)

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
stretchStiffness = 100000.0
bendStiffness = 20.0
shearStiffness = 10.0
damping = 0.8

particle_api = particleUtils.add_physx_particle_cloth(
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
cloth_prim = stage.GetPrimAtPath(cloth_mesh_path)
cloth_color_attr = cloth_prim.GetAttribute("primvars:displayColor")
cloth_color_attr.Set([Gf.Vec3f(0.2, 0.0, 0.220)])

print(f"Cloth spawned at {cloth_mesh_path} with {num_verts} vertices")