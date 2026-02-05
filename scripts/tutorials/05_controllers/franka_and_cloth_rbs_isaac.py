# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
This script demonstrates how to use the differential inverse kinematics controller with the simulator.

The differential IK controller can be configured in different modes. It uses the Jacobians computed by
PhysX. This helps perform parallelized computation of the inverse kinematics.

.. code-block:: bash

    # Usage
    ./isaaclab.sh -p scripts/tutorials/05_controllers/ik_control.py

"""

"""Launch Isaac Sim Simulator first."""

import argparse

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Tutorial on using the differential IK controller.")
parser.add_argument("--robot", type=str, default="franka_panda", help="Name of the robot.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to spawn.")
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import torch
import time

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg, RigidObject, RigidObjectCfg, DeformableObject, DeformableObjectCfg
from isaaclab.controllers import DifferentialIKController, DifferentialIKControllerCfg
from isaaclab.managers import SceneEntityCfg
from isaaclab.markers import VisualizationMarkers
from isaaclab.markers.config import FRAME_MARKER_CFG
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab.utils.math import subtract_frame_transforms
from isaaclab.sim import SimulationCfg, PhysxCfg
from isaacsim.core.utils.prims import get_prim_at_path
from pxr import UsdGeom, Sdf, Gf, PhysxSchema
import omni.kit.commands
import omni.physxdemos as demo

from omni.physx.scripts import physicsUtils, deformableUtils

##
# Pre-defined configs
##
from isaaclab_assets import FRANKA_PANDA_HIGH_PD_CFG, UR10_CFG  # isort:skip
from isaaclab.assets import ArticulationCfg
from isaaclab.actuators.actuator_cfg import ImplicitActuatorCfg
from isaaclab.assets import ParticleClothObject, ParticleClothObjectCfg

from robotblockset.franka_isaac import franka_isaac

@configclass
class TableTopSceneCfg(InteractiveSceneCfg):
    """Configuration for a cart-pole scene."""

    # ground plane
    ground = AssetBaseCfg(
        prim_path="/World/defaultGroundPlane",
        spawn=sim_utils.GroundPlaneCfg(),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, 0)),
    )

    # lights
    dome_light = AssetBaseCfg(
        prim_path="/World/Light", spawn=sim_utils.DomeLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75))
    )

    cloth = ParticleClothObjectCfg(
        prim_path="/World/envs/env_.*/Cloth",
        init_state=ParticleClothObjectCfg.InitialStateCfg(pos=(0.5, 0.0, 0.6), rot=(0.5, 0.5, 0.5, 0.5)),
        spawn=sim_utils.UsdFileCfg(
            usd_path="/workspace/isaaclab/source/isaaclab_tasks/isaaclab_tasks/sim/particle_cloth_two_pinch_half_width_box_bendy_noHandles.usd",
            # usd_path="/workspace/isaaclab/source/isaaclab_tasks/isaaclab_tasks/sim/particle_cloth_new.usd",
            scale=(1.0, 1.0, 1.0),
        ),
    )

    # articulation
    if args_cli.robot == "franka_panda":
        robot = FRANKA_PANDA_HIGH_PD_CFG.replace(
            prim_path="{ENV_REGEX_NS}/Robot",
            init_state=ArticulationCfg.InitialStateCfg(
            joint_pos={
                "panda_joint1": 0.0385,
                "panda_joint2": -0.4821,
                "panda_joint3": 0.1572, 
                "panda_joint4": -1.5468,
                "panda_joint5": 0.0822,
                "panda_joint6": 1.0760,
                "panda_joint7": 1.0027,
                "panda_finger_joint.*": 0.02,
            },
            pos=(0.0, 0.0, 0.0),
            # rot=(0.7071, 0, 0, -0.7071),
            rot=(1.0, 0.0, 0.0, 0.0),
        ),)
    elif args_cli.robot == "ur10":
        robot = UR10_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
    else:
        raise ValueError(f"Robot {args_cli.robot} is not supported. Valid: franka_panda, ur10")

def set_cloth_texture(scene, image_path, fallback_tint=(0.9, 0.9, 0.9), roughness=0.55):
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
    for env_i in range(scene.cfg.num_envs):
        root = stage.GetPrimAtPath(f"/World/envs/env_{env_i}/Cloth")
        if not root:
            continue
        for prim in Usd.PrimRange(root):
            if prim.IsA(UsdGeom.Mesh):
                UsdShade.MaterialBindingAPI(prim).Bind(mat)

def run_simulator(sim: sim_utils.SimulationContext, scene: InteractiveScene):
    """Runs the simulation loop."""

    robot = scene["robot"]
    cloth = scene["cloth"]

    robot_rbs = franka_isaac(robot, sim, scene)

    time.sleep(2)
    # robot.SetTCP(robot.GetTCP())
    robot_rbs.GetState()
    robot_rbs.ResetCurrentTarget()

    # Define goals for the arm
    ee_goals = [
        [0.4, 0.0, 0.8, 0.0, 1.0, 0.0, 0.0],
    ]
    ee_goals = torch.tensor(ee_goals, device=sim.device)
    joint_pos_des = torch.zeros((1, 9), device=robot.device)
    joint_pos_des[:,-2:] = 0.04
    # Track the given command
    current_goal_idx = 0

    # Specify robot-specific parameters
    if args_cli.robot == "franka_panda":
        robot_entity_cfg = SceneEntityCfg("robot", joint_names=["panda_joint.*"], body_names=["panda_hand"])
    elif args_cli.robot == "ur10":
        robot_entity_cfg = SceneEntityCfg("robot", joint_names=[".*"], body_names=["ee_link"])
    else:
        raise ValueError(f"Robot {args_cli.robot} is not supported. Valid: franka_panda, ur10")
    # Resolving the scene entities
    robot_entity_cfg.resolve(scene)
    # Obtain the frame index of the end-effector
    # For a fixed base robot, the frame index is one less than the body index. This is because
    # the root body is not included in the returned Jacobians.
    if robot.is_fixed_base:
        ee_jacobi_idx = robot_entity_cfg.body_ids[0] - 1
    else:
        ee_jacobi_idx = robot_entity_cfg.body_ids[0]

    # Define simulation stepping
    sim_dt = sim.get_physics_dt()
    count = 0

    while simulation_app.is_running():

        if count == 360:
            robot_rbs.ResetCurrentTarget()
            robot_rbs.CMoveFor([0.0, 0.0, -0.2], t=2)
            robot_rbs.CMoveFor([0.2, 0.0, 0.2], t=2)
            robot_rbs.CMove([ 0.3305,  0.0012,  0.5436,  0.2225, -0.7545,  0.3191, -0.5286], t=2)
            robot_rbs.CMove([ 0.3305,  0.0012,  0.5436,  0.2225, -0.7545,  0.3191, -0.5286], t=2)

        # scene.write_data_to_sim()
        # perform step
        sim.step()
        # update sim-time
        count += 1
        # update buffers
        scene.update(sim_dt)

def main():
    """Main function."""
    # Load kit helper
    sim_cfg = sim_utils.SimulationCfg(
        dt=1/120,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
        physx=PhysxCfg(
            gpu_max_particle_contacts = 2**21, # Default is 2**20
            gpu_max_soft_body_contacts = 2**21 # Default is 2**20
        ) ,
        device=args_cli.device)
    sim = sim_utils.SimulationContext(sim_cfg)
    # Set main camera
    sim.set_camera_view([2.5, 2.5, 2.5], [0.0, 0.0, 0.0])
    # Design scene
    scene_cfg = TableTopSceneCfg(num_envs=args_cli.num_envs, env_spacing=2.0)
    scene = InteractiveScene(scene_cfg)
    set_cloth_texture(scene, "/workspace/isaaclab/source/isaaclab_tasks/isaaclab_tasks/sim/cloth_texture/cloth_texture_1.jpeg",
        roughness=0.6)
    # Play the simulator
    sim.reset()
    # Now we are ready!
    print("[INFO]: Setup complete...")
    # Run the simulator
    run_simulator(sim, scene)


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
