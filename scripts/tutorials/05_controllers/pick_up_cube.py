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

    # cuboid = RigidObjectCfg(
    #     prim_path="/World/envs/env_.*/cuboid",
    #     spawn=sim_utils.CuboidCfg(
    #         size=(0.05, 0.05, 0.05),
    #         visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.5, 0.0, 0.5), metallic=0.2),
    #         rigid_props=sim_utils.RigidBodyPropertiesCfg(
    #             solver_position_iteration_count=4, solver_velocity_iteration_count=0
    #         ),
    #         mass_props=sim_utils.MassPropertiesCfg(mass=0.5),
    #         collision_props=sim_utils.CollisionPropertiesCfg()),
    #     init_state=RigidObjectCfg.InitialStateCfg(pos=(0.5, 0.0, 0.0))
    #     )

    # cuboid = DeformableObjectCfg(
    #     prim_path="/World/envs/env_.*/cuboid",
    #     spawn=sim_utils.MeshCuboidCfg(
    #         size=(0.05, 0.05, 0.05),
    #         deformable_props=sim_utils.DeformableBodyPropertiesCfg(rest_offset=0.0, contact_offset=0.001),
    #         visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.5, 0.1, 0.0)),
    #         physics_material=sim_utils.DeformableBodyMaterialCfg(poissons_ratio=0.4, youngs_modulus=1e5),
    #     ),
    #     init_state=DeformableObjectCfg.InitialStateCfg(pos=(0.5, 0.0, 0.025)),
    # )

    # cuboid = DeformableObjectCfg(
    #     prim_path="/World/envs/env_.*/cuboid",
    #     spawn=sim_utils.MeshSphereCfg(
    #         radius=0.03,
    #         deformable_props=sim_utils.DeformableBodyPropertiesCfg(
    #             rest_offset=0.00, 
    #             contact_offset=0.00001,
    #             # solver_position_iteration_count=5,
    #             # simulation_hexahedral_resolution=20,
    #             # self_collision=True,
    #             # kinematic_enabled=True,
    #             vertex_velocity_damping=0
    #             ),
    #         visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.5, 0.1, 0.0)),
    #         physics_material=sim_utils.DeformableBodyMaterialCfg(poissons_ratio=0.4, youngs_modulus=1e5),
    #     ),
    #     init_state=DeformableObjectCfg.InitialStateCfg(pos=(0.5, 0.0, 0.025)),
    # )

    # cuboid = DeformableObjectCfg(
    #     prim_path="/World/envs/env_.*/cuboid",
    #     init_state=DeformableObjectCfg.InitialStateCfg(pos=(0.5, -0.3, 0.1)),
    #     spawn=None,
    # )

    # cloth = ParticleClothObjectCfg(
    #     prim_path="/World/envs/env_.*/cuboid",
    #     init_state=ParticleClothObjectCfg.InitialStateCfg(pos=(0.9, 0, 0.1), rot=(0, 0, 0, 1)),
    #     spawn=sim_utils.UsdFileCfg(
    #         usd_path="/workspace/isaaclab/source/isaaclab_tasks/isaaclab_tasks/sim/cloth_particle_with_handle.usd",
    #         scale=(1.0, 1.0, 1.0),
    #     ),
    # )

    cloth_with_handles = AssetBaseCfg(
        prim_path="/World/envs/env_.*/cloth",
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.9, 0, 0.1), rot=(1, 0, 0, 0)),
        spawn=sim_utils.UsdFileCfg(
            usd_path="/workspace/isaaclab/source/isaaclab_tasks/isaaclab_tasks/sim/deformable_with_handles.usd",
            scale=(1.0, 1.0, 1.0),
        ),
    )

    handle_1 = RigidObjectCfg(
        prim_path="/World/envs/env_.*/cloth/Cube",
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0, 0, 0), rot=(1, 0, 0, 0)),
        spawn=None,
    )

    handle_2 = RigidObjectCfg(
        prim_path="/World/envs/env_.*/cloth/Cube_01",
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0, 0, 0), rot=(1, 0, 0, 0)),
        spawn=None,
    )

    # cloth = DeformableObjectCfg(
    #     prim_path="/World/envs/env_.*/cuboid/cuboid",
    #     init_state=DeformableObjectCfg.InitialStateCfg(pos=(0.9, 0, 0.1), rot=(0, 0, 0, 1)),
    #     spawn=None,
    # )

    # cube = RigidObjectCfg(
    #     prim_path="/World/envs/env_.*/cuboid/Cube",
    #     init_state=RigidObjectCfg.InitialStateCfg(pos=(0.9, 0, 0.1), rot=(0, 0, 0, 1)),
    #     spawn=None,
    # )

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
                "panda_finger_joint.*": 0.005,
            },
            pos=(0.0, 0.4, 0.0),
            # rot=(0.7071, 0, 0, -0.7071),
            rot=(1.0, 0.0, 0.0, 0.0),
        ),)
        #     spawn=sim_utils.UsdFileCfg(
        #     usd_path="/workspace/isaaclab/source/isaaclab_tasks/isaaclab_tasks/sim/franka_with_plates_flattened.usd",
        #     activate_contact_sensors=True,
        #     rigid_props=sim_utils.RigidBodyPropertiesCfg(
        #         disable_gravity=False,
        #         max_depenetration_velocity=5.0,
        #     ),
        #     articulation_props=sim_utils.ArticulationRootPropertiesCfg(
        #         enabled_self_collisions=False, solver_position_iteration_count=12, solver_velocity_iteration_count=1
        #     ),
        # ))
    elif args_cli.robot == "ur10":
        robot = UR10_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
    else:
        raise ValueError(f"Robot {args_cli.robot} is not supported. Valid: franka_panda, ur10")

def create_sphere_mesh(stage, target_path):
    _, tmp_path = omni.kit.commands.execute("CreateMeshPrim", prim_type="Sphere", select_new_prim=False)
    omni.kit.commands.execute("MovePrim", path_from=tmp_path, path_to=target_path)
    omni.usd.get_context().get_selection().set_selected_prim_paths([], False)
    return UsdGeom.Mesh.Get(stage, target_path)

def add_deformable(stage, default_prim_path, prim_path, x_position):
    scale = 0.5
    deformable_body_material_path = Sdf.Path(default_prim_path + "/deformableMaterial")
    deformableUtils.add_deformable_body_material(stage, deformable_body_material_path, youngs_modulus=50000.0)
    blob_left_mesh = create_sphere_mesh(stage, default_prim_path + prim_path)
    blob_left_mesh.ClearXformOpOrder()
    blob_left_mesh.AddTranslateOp().Set(Gf.Vec3f(x_position, 0.0, 4.0) * scale)
    blob_left_mesh.AddScaleOp(UsdGeom.XformOp.PrecisionDouble).Set(Gf.Vec3d(4.3, 1.0, 1.0) * scale)
    blob_left_mesh.CreateDisplayColorAttr().Set([Gf.Vec3f(0.0, 0.0, 0.5)])

    deformableUtils.add_physx_deformable_body(
        stage, blob_left_mesh.GetPath(), collision_simplification=False, simulation_hexahedral_resolution=15
    )

    physicsUtils.add_physics_material_to_prim(stage, blob_left_mesh.GetPrim(), deformable_body_material_path)

def create_cloth_mesh(stage, target_path, mesh_size, mesh_resolution):
    _, tmp_path = omni.kit.commands.execute("CreateMeshPrim", 
                                            prim_type="Cube", 
                                            select_new_prim=False,
                                            u_patches=mesh_resolution,
                                            v_patches=mesh_resolution,
                                            w_patches=2,
                                            half_scale=mesh_size / 2)
    omni.kit.commands.execute("MovePrim", path_from=tmp_path, path_to=target_path)
    omni.usd.get_context().get_selection().set_selected_prim_paths([], False)
    return UsdGeom.Mesh.Get(stage, target_path)

def create_cloth(stage, env_idx):
    """Creates a high-resolution thin cube to act as a deformable cloth."""
    
    # ✅ Define the deformable object path
    cloth_prim_path = f"/World/envs/env_{env_idx}/cuboid"
    print("***************************************")
    print("Creating cloth at path", cloth_prim_path)
    print("***************************************")

    # Create sphere mesh used as the 'skin mesh' for the deformable body
    mesh_size = 100 # in cm
    mesh_resolution = 50
    skin_mesh = create_cloth_mesh(stage, cloth_prim_path, mesh_size, mesh_resolution)
    skin_mesh.GetPrim().GetAttribute("xformOp:translate").Set(Gf.Vec3f(0.8, 0.0, 0.01))
    skin_mesh.GetPrim().GetAttribute("xformOp:scale").Set(Gf.Vec3f(0.7, 0.7, 0.005))

    skin_mesh.CreateDisplayColorAttr().Set([Gf.Vec3f(0.0, 0.0, 0.5)])

    # Create tet meshes for simulation and collision based on the skin mesh
    simulation_resolution = mesh_resolution

    # Apply PhysxDeformableBodyAPI and PhysxCollisionAPI to skin mesh and set parameter to default values
    _ = deformableUtils.add_physx_deformable_body(
        stage,
        skin_mesh.GetPath(),
        collision_simplification=True,
        simulation_hexahedral_resolution=simulation_resolution,
        self_collision=True,
    )

    # ✅ Set Rest Offset and Contact Offset using PhysxCollisionAPI
    deformable_body_prim = stage.GetPrimAtPath(skin_mesh.GetPath())

    if deformable_body_prim:
        collision_api = PhysxSchema.PhysxCollisionAPI.Apply(deformable_body_prim)

        # ✅ Set the Rest Offset and Contact Offset (THIS WORKS!)
        collision_api.GetRestOffsetAttr().Set(0.0)  # Controls surface penetration before contact
        collision_api.GetContactOffsetAttr().Set(0.001)  # Determines early collision detection

    # Create a deformable body material and set it on the deformable body
    deformable_material_path = omni.usd.get_stage_next_free_path(stage, cloth_prim_path + "/deformableBodyMaterial", True)
    deformableUtils.add_deformable_body_material(
        stage,
        deformable_material_path,
        youngs_modulus=1000000.0,
        poissons_ratio=0.49,
        damping_scale=0.5,
        dynamic_friction=1.5,
        density=10,
    )
    physicsUtils.add_physics_material_to_prim(stage, skin_mesh.GetPrim(), deformable_material_path)

    return cloth_prim_path

def run_simulator(sim: sim_utils.SimulationContext, scene: InteractiveScene):
    """Runs the simulation loop."""

    robot = scene["robot"]

    # cloth_prim_path = create_cloth(scene.stage, 0)
    # cloth_cfg = DeformableObjectCfg(
    #             prim_path=cloth_prim_path, 
    #             spawn=None,
    #             init_state=DeformableObjectCfg.InitialStateCfg(
    #                 pos=(0.4, 0.0, 0.01), rot=(1, 0, 0, 0))
    #             )
    # cloth = DeformableObject(cloth_cfg)
    # cloth = scene["cloth"]
    # handle = scene["cube"]

    # handle = scene["cuboid"]
    # cloth_with_handles = scene["cloth_with_handles"]
    handle_1 = scene["handle_1"]
    handle_2 = scene["handle_2"]

    # Create controller
    diff_ik_cfg = DifferentialIKControllerCfg(command_type="pose", use_relative_mode=False, ik_method="dls")
    diff_ik_controller = DifferentialIKController(diff_ik_cfg, num_envs=scene.num_envs, device=sim.device)

    # Define goals for the arm
    ee_goals = [
        [0.3, -0.1, 0.8, 0.0, 1.0, 0.0, 0.0],
    ]
    ee_goals = torch.tensor(ee_goals, device=sim.device)
    joint_pos_des = torch.zeros((1, 9), device=robot.device)
    joint_pos_des[:,-2:] = 0.04
    # Track the given command
    current_goal_idx = 0
    # Create buffers to store actions
    ik_commands = torch.zeros(scene.num_envs, diff_ik_controller.action_dim, device=robot.device)
    ik_commands[:] = ee_goals[current_goal_idx]

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
    cloth_default_positions = None
    # Simulation loop
    while simulation_app.is_running():
       
        if count % 800 == 0:
            # reset time
            count = 0
            # reset joint state
            joint_pos = robot.data.default_joint_pos.clone()
            joint_vel = robot.data.default_joint_vel.clone()
            robot.write_joint_state_to_sim(joint_pos, joint_vel)
            robot.reset()

            root_pose_w = robot.data.root_state_w[:, 0:7]
            handle_pose = handle_2.data.root_state_w[:, 0:7]
            handle_pos_b, _ = subtract_frame_transforms(
                root_pose_w[:, 0:3], root_pose_w[:, 3:7], handle_pose[:, 0:3], handle_pose[:, 3:7]
            )
            ee_goals[current_goal_idx, 0:3] = handle_pos_b[0]
            ee_goals[current_goal_idx, 2] = 0.12
            ik_commands[:] = ee_goals[current_goal_idx]
            joint_pos_des[:,0:7] = joint_pos[:, robot_entity_cfg.joint_ids].clone()
            joint_pos_des[:,-2:] = 0.04
            # reset controller
            diff_ik_controller.reset()
            diff_ik_controller.set_command(ik_commands)
            # change goal
            current_goal_idx = (current_goal_idx + 1) % len(ee_goals)
            cloth_grasped = 0
        else:
            # obtain quantities from simulation
            jacobian_w = robot.root_physx_view.get_jacobians()[
                :, ee_jacobi_idx, :, robot_entity_cfg.joint_ids]
            base_rot = robot.data.root_quat_w
            jacobian_b = diff_ik_controller.get_jacobian_in_root_frame(jacobian_w, base_rot)
            ee_pose_w = robot.data.body_state_w[:, robot_entity_cfg.body_ids[0], 0:7]
            root_pose_w = robot.data.root_state_w[:, 0:7]
            joint_pos = robot.data.joint_pos[:, robot_entity_cfg.joint_ids]
            # compute frame in root frame
            ee_pos_b, ee_quat_b = subtract_frame_transforms(
                root_pose_w[:, 0:3], root_pose_w[:, 3:7], ee_pose_w[:, 0:3], ee_pose_w[:, 3:7]
            )
            # compute the joint commands
            joint_pos_des[:,0:7] = diff_ik_controller.compute(ee_pos_b, ee_quat_b, jacobian_b, joint_pos)

        # apply actions
        # robot.set_joint_position_target(joint_pos_des, joint_ids=robot_entity_cfg.joint_ids)
        robot.set_joint_position_target(joint_pos_des, joint_ids=torch.arange(0, 9).tolist())
        scene.write_data_to_sim()
        # perform step
        sim.step()
        # update sim-time
        count += 1
        # update buffers
        scene.update(sim_dt)

        # obtain quantities from simulation
        ee_pose_w = robot.data.body_state_w[:, robot_entity_cfg.body_ids[0], 0:7]
        ee_pos_w = ee_pose_w[0,0:3]
        # ee_pose_w = robot.data.body_state_w[:, robot_entity_cfg.body_ids[0], 0:7]

        # if not(cloth_grasped) and torch.norm(ee_pos_w[0:3] - ee_goals[0][0:3]) < 0.01:
        if not(cloth_grasped) and count >= 300:
            gripper_count = 0
            joint_pos_des[:,-2:] = 0.00
            robot.set_joint_position_target(joint_pos_des, joint_ids=torch.arange(0, 9).tolist())
            while(gripper_count < 70):
                scene.write_data_to_sim()
                sim.step()
                count += 1
                gripper_count += 1
                scene.update(sim_dt)
            cloth_grasped = 1
            ee_goals[current_goal_idx][0] = 0.5
            ee_goals[current_goal_idx][1] = 0.0
            ee_goals[current_goal_idx][2] = 0.5
            ik_commands[:] = ee_goals[current_goal_idx]
            diff_ik_controller.reset()
            diff_ik_controller.set_command(ik_commands)

def main():
    """Main function."""
    # Load kit helper
    sim_cfg = sim_utils.SimulationCfg(
        dt=0.01,
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
