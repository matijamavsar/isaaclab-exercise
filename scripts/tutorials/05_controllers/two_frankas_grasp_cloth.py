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
from isaaclab.utils.math import quat_from_euler_xyz

from omni.physx.scripts import physicsUtils, deformableUtils

##
# Pre-defined configs
##
from isaaclab_assets import FRANKA_PANDA_HIGH_PD_CFG, UR10_CFG  # isort:skip
from isaaclab.assets import ArticulationCfg
from isaaclab.actuators.actuator_cfg import ImplicitActuatorCfg
from isaaclab.assets import ParticleClothObject, ParticleClothObjectCfg

import os
import cv2
from isaaclab.sensors import CameraCfg

@configclass
class TableTopSceneCfg(InteractiveSceneCfg):
    """Configuration for a cart-pole scene."""

    # ground plane
    ground = AssetBaseCfg(
        prim_path="/World/defaultGroundPlane",
        spawn=sim_utils.GroundPlaneCfg(),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, 0)),
    )

    camera = CameraCfg(
        prim_path="/World/Camera",
        offset=CameraCfg.OffsetCfg(
            pos=(2.8, 2.8, 2.0),
            rot=(0.5, 0.0, 0.0, 0.5),  # looking down at the scene
            convention="world",
        ),
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=24.0,
            focus_distance=4.0,
            horizontal_aperture=20.955,
            clipping_range=(0.1, 20.0),
        ),
        width=1280,
        height=720,
    )

    # lights
    dome_light = AssetBaseCfg(
        prim_path="/World/Light", spawn=sim_utils.DomeLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75))
    )

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

    # articulation
    if args_cli.robot == "franka_panda":
        robot_1 = FRANKA_PANDA_HIGH_PD_CFG.replace(
            prim_path="{ENV_REGEX_NS}/Robot1",
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
            pos=(-0.1, 0.4, 0.0),
            # rot=(0.7071, 0, 0, -0.7071),
            rot=(1.0, 0.0, 0.0, 0.0),
        ),)

        robot_2 = FRANKA_PANDA_HIGH_PD_CFG.replace(
            prim_path="{ENV_REGEX_NS}/Robot2",
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
            pos=(1.7, 0.4, 0.0),
            # rot=(0.7071, 0, 0, -0.7071),
            rot=(0.0, 0.0, 0.0, 1.0),
        ),)
    elif args_cli.robot == "ur10":
        robot = UR10_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
    else:
        raise ValueError(f"Robot {args_cli.robot} is not supported. Valid: franka_panda, ur10")

def run_simulator(sim: sim_utils.SimulationContext, scene: InteractiveScene):
    """Runs the simulation loop."""

    # ------------------------------------------------------------
    # GET SCENE OBJECTS
    # ------------------------------------------------------------
    robot_1 = scene["robot_1"]
    robot_2 = scene["robot_2"]

    handle_1 = scene["handle_1"]
    handle_2 = scene["handle_2"]

    camera = scene["camera"]

    device = sim.device

    camera.set_world_poses_from_view(torch.tensor([2.6, 1.6, 2.3], device=device).unsqueeze(0),
                                     torch.tensor([0.9, 0.0, 0.1], device=device))

    # ------------------------------------------------------------
    # IK CONTROLLERS
    # ------------------------------------------------------------
    diff_ik_cfg = DifferentialIKControllerCfg(
        command_type="pose",
        use_relative_mode=False,
        ik_method="pinv",
    )

    ik_1 = DifferentialIKController(diff_ik_cfg, num_envs=scene.num_envs, device=device)
    ik_2 = DifferentialIKController(diff_ik_cfg, num_envs=scene.num_envs, device=device)

    # ------------------------------------------------------------
    # ROBOT ENTITY CONFIGS
    # ------------------------------------------------------------
    robot1_entity = SceneEntityCfg(
        "robot_1",
        joint_names=["panda_joint.*"],
        body_names=["panda_hand"],
    )
    robot2_entity = SceneEntityCfg(
        "robot_2",
        joint_names=["panda_joint.*"],
        body_names=["panda_hand"],
    )

    robot1_entity.resolve(scene)
    robot2_entity.resolve(scene)

    # ------------------------------------------------------------
    # EE JACOBIAN INDICES
    # ------------------------------------------------------------
    ee_jacobi_idx_1 = (
        robot1_entity.body_ids[0] - 1
        if robot_1.is_fixed_base
        else robot1_entity.body_ids[0]
    )
    ee_jacobi_idx_2 = (
        robot2_entity.body_ids[0] - 1
        if robot_2.is_fixed_base
        else robot2_entity.body_ids[0]
    )

    # ------------------------------------------------------------
    # JOINT TARGET BUFFERS
    # ------------------------------------------------------------
    joint_pos_des_1 = torch.zeros(
        (scene.num_envs, robot_1.num_joints),
        device=device,
    )
    joint_pos_des_2 = torch.zeros(
        (scene.num_envs, robot_2.num_joints),
        device=device,
    )

    # open grippers initially
    joint_pos_des_1[:, -2:] = 0.04
    joint_pos_des_2[:, -2:] = 0.04

    # ------------------------------------------------------------
    # SIMULATION STATE
    # ------------------------------------------------------------
    sim_dt = sim.get_physics_dt()
    count = 0
    cloth_grasped = 0

    # ------------------------------------------------------------
    # CAMERA + VIDEO WRITER
    # ------------------------------------------------------------
    camera = scene["camera"]

    os.makedirs("/workspace/isaaclab/logs/output", exist_ok=True)

    video_writer = None
    video_idx = 0
    segment_length = 800
    global_step = 0

    # Simulation loop
    while simulation_app.is_running():

        if global_step % segment_length == 0:
            # close previous segment
            if video_writer is not None:
                video_writer.release()

            # open new segment
            camera.update(sim_dt)
            img = camera.data.output["rgb"][0].cpu().numpy()
            h, w, _ = img.shape
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")

            video_path = f"/workspace/logs/output/dual_arm_cloth_{video_idx:03d}.mp4"
            print(f"[INFO] Recording {video_path}")

            video_writer = cv2.VideoWriter(video_path, fourcc, 30, (w, h))
            video_idx += 1

        # ------------------------------------------------------------
        # RESET PHASE
        # ------------------------------------------------------------
        if count % 800 == 0:
            count = 0

            # --- reset Robot 1 ---
            joint_pos_1 = robot_1.data.default_joint_pos.clone()
            joint_vel_1 = robot_1.data.default_joint_vel.clone()
            robot_1.write_joint_state_to_sim(joint_pos_1, joint_vel_1)
            robot_1.reset()

            # --- reset Robot 2 ---
            joint_pos_2 = robot_2.data.default_joint_pos.clone()
            joint_vel_2 = robot_2.data.default_joint_vel.clone()
            robot_2.write_joint_state_to_sim(joint_pos_2, joint_vel_2)
            robot_2.reset()

            # --- compute handle targets in each robot root frame ---
            root1_pose_w = robot_1.data.root_state_w[:, 0:7]
            root2_pose_w = robot_2.data.root_state_w[:, 0:7]

            h1_pose_w = handle_1.data.root_state_w[:, 0:7]
            h2_pose_w = handle_2.data.root_state_w[:, 0:7]

            # Robot 1 → handle_2
            h2_pos_b1, _ = subtract_frame_transforms(
                root1_pose_w[:, 0:3], root1_pose_w[:, 3:7],
                h2_pose_w[:, 0:3], h2_pose_w[:, 3:7],
            )

            # Robot 2 → handle_1
            h1_pos_b2, _ = subtract_frame_transforms(
                root2_pose_w[:, 0:3], root2_pose_w[:, 3:7],
                h1_pose_w[:, 0:3], h1_pose_w[:, 3:7],
            )

            # --- IK commands (pose in root frame) ---
            cmd_1 = torch.zeros((scene.num_envs, 7), device=sim.device)
            cmd_2 = torch.zeros((scene.num_envs, 7), device=sim.device)

            cmd_1[:, 0:3] = h2_pos_b1
            cmd_2[:, 0:3] = h1_pos_b2
            cmd_1[:, 2] = 0.12
            cmd_2[:, 2] = 0.12

            # neutral EE orientation (tweak if needed)
            cmd_1[:, 3:] = torch.tensor([0.0, 1.0, 0.0, 0.0], device=sim.device)
            cmd_2[:, 3:] = torch.tensor([0.0, 1.0, 0.0, 0.0], device=sim.device)

            # --- reset IK controllers ---
            ik_1.reset()
            ik_2.reset()
            ik_1.set_command(cmd_1)
            ik_2.set_command(cmd_2)

            # --- initialize joint targets ---
            joint_pos_des_1[:, 0:7] = joint_pos_1[:, robot1_entity.joint_ids]
            joint_pos_des_2[:, 0:7] = joint_pos_2[:, robot2_entity.joint_ids]
            joint_pos_des_1[:, -2:] = 0.04
            joint_pos_des_2[:, -2:] = 0.04

            cloth_grasped = 0

        # ------------------------------------------------------------
        # IK TRACKING PHASE
        # ------------------------------------------------------------
        else:
            # ================= Robot 1 =================
            jac_w_1 = robot_1.root_physx_view.get_jacobians()[
                :, ee_jacobi_idx_1, :, robot1_entity.joint_ids
            ]
            jac_b_1 = ik_1.get_jacobian_in_root_frame(jac_w_1, robot_1.data.root_quat_w)

            ee_pose_w_1 = robot_1.data.body_state_w[:, robot1_entity.body_ids[0], 0:7]
            root_pose_w_1 = robot_1.data.root_state_w[:, 0:7]
            joint_pos_1 = robot_1.data.joint_pos[:, robot1_entity.joint_ids]

            ee_pos_b_1, ee_quat_b_1 = subtract_frame_transforms(
                root_pose_w_1[:, 0:3], root_pose_w_1[:, 3:7],
                ee_pose_w_1[:, 0:3], ee_pose_w_1[:, 3:7],
            )

            joint_pos_des_1[:, 0:7] = ik_1.compute(
                ee_pos_b_1, ee_quat_b_1, jac_b_1, joint_pos_1
            )

            # ================= Robot 2 =================
            jac_w_2 = robot_2.root_physx_view.get_jacobians()[
                :, ee_jacobi_idx_2, :, robot2_entity.joint_ids
            ]
            jac_b_2 = ik_2.get_jacobian_in_root_frame(jac_w_2, robot_2.data.root_quat_w)

            ee_pose_w_2 = robot_2.data.body_state_w[:, robot2_entity.body_ids[0], 0:7]
            root_pose_w_2 = robot_2.data.root_state_w[:, 0:7]
            joint_pos_2 = robot_2.data.joint_pos[:, robot2_entity.joint_ids]

            ee_pos_b_2, ee_quat_b_2 = subtract_frame_transforms(
                root_pose_w_2[:, 0:3], root_pose_w_2[:, 3:7],
                ee_pose_w_2[:, 0:3], ee_pose_w_2[:, 3:7],
            )

            joint_pos_des_2[:, 0:7] = ik_2.compute(
                ee_pos_b_2, ee_quat_b_2, jac_b_2, joint_pos_2
            )

        # ------------------------------------------------------------
        # APPLY ACTIONS
        # ------------------------------------------------------------
        robot_1.set_joint_position_target(joint_pos_des_1, joint_ids=torch.arange(0, 9).tolist())
        robot_2.set_joint_position_target(joint_pos_des_2, joint_ids=torch.arange(0, 9).tolist())

        scene.write_data_to_sim()
        sim.step()
        count += 1
        scene.update(sim_dt)

        camera.update(sim_dt)
        img_rgb = camera.data.output["rgb"][0].cpu().numpy()
        img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
        video_writer.write(img_bgr)

        # ------------------------------------------------------------
        # SIMULTANEOUS GRASP
        # ------------------------------------------------------------
        if not cloth_grasped and count >= 300:
            gripper_count = 0

            joint_pos_des_1[:, -2:] = 0.0
            joint_pos_des_2[:, -2:] = 0.0

            robot_1.set_joint_position_target(joint_pos_des_1, joint_ids=torch.arange(0, 9).tolist())
            robot_2.set_joint_position_target(joint_pos_des_2, joint_ids=torch.arange(0, 9).tolist())

            while gripper_count < 70:
                scene.write_data_to_sim()
                sim.step()
                count += 1
                gripper_count += 1
                scene.update(sim_dt)

                camera.update(sim_dt)
                img_rgb = camera.data.output["rgb"][0].cpu().numpy()
                img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
                video_writer.write(img_bgr)

            cloth_grasped = 1
            cmd_1[:,2] = 0.6
            cmd_2[:,2] = 0.6
            ik_1.reset()
            ik_2.reset()
            ik_1.set_command(cmd_1)
            ik_2.set_command(cmd_2)

        global_step += 1

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
