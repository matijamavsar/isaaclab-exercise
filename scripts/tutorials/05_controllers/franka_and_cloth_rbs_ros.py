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
import time

##
# Pre-defined configs
##
from isaaclab_assets import FRANKA_PANDA_HIGH_PD_CFG, UR10_CFG  # isort:skip
from isaaclab.assets import ArticulationCfg
from isaaclab.actuators.actuator_cfg import ImplicitActuatorCfg
from isaaclab.assets import ParticleClothObject, ParticleClothObjectCfg

# ROS 2
import rclpy
from rclpy.executors import SingleThreadedExecutor
from sensor_msgs.msg import JointState
from std_msgs.msg import Header
from trajectory_msgs.msg import JointTrajectory
from rclpy.action import ActionServer, GoalResponse, CancelResponse
from control_msgs.action import FollowJointTrajectory

from robotblockset.ros2.franka_ros2 import franka_ros2

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
                "panda_finger_joint.*": 0.005,
            },
            pos=(0.0, 0.0, 0.0),
            # rot=(0.7071, 0, 0, -0.7071),
            rot=(1.0, 0.0, 0.0, 0.0),
        ),)
    elif args_cli.robot == "ur10":
        robot = UR10_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
    else:
        raise ValueError(f"Robot {args_cli.robot} is not supported. Valid: franka_panda, ur10")

# -------------------- GLOBAL ROS STATE (no "self") --------------------
_ros_node = None
_ros_exec = None
_ros_state_pub = {}      # env_i -> JointState publisher
_ros_cmd_sub = {}        # env_i -> JointTrajectory subscriber
_ros_joint_names = []    # list[str]
_ros_desired_q = None    # torch.Tensor [num_envs, dof]
_ros_desired_t = None    # list[rclpy.time.Time | None]
_use_ros_cmd = True
_traj_queue = {}      # env_i -> list[torch.Tensor (dof,)]
_traj_active = {}     # env_i -> torch.Tensor | None (current target)
_settle_counts = {}   # env_i -> int (consecutive steps within tol)
_action_servers = {}     # env_i -> ActionServer
_goal_active = {}        # env_i -> goal_handle or None
_goal_remaining = {}     # env_i -> int remaining waypoints for the active goal

def ros_spin_once():
    if _ros_exec is not None:
        _ros_exec.spin_once(timeout_sec=0.0)

def js_msg(names, positions, velocities=None):
    msg = JointState()
    msg.header = Header()
    msg.header.stamp = _ros_node.get_clock().now().to_msg()
    msg.name = names
    msg.position = [float(x) for x in positions]
    if velocities is not None:
        msg.velocity = [float(x) for x in velocities]
    return msg

# -------------------- MAIN SIM LOOP (single robot, no "self") --------------------
def run_simulator(sim: sim_utils.SimulationContext, scene: InteractiveScene):
    global _ros_node, _ros_exec, _ros_state_pub, _ros_cmd_sub
    global _ros_joint_names, _ros_desired_q, _ros_desired_t, _use_ros_cmd

    robot = scene["robot"]
    num_envs = scene.num_envs
    device = robot.device

    # ----- ROS2 setup -----
    rclpy.init(args=None)
    _ros_node = rclpy.create_node("isaaclab_joint_io")
    _ros_exec = SingleThreadedExecutor()
    _ros_exec.add_node(_ros_node)

    # Resolve joint names (fallback for Franka)
    _default_franka = [
        "panda_joint1","panda_joint2","panda_joint3","panda_joint4",
        "panda_joint5","panda_joint6","panda_joint7",
        "panda_finger_joint1","panda_finger_joint2"
    ]
    names = getattr(robot.data, "joint_names", None)
    _ros_joint_names = [str(n) for n in names] if names and len(names) else _default_franka

    dof = robot.num_joints
    _ros_desired_q = torch.zeros(num_envs, dof, device=device)
    _ros_desired_t = [None for _ in range(num_envs)]
    _ros_state_pub = {}
    _ros_cmd_sub = {}

    for env_i in range(num_envs):
        _traj_queue[env_i] = []
        _traj_active[env_i] = None
        _settle_counts[env_i] = 0
        _goal_active[env_i] = None
        _goal_remaining[env_i] = 0

    ns_robot = "robot1"

    # Subscriber callback (per env)
    def make_cb(env_i: int):
        def _cb(msg: JointTrajectory):
            nonlocal robot
            for pt in msg.points:
                q = robot.data.joint_pos[env_i].clone()      # [9]
                q[:7] = torch.tensor(pt.positions[:7], device=robot.device)
                _traj_queue[env_i].append(q)                 # no checks, just push
            print(f"[env {env_i}] queued {len(msg.points)} targets")
        return _cb

    # Create pubs/subs per env
    for env_i in range(num_envs):
        base_ns    = f"/env_{env_i}/{ns_robot}"
        state_topic = f"{base_ns}/joint_states"
        cmd_topic   = f"{base_ns}/joint_trajectory_command"

        _ros_state_pub[env_i] = _ros_node.create_publisher(JointState, state_topic, 10)
        _ros_cmd_sub[env_i]   = _ros_node.create_subscription(JointTrajectory, cmd_topic, make_cb(env_i), 10)

    def make_goal_cb(env_i: int):
        def _goal_cb(goal_request: FollowJointTrajectory.Goal):
            # Accept everything (simple server)
            return GoalResponse.ACCEPT
        return _goal_cb

    def make_cancel_cb(env_i: int):
        def _cancel_cb(goal_handle):
            # Clear queue for this env and drop the active goal
            _traj_queue[env_i].clear()
            _goal_remaining[env_i] = 0
            _goal_active[env_i] = None
            return CancelResponse.ACCEPT
        return _cancel_cb

    def make_execute_cb(env_i: int):
        def _execute_cb(goal_handle):
            nonlocal robot
            traj = goal_handle.request.trajectory  # trajectory_msgs/JointTrajectory

            # Enqueue all points (first 7 joints overwrite arm; last 2 (fingers) unchanged)
            for pt in traj.points:
                q = robot.data.joint_pos[env_i].clone()   # [9]
                q[:7] = torch.tensor(pt.positions[:7], device=robot.device)
                _traj_queue[env_i].append(q)

            # Register as active goal and wait until the queue points from this goal are sent
            _goal_active[env_i] = goal_handle
            _goal_remaining[env_i] += len(traj.points)

            result = FollowJointTrajectory.Result()
            # Wait until the sim loop drains the enqueued points or cancellation happens
            # while rclpy.ok():
            #     if goal_handle.is_cancel_requested:
            #         _traj_queue[env_i].clear()
            #         _goal_remaining[env_i] = 0
            #         _goal_active[env_i] = None
            #         goal_handle.canceled()
            #         result.error_code = FollowJointTrajectory.Result.INVALID_GOAL
            #         result.error_string = "Canceled"
            #         return result
            #     if _goal_remaining[env_i] <= 0 and _goal_active[env_i] is goal_handle:
            #         break
                # time.sleep(0.01)  # non-busy wait

            # Mark success
            if _goal_active[env_i] is goal_handle:
                _goal_active[env_i] = None
            goal_handle.succeed()
            result.error_code = FollowJointTrajectory.Result.SUCCESSFUL
            result.error_string = ""
            return result
        return _execute_cb
    
        # Action server endpoint (choose the name you want your client to use)
    action_name = f"{base_ns}/follow_joint_trajectory"
    _action_servers[env_i] = ActionServer(
        _ros_node,
        FollowJointTrajectory,
        action_name,
        execute_callback=make_execute_cb(env_i),
        goal_callback=make_goal_cb(env_i),
        cancel_callback=make_cancel_cb(env_i),
    )

    # Publish joint_states at 30 Hz
    def publish_joint_states():
        # robot.data.joint_pos: [num_envs, dof]
        names = _ros_joint_names
        q_all = robot.data.joint_pos[:,0:7].detach().cpu().numpy()
        v_all = robot.data.joint_vel[:,0:7].detach().cpu().numpy()
        dof = q_all.shape[1]
        for env_i in range(num_envs):
            q = q_all[env_i].tolist()
            v = v_all[env_i].tolist()
            # match lengths just in case
            # if len(q) < len(names): q += [0.0] * (len(names) - len(q))
            # if len(q) > len(names): q = q[:len(names)]
            pub = _ros_state_pub[env_i]
            pub.publish(js_msg(names, q, v))
            ros_spin_once()
    # _ros_node.create_timer(1.0/100.0, publish_joint_states)

    # robot_rbs = franka_ros2(ns='env_0/robot1')
    # robot_rbs.GetState()
    # print(robot_rbs.q)

    cloth = scene["cloth"]

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
    # robot_entity_cfg.resolve(scene)
    # Obtain the frame index of the end-effector
    # For a fixed base robot, the frame index is one less than the body index. This is because
    # the root body is not included in the returned Jacobians.

    # Define simulation stepping
    sim_dt = sim.get_physics_dt()
    count = 0
    q = None
    # Simulation loop
    while simulation_app.is_running():
       
        publish_joint_states()

        # ---- Queue executor (per env) ----
        for ei in range(num_envs):
            if _traj_queue[ei]:
                q = _traj_queue[ei].pop(0).unsqueeze(0)  # [1,9]
                env_tensor = torch.tensor([ei], device=robot.device)
                robot.write_joint_position_to_sim(q, env_ids=env_tensor)
                robot.set_joint_position_target(q, env_ids=env_tensor)

                # If an action goal is active for this env, count this point as sent
                if _goal_active[ei] is not None and _goal_remaining[ei] > 0:
                    _goal_remaining[ei] -= 1

            elif q is not None:
                env_tensor = torch.tensor([ei], device=robot.device)
                robot.write_joint_position_to_sim(q, env_ids=env_tensor)
                robot.set_joint_position_target(q, env_ids=env_tensor)

        sim.step()
        # update sim-time
        count += 1
        # update buffers
        scene.update(sim_dt)


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
