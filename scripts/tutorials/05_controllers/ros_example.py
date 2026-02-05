# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
"""
Make sure that you do

export ROS_DISTRO=humble
export RMW_IMPLEMENTATION=rmw_fastrtps_cpp
export isaac_sim_package_path=/isaac-sim
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$isaac_sim_package_path/exts/isaacsim.ros2.bridge/humble/lib
"""
# ros2_joint_control.py
# Copyright ...
"""
Make sure (ROS 2 Humble):
export ROS_DISTRO=humble
export RMW_IMPLEMENTATION=rmw_fastrtps_cpp
"""

import argparse
import torch

from isaaclab.app import AppLauncher

# --- CLI & App ---
parser = argparse.ArgumentParser(description="ROS2 joint control demo (Isaac Lab).")
parser.add_argument("--robot", type=str, default="franka_panda", help="franka_panda or ur10")
parser.add_argument("--num_envs", type=int, default=1)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# Enable ROS 2 bridge
import omni.kit.app as kit
ext_mgr = kit.get_app().get_extension_manager()
ext_mgr.set_extension_enabled("omni.isaac.ros2_bridge", True)

# --- ROS 2 node ---
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import Header

# --- Isaac Lab imports ---
import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg, RigidObjectCfg
from isaaclab.assets import ArticulationCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.managers import SceneEntityCfg
from isaaclab.sim import SimulationCfg, PhysxCfg
from isaaclab.utils import configclass
from isaaclab_assets import FRANKA_PANDA_HIGH_PD_CFG, UR10_CFG


@configclass
class TableTopSceneCfg(InteractiveSceneCfg):
    ground = AssetBaseCfg(
        prim_path="/World/defaultGroundPlane",
        spawn=sim_utils.GroundPlaneCfg(),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, 0)),
    )
    dome_light = AssetBaseCfg(
        prim_path="/World/Light",
        spawn=sim_utils.DomeLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75)),
    )
    # cuboid = RigidObjectCfg(
    #     prim_path="/World/envs/env_.*/cuboid",
    #     spawn=sim_utils.CuboidCfg(
    #         size=(0.05, 0.05, 0.05),
    #         visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.5, 0.0, 0.5), metallic=0.2),
    #         rigid_props=sim_utils.RigidBodyPropertiesCfg(solver_position_iteration_count=4),
    #         mass_props=sim_utils.MassPropertiesCfg(mass=0.5),
    #         collision_props=sim_utils.CollisionPropertiesCfg(),
    #     ),
    #     init_state=RigidObjectCfg.InitialStateCfg(pos=(0.5, 0.0, 0.0)),
    # )

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
                pos=(0.3, 0.1, 0.0),
                rot=(0.7071, 0, 0, -0.7071),
            ),
        )
    elif args_cli.robot == "ur10":
        robot = UR10_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
    else:
        raise ValueError("Supported robots: franka_panda, ur10")


class JointStateBridge(Node):
    """
    Subscribes: /robot/command_joint_state (JointState)
    Publishes:  /joint_states (JointState)
    Keeps a target tensor in Isaac joint order.
    """
    def __init__(self, joint_names_ordered, device):
        super().__init__("isaaclab_joint_state_bridge")
        self.declare_parameter("command_topic", "/robot/command_joint_state")
        self.declare_parameter("state_topic", "/joint_states")

        self.joint_names_ordered = list(joint_names_ordered)
        self.device = device

        cmd_topic = self.get_parameter("command_topic").get_parameter_value().string_value
        state_topic = self.get_parameter("state_topic").get_parameter_value().string_value

        self._target = None  # torch tensor [n_joints] on device
        self._have_cmd = False

        self.sub = self.create_subscription(JointState, cmd_topic, self._cmd_cb, 10)
        self.pub = self.create_publisher(JointState, state_topic, 10)

    def _cmd_cb(self, msg: JointState):
        name_to_idx = {n: i for i, n in enumerate(msg.name)}
        tgt = torch.nan * torch.ones(len(self.joint_names_ordered), device=self.device)
        for j, n in enumerate(self.joint_names_ordered):
            if n in name_to_idx and name_to_idx[n] < len(msg.position):
                tgt[j] = float(msg.position[name_to_idx[n]])
        # Fill any NaNs with previous target later (or hold current)
        self._target = tgt
        self._have_cmd = True

    def get_target(self, fallback: torch.Tensor) -> torch.Tensor:
        """
        Returns a complete target vector (fills NaNs with fallback).
        fallback: current joint positions [1, n] (from sim)
        """
        if not self._have_cmd or self._target is None:
            return fallback[0]
        tgt = self._target.clone()
        # replace NaNs with current (hold those joints)
        nan_mask = torch.isnan(tgt)
        tgt[nan_mask] = fallback[0, nan_mask]
        return tgt

    def publish_state(self, names, pos: torch.Tensor, vel: torch.Tensor = None):
        msg = JointState()
        msg.header = Header()
        msg.name = list(names)
        msg.position = pos.tolist()
        if vel is not None:
            msg.velocity = vel.tolist()
        self.pub.publish(msg)


def run_simulator(sim: sim_utils.SimulationContext, scene: InteractiveScene, ros_node: JointStateBridge):
    robot = scene["robot"]

    # Resolve entity to get joint IDs and names in simulator order
    if args_cli.robot == "franka_panda":
        entity = SceneEntityCfg("robot", joint_names=["panda_joint.*"], body_names=["panda_hand"])
    elif args_cli.robot == "ur10":
        entity = SceneEntityCfg("robot", joint_names=[".*"], body_names=["ee_link"])
    entity.resolve(scene)

    joint_ids = entity.joint_ids  # list of indices
    # Joint names in the same order as joint_ids:
    # joint_names_all = robot.find_joints(".*")[0]  # list of ALL joint names in asset order
    joint_names_ordered = ["panda_joint1",
                           "panda_joint2",
                           "panda_joint3",
                           "panda_joint4",
                           "panda_joint5",
                           "panda_joint6",
                           "panda_joint7",
    ]

    # Re-init ROS node with correct names (if needed)
    ros_node.joint_names_ordered = joint_names_ordered

    sim_dt = sim.get_physics_dt()
    count = 0

    # One-time: open gripper a bit if present (Franka has 9 DoFs including fingers)
    joint_pos_des = robot.data.default_joint_pos.clone()
    if joint_pos_des.shape[1] >= 9:
        joint_pos_des[:, -2:] = 0.04

    while simulation_app.is_running():
        # Pump ROS callbacks
        rclpy.spin_once(ros_node, timeout_sec=0.0)

        # Current joint state (in Isaac order)
        q_now = robot.data.joint_pos[:, joint_ids]     # [1, n]
        dq_now = robot.data.joint_vel[:, joint_ids]    # [1, n]

        # Target from ROS (hold current for missing/NaN joints)
        q_target = ros_node.get_target(q_now)          # [n]
        joint_pos_des[:, joint_ids] = q_target.unsqueeze(0)

        # Apply targets
        robot.set_joint_position_target(joint_pos_des, joint_ids=list(range(joint_pos_des.shape[1])))
        scene.write_data_to_sim()
        sim.step()
        count += 1
        # scene.update(sim_dt)

        # Publish /joint_states
        ros_node.publish_state(joint_names_ordered, q_now[0].cpu(), dq_now[0].cpu())


def main():
    sim_cfg = SimulationCfg(
        dt=0.01,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
        physx=PhysxCfg(
            gpu_max_particle_contacts=2**21,
            gpu_max_soft_body_contacts=2**21,
        ),
        device=args_cli.device,
    )
    sim = sim_utils.SimulationContext(sim_cfg)

    # Camera view is optional
    sim.set_camera_view([2.5, 2.5, 2.5], [0.0, 0.0, 0.0])

    scene_cfg = TableTopSceneCfg(num_envs=args_cli.num_envs, env_spacing=2.0)
    scene = InteractiveScene(scene_cfg)

    # Start the sim
    sim.reset()
    print("[INFO]: Setup complete... waiting for ROS joint commands.")

    # Init ROS 2
    rclpy.init(args=None)
    # Temporary placeholder names; replaced after resolving the robot in run_simulator
    ros_node = JointStateBridge(joint_names_ordered=[], device=sim.device)

    # Run
    run_simulator(sim, scene, ros_node)


if __name__ == "__main__":
    main()
    simulation_app.close()
