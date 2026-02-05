# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.assets import DeformableObjectCfg, ArticulationCfg
from isaaclab.controllers.differential_ik_cfg import DifferentialIKControllerCfg
from isaaclab.envs.mdp.actions.actions_cfg import DifferentialInverseKinematicsActionCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.sim.spawners import UsdFileCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAACLAB_NUCLEUS_DIR

import isaaclab_tasks.manager_based.manipulation.cloth_lift.mdp as mdp

from . import joint_pos_env_cfg

##
# Pre-defined configs
##
from isaaclab_assets.robots.franka import FRANKA_PANDA_HIGH_PD_CFG  # isort: skip


@configclass
class FrankaClothLiftEnvCfg(joint_pos_env_cfg.FrankaClothLiftEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # Set Franka as robot
        # We switch here to a stiffer PD controller for IK tracking to be better.
        self.scene.robot_1 = FRANKA_PANDA_HIGH_PD_CFG.replace(
            prim_path="{ENV_REGEX_NS}/Robot1",
            init_state=ArticulationCfg.InitialStateCfg(
            joint_pos={
                "panda_joint1": 1.157,
                "panda_joint2": -1.066,
                "panda_joint3": -0.155,
                "panda_joint4": -2.239,
                "panda_joint5": -1.841,
                "panda_joint6": 1.003,
                "panda_joint7": 0.469,
                "panda_finger_joint.*": 0.035,
            },
                pos=(-0.50, 0, 0), rot=(1.0, 0, 0, 0)
            )
        )
        self.scene.robot_2 = FRANKA_PANDA_HIGH_PD_CFG.replace(
            prim_path="{ENV_REGEX_NS}/Robot2",
            init_state=ArticulationCfg.InitialStateCfg(
            joint_pos={
                "panda_joint1": 1.157,
                "panda_joint2": -1.066,
                "panda_joint3": -0.155,
                "panda_joint4": -2.239,
                "panda_joint5": -1.841,
                "panda_joint6": 1.003,
                "panda_joint7": 0.469,
                "panda_finger_joint.*": 0.035,
            },
                pos=(1.50, 0, 0), rot=(0, 0, 0, 1.0)
            )
        )

        # Set actions for the specific robot type (franka)
        self.actions.arm_action_1 = DifferentialInverseKinematicsActionCfg(
            asset_name="robot_1",
            joint_names=["panda_joint.*"],
            body_name="panda_hand",
            controller=DifferentialIKControllerCfg(command_type="position", use_relative_mode=False, ik_method="dls"),
            body_offset=DifferentialInverseKinematicsActionCfg.OffsetCfg(pos=[0.0, 0.0, 0.107]),
        )
        self.actions.arm_action_2 = DifferentialInverseKinematicsActionCfg(
            asset_name="robot_2",
            joint_names=["panda_joint.*"],
            body_name="panda_hand",
            controller=DifferentialIKControllerCfg(command_type="position", use_relative_mode=False, ik_method="dls"),
            body_offset=DifferentialInverseKinematicsActionCfg.OffsetCfg(pos=[0.0, 0.0, 0.107]),
        )

##
# Deformable object lift environment.
##


@configclass
class FrankaClothLiftEnvCfg(FrankaClothLiftEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        self.scene.object = DeformableObjectCfg(
            prim_path="{ENV_REGEX_NS}/Object",
            init_state=DeformableObjectCfg.InitialStateCfg(pos=(0.5, 0, 0.05), rot=(0.707, 0, 0, 0.707)),
            spawn=UsdFileCfg(
                usd_path="/workspace/isaaclab/source/isaaclab_tasks/isaaclab_tasks/sim/cloth_deformable.usd",
                scale=(1.0, 1.0, 0.005),
            ),
        )

        self.scene.robot_1.actuators["panda_hand"].effort_limit = 50.0
        self.scene.robot_1.actuators["panda_hand"].stiffness = 40.0
        self.scene.robot_1.actuators["panda_hand"].damping = 10.0
        self.scene.robot_2.actuators["panda_hand"].effort_limit = 50.0
        self.scene.robot_2.actuators["panda_hand"].stiffness = 40.0
        self.scene.robot_2.actuators["panda_hand"].damping = 10.0

        # Disable replicate physics as it doesn't work for deformable objects
        # FIXME: This should be fixed by the PhysX replication system.
        self.scene.replicate_physics = False

        # Set events for the specific object type
        self.events.reset_object_position = EventTerm(
            func=mdp.reset_nodal_state_uniform,
            mode="reset",
            params={
                "position_range": {"x": (0.0, 0.0), "y": (0.0, 0.0), "z": (0.1, 0.1)},
                "velocity_range": {},
                "asset_cfg": SceneEntityCfg("object"),
            },
        )
