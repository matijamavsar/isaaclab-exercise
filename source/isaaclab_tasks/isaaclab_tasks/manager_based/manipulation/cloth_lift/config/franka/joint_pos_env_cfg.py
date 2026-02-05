# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.assets import RigidObjectCfg, ArticulationCfg
from isaaclab.sensors import FrameTransformerCfg
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import OffsetCfg
from isaaclab.sim.schemas.schemas_cfg import RigidBodyPropertiesCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import UsdFileCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

from isaaclab_tasks.manager_based.manipulation.lift import mdp
from isaaclab_tasks.manager_based.manipulation.cloth_lift.lift_env_cfg import ClothLiftEnvCfg

##
# Pre-defined configs
##
from isaaclab.markers.config import FRAME_MARKER_CFG  # isort: skip
from isaaclab_assets.robots.franka import FRANKA_PANDA_CFG  # isort: skip


@configclass
class FrankaClothLiftEnvCfg(ClothLiftEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # Set Franka as robot
        self.scene.robot_1 = FRANKA_PANDA_CFG.replace(
            prim_path="{ENV_REGEX_NS}/Robot1",
            init_state=ArticulationCfg.InitialStateCfg(
                pos=(-0.75, 0, 0), rot=(1.0, 0, 0, 0)
            )
        )
        self.scene.robot_2 = FRANKA_PANDA_CFG.replace(
            prim_path="{ENV_REGEX_NS}/Robot2",
            init_state=ArticulationCfg.InitialStateCfg(
                pos=(0.75, 0, 0), rot=(0, 0, 0, 1.0)
            )
        )

        # Set actions for the specific robot type (franka)
        self.actions.arm_action_1 = mdp.JointPositionActionCfg(
            asset_name="robot_1", joint_names=["panda_joint.*"], scale=0.5, use_default_offset=True
        )
        self.actions.arm_action_2 = mdp.JointPositionActionCfg(
            asset_name="robot_2", joint_names=["panda_joint.*"], scale=0.5, use_default_offset=True
        )
        self.actions.gripper_action_1 = mdp.BinaryJointPositionActionCfg(
            asset_name="robot_1",
            joint_names=["panda_finger.*"],
            open_command_expr={"panda_finger_.*": 0.04},
            close_command_expr={"panda_finger_.*": 0.0},
        )
        self.actions.gripper_action_2 = mdp.BinaryJointPositionActionCfg(
            asset_name="robot_2",
            joint_names=["panda_finger.*"],
            open_command_expr={"panda_finger_.*": 0.04},
            close_command_expr={"panda_finger_.*": 0.0},
        )
        # Set the body name for the end effector
        self.commands.object_pose_1.body_name = "panda_hand"
        self.commands.object_pose_2.body_name = "panda_hand"

        # Listens to the required transforms
        self.scene.ee_frame_1 = FrameTransformerCfg(
            prim_path="{ENV_REGEX_NS}/Robot1/panda_link0",
            debug_vis=False,
            target_frames=[
                FrameTransformerCfg.FrameCfg(
                    prim_path="{ENV_REGEX_NS}/Robot1/panda_hand",
                    name="end_effector_1",
                    offset=OffsetCfg(
                        pos=[0.0, 0.0, 0.1034],
                    ),
                ),
            ],
        )
        self.scene.ee_frame_2 = FrameTransformerCfg(
            prim_path="{ENV_REGEX_NS}/Robot2/panda_link0",
            debug_vis=False,
            target_frames=[
                FrameTransformerCfg.FrameCfg(
                    prim_path="{ENV_REGEX_NS}/Robot2/panda_hand",
                    name="end_effector_2",
                    offset=OffsetCfg(
                        pos=[0.0, 0.0, 0.1034],
                    ),
                ),
            ],
        )


@configclass
class FrankaClothLiftEnvCfg_PLAY(FrankaClothLiftEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()
        # make a smaller scene for play
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        # disable randomization for play
        self.observations.policy.enable_corruption = False
