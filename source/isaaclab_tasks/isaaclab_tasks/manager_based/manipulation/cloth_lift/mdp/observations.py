# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import RigidObject, DeformableObject, ParticleClothObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import subtract_frame_transforms

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def cloth_position_in_robot_root_frame(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """Samples N equidistant object positions from the deformable body and transforms them into the robot's root frame."""
    robot: RigidObject = env.scene[robot_cfg.name]
    object: ParticleClothObject = env.scene[object_cfg.name]

    N_samples = 10

    # ✅ Get all nodal positions (num_envs, num_nodes, 3)
    import ipdb; ipdb.set_trace()
    cloth_positions_w = object.data.nodal_pos_w  # (N_envs, N_points, 3)

    # ✅ Sample 100 points **evenly spaced** across the mesh
    num_points = cloth_positions_w.shape[1]  # Total number of cloth points
    sample_indices = torch.linspace(0, num_points - 1, N_samples, dtype=torch.long, device=cloth_positions_w.device)
    sampled_positions_w = cloth_positions_w[:, sample_indices, :]  # (N_envs, N_samples, 3)

    # ✅ Convert sampled points to the robot’s root frame
    object_pos_b, _ = subtract_frame_transforms(
        robot.data.root_state_w[:, :3].unsqueeze(1).repeat(1,N_samples,1),  # (N_envs, 1, 3)
        robot.data.root_state_w[:, 3:7].unsqueeze(1).repeat(1,N_samples,1),  # (N_envs, 1, 4)
        sampled_positions_w  # (N_envs, N_samples, 3)
    )

    return object_pos_b.view(env.num_envs, -1) 


def object_position_in_robot_root_frame(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """Samples N equidistant object positions from the deformable body and transforms them into the robot's root frame."""
    robot: RigidObject = env.scene[robot_cfg.name]
    object: DeformableObject = env.scene[object_cfg.name]

    N_samples = 10

    # ✅ Get all nodal positions (num_envs, num_nodes, 3)
    cloth_positions_w = object.data.nodal_pos_w  # (N_envs, N_points, 3)

    # ✅ Sample 100 points **evenly spaced** across the mesh
    num_points = cloth_positions_w.shape[1]  # Total number of cloth points
    sample_indices = torch.linspace(0, num_points - 1, N_samples, dtype=torch.long, device=cloth_positions_w.device)
    sampled_positions_w = cloth_positions_w[:, sample_indices, :]  # (N_envs, N_samples, 3)

    # ✅ Convert sampled points to the robot’s root frame
    object_pos_b, _ = subtract_frame_transforms(
        robot.data.root_state_w[:, :3].unsqueeze(1).repeat(1,N_samples,1),  # (N_envs, 1, 3)
        robot.data.root_state_w[:, 3:7].unsqueeze(1).repeat(1,N_samples,1),  # (N_envs, 1, 4)
        sampled_positions_w  # (N_envs, N_samples, 3)
    )

    return object_pos_b.view(env.num_envs, -1) 

