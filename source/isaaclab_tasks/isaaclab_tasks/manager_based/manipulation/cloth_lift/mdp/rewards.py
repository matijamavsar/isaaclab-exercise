# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import RigidObject, DeformableObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import FrameTransformer
from isaaclab.utils.math import combine_frame_transforms

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def object_mean_height(
    env: ManagerBasedRLEnv, object_cfg: SceneEntityCfg = SceneEntityCfg("object")
) -> torch.Tensor:
    """Reward the agent for lifting the deformable object (cloth)."""
    object: DeformableObject = env.scene[object_cfg.name]
    
    # ✅ Get all nodal positions (num_envs, num_nodes, 3)
    cloth_positions = object.data.nodal_pos_w  # (N_envs, N_points, 3)
    
    # ✅ Compute the mean height of all cloth points
    mean_cloth_height = cloth_positions[:, :, 2].mean(dim=1)  # (num_envs,)
    return mean_cloth_height


def object_ee_distance(
    env: ManagerBasedRLEnv,
    std: float = 1.0,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
) -> torch.Tensor:
    """Reward the agent for reaching the closest point on the object using tanh-kernel."""
    # extract the used quantities (to enable type-hinting)
    object: DeformableObject = env.scene[object_cfg.name]
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]

    # ✅ Get all nodal positions (num_envs, num_nodes, 3)
    cloth_positions_w = object.data.nodal_pos_w  # (N_envs, N_points, 3)

    # ✅ End-effector position (num_envs, 3)
    ee_w = ee_frame.data.target_pos_w[..., 0, :]  # (N_envs, 3)

    # ✅ Compute distances to all cloth points
    ee_cloth_distances = torch.norm(cloth_positions_w - ee_w.unsqueeze(1), dim=-1)  # (N_envs, N_points)

    # ✅ Take the minimum distance to any cloth point
    min_distance = torch.min(ee_cloth_distances, dim=1)[0]  # (N_envs,)

    return 1 - torch.tanh(min_distance / std)



