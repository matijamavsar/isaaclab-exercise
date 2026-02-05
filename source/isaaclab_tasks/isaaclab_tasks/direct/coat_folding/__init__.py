# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
"""
Franka-Cloth environment.
"""

import gymnasium as gym

from . import agents

##
# Register Gym environments.
##

gym.register(
    id="DMP-Based-Coat-Fold-Init-Motion",
    entry_point=f"{__name__}.dmp_based_coat_fold_init_motion:FrankaDMPCoatFoldEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.dmp_based_coat_fold_init_motion:FrankaDMPCoatFoldEnvCfg",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg_decimated:FrankaCoatFoldPPORunnerCfg",
    },
)

gym.register(
    id="DMP-Based-Deformable-Coat-Fold-Init-Motion",
    entry_point=f"{__name__}.dmp_based_deformable_coat_fold_init_motion:FrankaDMPCoatFoldEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.dmp_based_deformable_coat_fold_init_motion:FrankaDMPCoatFoldEnvCfg",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg_decimated:FrankaCoatFoldPPORunnerCfg",
    },
)