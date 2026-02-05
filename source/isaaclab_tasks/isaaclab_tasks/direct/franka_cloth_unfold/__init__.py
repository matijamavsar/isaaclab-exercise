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
    id="DMP-Based-Multi-Update-Unfold-Curobo",
    entry_point=f"{__name__}.dmp_based_multi_update_unfold_curobo:FrankaDMPClothPlaceEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.dmp_based_multi_update_unfold_curobo:FrankaDMPClothPlaceEnvCfg",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg_multiUpdate:FrankaClothPPORunnerCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_sac_cfg.yaml",
    },
)

gym.register(
    id="DMP-Based-Multi-Update-Unfold-ROS",
    entry_point=f"{__name__}.dmp_based_multi_update_unfold_ros:FrankaDMPClothPlaceEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.dmp_based_multi_update_unfold_ros:FrankaDMPClothPlaceEnvCfg",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg_multiUpdate:FrankaClothPPORunnerCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_sac_cfg.yaml",
    },
)

gym.register(
    id="DMP-Based-Multi-Update-Unfold-RBS",
    entry_point=f"{__name__}.dmp_based_multi_update_unfold_rbs:FrankaDMPClothPlaceEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.dmp_based_multi_update_unfold_rbs:FrankaDMPClothPlaceEnvCfg",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg_multiUpdate:FrankaClothPPORunnerCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_sac_cfg.yaml",
    },
)

gym.register(
    id="Refactored-One-Robot-RBS",
    entry_point=f"{__name__}.refactored_one_robot_rbs:FrankaDMPClothPlaceEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.refactored_one_robot_rbs:FrankaDMPClothPlaceEnvCfg",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg_multiUpdate:FrankaClothPPORunnerCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_sac_cfg.yaml",
    },
)

gym.register(
    id="Refactored-One-Robot-Curobo",
    entry_point=f"{__name__}.refactored_one_robot_curobo:FrankaDMPClothPlaceEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.refactored_one_robot_curobo:FrankaDMPClothPlaceEnvCfg",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg_multiUpdate:FrankaClothPPORunnerCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_sac_cfg.yaml",
    },
)

gym.register(
    id="Refactored-Only-Cloth",
    entry_point=f"{__name__}.refactored_only_cloth:FrankaDMPClothPlaceEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.refactored_only_cloth:FrankaDMPClothPlaceEnvCfg",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg_decimated:FrankaClothPPORunnerCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
    },
)

gym.register(
    id="Refactored-Only-Cloth-Resnet",
    entry_point=f"{__name__}.refactored_only_cloth_resnet:FrankaDMPClothPlaceEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.refactored_only_cloth_resnet:FrankaDMPClothPlaceEnvCfg",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg_decimated:FrankaClothPPORunnerCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
    },
)

gym.register(
    id="Refactored-Only-Cloth-CNN-PPO",
    entry_point=f"{__name__}.refactored_only_cloth_cnn_ppo:FrankaDMPClothPlaceEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.refactored_only_cloth_cnn_ppo:FrankaDMPClothPlaceEnvCfg",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg_decimated_cnn:FrankaClothPPORunnerCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
    },
)

gym.register(
    id="Refactored-Only-Cloth-CNN-PPO-State",
    entry_point=f"{__name__}.refactored_only_cloth_cnn_ppo:FrankaDMPClothPlaceEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.refactored_only_cloth_cnn_ppo:FrankaDMPClothPlaceEnvCfg",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg_decimated:FrankaClothPPORunnerCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
    },
)

gym.register(
    id="Refactored-Only-Cloth-Resnet-USD",
    entry_point=f"{__name__}.refactored_only_cloth_resnet_usd:FrankaDMPClothPlaceEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.refactored_only_cloth_resnet_usd:FrankaDMPClothPlaceEnvCfg",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg_decimated:FrankaClothPPORunnerCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
    },
)

gym.register(
    id="Refactored-Only-Cloth-Resnet-Linear",
    entry_point=f"{__name__}.refactored_only_cloth_resnet_linear:FrankaDMPClothPlaceEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.refactored_only_cloth_resnet_linear:FrankaDMPClothPlaceEnvCfg",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg_decimated:FrankaClothPPORunnerCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
    },
)

gym.register(
    id="Hanging-Cloth",
    entry_point=f"{__name__}.hanging_cloth:FrankaDMPClothPlaceEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.hanging_cloth:FrankaDMPClothPlaceEnvCfg",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg_decimated:FrankaClothPPORunnerCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
    },
)

gym.register(
    id="Hanging-Cloth-Isaac",
    entry_point=f"{__name__}.hanging_cloth_isaac:FrankaDMPClothPlaceEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.hanging_cloth_isaac:FrankaDMPClothPlaceEnvCfg",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg_decimated:FrankaClothPPORunnerCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_sac_cfg.yaml",
    },
)

gym.register(
    id="Refactored-Only-Cloth-Autoencoder",
    entry_point=f"{__name__}.refactored_only_cloth_autoencoder:FrankaDMPClothPlaceEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.refactored_only_cloth_autoencoder:FrankaDMPClothPlaceEnvCfg",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg_decimated:FrankaClothPPORunnerCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_sac_cfg.yaml",
    },
)
