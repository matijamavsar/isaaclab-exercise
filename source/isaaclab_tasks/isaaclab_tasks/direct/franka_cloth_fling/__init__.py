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
    id="Bimanual-Residual-Place-Direct-v0",
    entry_point=f"{__name__}.bimanual_residual_place:FrankaClothEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.bimanual_residual_place:FrankaClothEnvCfg",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:FrankaClothPPORunnerCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
    },
)

gym.register(
    id="Bimanual-Imitation-Franka-Place-Cloth-Direct-v0",
    entry_point=f"{__name__}.bimanual_imitation_place_cloth_deformable:FrankaClothEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.bimanual_imitation_place_cloth_deformable:FrankaClothEnvCfg",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:FrankaClothPPORunnerCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
    },
)

gym.register(
    id="Bimanual-Imitation-Franka-Place-Cloth-Particle-v0",
    entry_point=f"{__name__}.bimanual_imitation_place_cloth_particle:FrankaClothEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.bimanual_imitation_place_cloth_particle:FrankaClothEnvCfg",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:FrankaClothPPORunnerCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
    },
)

gym.register(
    id="Less-Explore-Bimanual-Franka-Place-Cloth-Direct-v0",
    entry_point=f"{__name__}.bimanual_place_cloth_deformable:FrankaClothEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.bimanual_place_cloth_deformable:FrankaClothEnvCfg",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg_lessExplore:FrankaClothPPORunnerCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
    },
)

gym.register(
    id="DMP-Based-Cloth-Place",
    entry_point=f"{__name__}.dmp_based_cloth_place:FrankaDMPClothPlaceEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.dmp_based_cloth_place:FrankaDMPClothPlaceEnvCfg",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:FrankaClothPPORunnerCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
    },
)

gym.register(
    id="DMP-Based-Cloth-Place-Decimated",
    entry_point=f"{__name__}.dmp_based_cloth_place_decimated:FrankaDMPClothPlaceEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.dmp_based_cloth_place_decimated:FrankaDMPClothPlaceEnvCfg",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg_decimated:FrankaClothPPORunnerCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
    },
)

gym.register(
    id="DMP-Based-Deformable-Randomized-Init-Motion",
    entry_point=f"{__name__}.dmp_based_deformable_randomized_init_motion:FrankaDMPClothPlaceEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.dmp_based_deformable_randomized_init_motion:FrankaDMPClothPlaceEnvCfg",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg_decimated:FrankaClothPPORunnerCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
    },
)

gym.register(
    id="DMP-Based-Particle-Randomized",
    entry_point=f"{__name__}.dmp_based_particle_randomized:FrankaDMPClothPlaceEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.dmp_based_particle_randomized:FrankaDMPClothPlaceEnvCfg",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg_decimated:FrankaClothPPORunnerCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
    },
)


gym.register(
    id="DMP-Based-Particle-Randomized-Velocity",
    entry_point=f"{__name__}.dmp_based_particle_randomized_init_motion_vel:FrankaDMPClothPlaceEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.dmp_based_particle_randomized_init_motion_vel:FrankaDMPClothPlaceEnvCfg",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg_decimated:FrankaClothPPORunnerCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_sac_cfg.yaml",
    },
)

gym.register(
    id="DMP-Based-Particle-Randomized-Position",
    entry_point=f"{__name__}.dmp_based_particle_randomized_init_motion_pos:FrankaDMPClothPlaceEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.dmp_based_particle_randomized_init_motion_pos:FrankaDMPClothPlaceEnvCfg",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg_decimated:FrankaClothPPORunnerCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_sac_cfg.yaml",
    },
)

gym.register(
    id="DMP-Based-Cloth-Fling",
    entry_point=f"{__name__}.dmp_based_cloth_fling_exercise:FrankaDMPClothPlaceEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.dmp_based_cloth_fling_exercise:FrankaDMPClothPlaceEnvCfg",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg_decimated:FrankaClothPPORunnerCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_sac_cfg.yaml",
    },
)

gym.register(
    id="DMP-Based-Particle-Multi-Update-Position",
    entry_point=f"{__name__}.dmp_based_particle_multi_update_pos:FrankaDMPClothPlaceEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.dmp_based_particle_multi_update_pos:FrankaDMPClothPlaceEnvCfg",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg_multiUpdate:FrankaClothPPORunnerCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_sac_cfg.yaml",
    },
)

gym.register(
    id="DMP-Based-Particle-Randomized-Relative-Position",
    entry_point=f"{__name__}.dmp_based_particle_randomized_init_motion_relpos:FrankaDMPClothPlaceEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.dmp_based_particle_randomized_init_motion_relpos:FrankaDMPClothPlaceEnvCfg",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg_decimated:FrankaClothPPORunnerCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_sac_cfg.yaml",
    },
)

gym.register(
    id="DMP-Based-Particle-Cloth-Fold",
    entry_point=f"{__name__}.dmp_based_particle_cloth_fold:FrankaDMPClothPlaceEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.dmp_based_particle_cloth_fold:FrankaDMPClothPlaceEnvCfg",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg_decimated:FrankaClothPPORunnerCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_sac_cfg.yaml",
    },
)

gym.register(
    id="DMP-Based-Particle-Residual-Position",
    entry_point=f"{__name__}.dmp_based_residual:FrankaDMPClothPlaceEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.dmp_based_residual:FrankaDMPClothPlaceEnvCfg",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg_decimated:FrankaClothPPORunnerCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_sac_cfg.yaml",
    },
)

gym.register(
    id="DMP-Based-Imitation-Learning",
    entry_point=f"{__name__}.dmp_based_particle_randomized_init_motion_imitation:FrankaDMPClothPlaceEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.dmp_based_particle_randomized_init_motion_imitation:FrankaDMPClothPlaceEnvCfg",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg_decimated:FrankaClothPPORunnerCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_sac_cfg.yaml",
    },
)

gym.register(
    id="DMP-Based-Particle-Randomized-Init-Motion-RLGamesNoPPO",
    entry_point=f"{__name__}.dmp_based_particle_randomized_init_motion:FrankaDMPClothPlaceEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.dmp_based_particle_randomized_init_motion:FrankaDMPClothPlaceEnvCfg",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_no_ppo_cfg.yaml",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg_decimated:FrankaClothPPORunnerCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_sac_cfg.yaml",
    },
)

gym.register(
    id="DMP-Based-Particle-Randomized-Position-Curobo",
    entry_point=f"{__name__}.dmp_based_particle_randomized_init_motion_pos_curobo:FrankaDMPClothPlaceEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.dmp_based_particle_randomized_init_motion_pos_curobo:FrankaDMPClothPlaceEnvCfg",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg_decimated:FrankaClothPPORunnerCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_sac_cfg.yaml",
    },
)

gym.register(
    id="No-Robots-Cloth-Fling",
    entry_point=f"{__name__}.no_robots_cloth_fling:FrankaDMPClothPlaceEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.no_robots_cloth_fling:FrankaDMPClothPlaceEnvCfg",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg_decimated:FrankaClothPPORunnerCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_sac_cfg.yaml",
    },
)

gym.register(
    id="No-Robots-Cloth-Fling-With-Handles",
    entry_point=f"{__name__}.no_robots_cloth_fling_with_handles:FrankaDMPClothPlaceEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.no_robots_cloth_fling_with_handles:FrankaDMPClothPlaceEnvCfg",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg_decimated:FrankaClothPPORunnerCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_sac_cfg.yaml",
    },
)

gym.register(
    id="Refactored-One-Robot-Fling",
    entry_point=f"{__name__}.refactored_one_robot_fling:FrankaDMPClothPlaceEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.refactored_one_robot_fling:FrankaDMPClothPlaceEnvCfg",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg_decimated:FrankaClothPPORunnerCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_sac_cfg.yaml",
    },
)

gym.register(
    id="Refactored-One-Robot-Fling-Curobo",
    entry_point=f"{__name__}.refactored_one_robot_fling_curobo:FrankaDMPClothPlaceEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.refactored_one_robot_fling_curobo:FrankaDMPClothPlaceEnvCfg",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg_decimated:FrankaClothPPORunnerCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_sac_cfg.yaml",
    },
)