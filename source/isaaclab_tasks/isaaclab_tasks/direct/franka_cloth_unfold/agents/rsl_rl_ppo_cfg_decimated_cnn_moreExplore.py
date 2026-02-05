# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg

from isaaclab.utils import configclass


@configclass
class FrankaClothPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 1
    max_iterations = 1500
    save_interval = 5
    experiment_name = "franka_unfold"
    empirical_normalization = True
    policy = RslRlPpoActorCriticCfg(
        class_name='ActorCriticCNN',
        init_noise_std=3.0,
        actor_hidden_dims=[256, 128, 64],
        critic_hidden_dims=[256, 128, 64],
        activation="elu",
        actor_cnn_cfg={
            "image": dict(
                output_channels=[32, 64, 64, 64, 64],
                kernel_size=[8, 4, 3, 3, 3],
                stride=[4, 2, 2, 2, 2],
                padding="zeros",
                norm="none",
                activation="elu",
                max_pool=False,
                global_pool="none",
                flatten=True,
            )
        },
        critic_cnn_cfg={
            "image": dict(
                output_channels=[32, 64, 64, 64, 64],
                kernel_size=[8, 4, 3, 3, 3],
                stride=[4, 2, 2, 2, 2],
                padding="zeros",
                norm="none",
                activation="elu",
                max_pool=False,
                global_pool="none",
                flatten=True,
            )
        },
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.3,
        entropy_coef=0.03,
        num_learning_epochs=4,
        num_mini_batches=8,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.005,
        max_grad_norm=1.0,
    )
