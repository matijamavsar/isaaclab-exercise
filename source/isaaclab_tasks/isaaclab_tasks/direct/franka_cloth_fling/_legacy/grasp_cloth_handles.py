import torch
from isaaclab.utils.math import subtract_frame_transforms
from isaaclab.utils.math import matrix_from_quat, quat_from_matrix


def go_to_pose(
    robot_dict,
    robot_entity_cfg_dict,
    diff_ik_controller_dict,
    goal_dict,
    gripper_value,
    max_time_steps,
    sim,
    sim_dt,
    scene,
    env_ids,
    device
):
    """
    Move multiple robots simultaneously toward goal poses by setting IK-based
    joint-position targets each simulation step.

    Args:
        robot_dict (dict): Dictionary of robot handles, e.g. {'robot_1': <robotObj>, 'robot_2': <robotObj>}
        robot_entity_cfg_dict (dict): Corresponding dictionary of robot entity configs
        diff_ik_controller_dict (dict): Dictionary of diff-IK controllers
        goal_dict (dict): Dictionary of end-effector pose goals for each robot
        gripper_value (float): Value to set for the last two gripper joints
        max_time_steps (int): How many simulation steps to apply
        sim, sim_dt, scene, device: Simulation-related arguments
    """
    ee_jacobi_idx = 7
    # Prepare a dict to hold joint-position targets for each robot
    joint_pos_des = {}

    # Initialize each robot's diff IK controller
    for key in robot_dict.keys():
        diff_ik_controller_dict[key].reset(env_ids=env_ids)

        ik_commands = torch.zeros(scene.num_envs,
                                  diff_ik_controller_dict[key].action_dim,
                                  device=device)
        ik_commands[:] = goal_dict[key]
        diff_ik_controller_dict[key].set_command(ik_commands)

        # Start everyone off with zero positions; set gripper positions
        joint_pos_des[key] = torch.zeros((scene.num_envs, 9), device=device)
        joint_pos_des[key][:, -2:] = gripper_value

    count = 0
    while count < max_time_steps:
        for key in robot_dict.keys():
            # Extract the relevant data for this robot
            jacobian = robot_dict[key].root_physx_view.get_jacobians()[
                :, ee_jacobi_idx, :, robot_entity_cfg_dict[key].joint_ids
            ]
            ee_pose_w = robot_dict[key].data.body_state_w[
                :, robot_entity_cfg_dict[key].body_ids[0], 0:7
            ]
            root_pose_w = robot_dict[key].data.root_state_w[:, 0:7]
            joint_pos = robot_dict[key].data.joint_pos[:, robot_entity_cfg_dict[key].joint_ids]

            # Compute the end-effector pose in the robot's base frame
            ee_pos_b, ee_quat_b = subtract_frame_transforms(
                root_pose_w[:, 0:3], root_pose_w[:, 3:7],
                ee_pose_w[:, 0:3], ee_pose_w[:, 3:7]
            )

            # IK-based joint command
            joint_pos_des[key][:, 0:7] = diff_ik_controller_dict[key].compute(
                ee_pos_b, ee_quat_b, jacobian, joint_pos
            )

            # Apply the computed target to this robot
            robot_dict[key].set_joint_position_target(
                joint_pos_des[key][env_ids], joint_ids=torch.arange(0, 9).tolist(), env_ids=env_ids
            )

        # Update the simulator
        scene.write_data_to_sim()
        sim.step()
        count += 1
        scene.update(sim_dt)

    return joint_pos_des


def go_to_joints(
    robot_dict,
    joint_targets_dict,
    steps,
    sim,
    sim_dt,
    env_ids,
    scene
):
    """
    Set a joint target for each robot and loop for a fixed number of simulation steps.

    Args:
        robot_dict (dict): Dictionary of robot handles
        joint_targets_dict (dict): Dictionary of per-robot joint targets (each an Nx9 tensor)
        steps (int): How many steps to simulate
        sim, sim_dt, scene: Simulation-related arguments
    """
    count = 0
    while count < steps:
        for key in robot_dict.keys():
            robot_dict[key].set_joint_position_target(
                joint_targets_dict[key][env_ids], joint_ids=torch.arange(0, 9).tolist(), env_ids=env_ids
            )

        scene.write_data_to_sim()
        sim.step()
        scene.update(sim_dt)
        count += 1


def grasp_cloth_handles(scene,
                        diff_ik_controller,
                        goal,
                        robot,
                        robot_entity_cfg,
                        sim, sim_dt,
                        env_ids,
                        device):
    # Initialize and reset each robot
    for key in robot.keys():
        joint_pos = robot[key].data.default_joint_pos[env_ids].clone()
        joint_vel = robot[key].data.default_joint_vel[env_ids].clone()
        robot[key].write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)
        robot[key].reset(env_ids=env_ids)

        # Transform goal from world frame to robot's base frame
        root_pose_w = robot[key].data.root_state_w[:, 0:7]
        goal_pos_b, goal_quat_b = subtract_frame_transforms(
            root_pose_w[:, 0:3], root_pose_w[:, 3:7],
            goal[key][:, 0:3], goal[key][:, 3:7]
        )
        goal_mat_b = matrix_from_quat(goal_quat_b)
        rot_z_mat = matrix_from_quat(
            torch.tensor([0.7071, 0, 0, 0.7071], device=device)
        )
        goal_quat_b = quat_from_matrix(goal_mat_b @ rot_z_mat)
        goal[key][:, 0:3] = goal_pos_b
        goal[key][:, 3:] = goal_quat_b
        goal[key][:, 2] = 0.12

    # -------------------------------------------------------------------------
    #  Move both robots down to the cloth handles
    # -------------------------------------------------------------------------
    joint_pos_last = go_to_pose(
        robot_dict=robot,
        robot_entity_cfg_dict=robot_entity_cfg,
        diff_ik_controller_dict=diff_ik_controller,
        goal_dict=goal,
        gripper_value=0.04,
        max_time_steps=60,
        sim=sim, sim_dt=sim_dt,
        scene=scene,
        env_ids=env_ids,
        device=device
    )

    # for key in robot.keys():
    #     goal[key][:, 2] = 0.12
    # joint_pos_last = go_to_pose(
    #     robot_dict=robot,
    #     robot_entity_cfg_dict=robot_entity_cfg,
    #     diff_ik_controller_dict=diff_ik_controller,
    #     goal_dict=goal,
    #     gripper_value=0.04,
    #     max_time_steps=30,
    #     sim=sim, sim_dt=sim_dt,
    #     scene=scene,
    #     device=device
    # )

    # Now close the gripper on each robot (go_to_joints() usage)
    # We'll clone the last positions and just set gripper to 0
    joint_targets = {}
    for key in robot.keys():
        joint_targets[key] = joint_pos_last[key].clone()
        joint_targets[key][:, -2:] = 0.00
        cur_stiff = robot[key].data.joint_stiffness.clone()
        cur_stiff[:,-2:] = 4000.0
        robot[key].write_joint_stiffness_to_sim(cur_stiff, env_ids=env_ids)

    go_to_joints(
        robot_dict=robot,
        joint_targets_dict=joint_targets,
        steps=25,
        sim=sim,
        sim_dt=sim_dt,
        env_ids=env_ids,
        scene=scene
    )

    # -------------------------------------------------------------------------
    #  Lift the cloth up
    # -------------------------------------------------------------------------
    # Update the 'goal' for each robot to some "above table" position
    for key in robot.keys():
        goal[key][:, 0] = 0.5
        goal[key][:, 2] = 0.5
    goal['robot_1'][:, 1] = 0.18
    goal['robot_2'][:, 1] = -0.18

    # Move both robots to the new goals
    go_to_pose(
        robot_dict=robot,
        robot_entity_cfg_dict=robot_entity_cfg,
        diff_ik_controller_dict=diff_ik_controller,
        goal_dict=goal,
        gripper_value=0.0,
        max_time_steps=30,
        sim=sim, sim_dt=sim_dt,
        scene=scene,
        env_ids=env_ids,
        device=device
    )
