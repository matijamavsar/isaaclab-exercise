import rclpy
from rclpy.executors import SingleThreadedExecutor
from sensor_msgs.msg import JointState
from std_msgs.msg import Header
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from builtin_interfaces.msg import Duration

from robotblockset.ros2.franka_ros2 import franka_ros2
from robotblockset.transformations import x2t
import time

import numpy as np

rclpy.init()
robot = franka_ros2(ns='env_0/robot1')

time.sleep(2)
robot.ResetCurrentTarget()

wait_time = 1.5

new_tcp = np.eye(4)
# robot.ResetCurrentTarget()
# robot.JPath(np.array([robot.q_ref, robot.q_ref]), t=2)
# time.sleep(2)
# robot.ResetCurrentTarget()
# robot.JPath(np.array([robot.q_ref, robot.q_ref]), t=2)
# time.sleep(2)
# breakpoint()

breakpoint()
robot.ResetCurrentTarget()
robot.CMove([ 0.3305,  0.0012,  0.5336,  0.2225, -0.7545,  0.3191, -0.5286], t=wait_time)
robot.Wait(wait_time)

print(robot.x_ref)

robot.ResetCurrentTarget()
robot.CMove([ 0.4305,  0.1012,  0.6336,  0.2225, -0.7545,  0.3191, -0.5286], t=wait_time)
robot.Wait(wait_time)
robot.ResetCurrentTarget()
robot.CMoveFor(np.array([-0.2, 0, 0.2]), t=wait_time)
robot.Wait(wait_time)
robot.ResetCurrentTarget()
robot.CMoveFor(np.array([0, 0, -0.2]), t=wait_time)
robot.Wait(wait_time)
robot.ResetCurrentTarget()

breakpoint()
q_init = np.array([-0.2106, -1.0490,  0.0865, -2.1993,  0.2727,  2.3895,  0.6149])
q_2 = q_init + 0.1
q_3 = q_init + 0.2
q_4 = q_init + 0.1
q_5 = q_init

q_list = np.array([q_init, q_2, q_3, q_4, q_5])
v_list = np.zeros_like(q_list)
t_list = np.linspace(0, 5, len(q_list))

robot.GoTo_qtraj_pub(q_list, v_list, v_list, time=t_list)