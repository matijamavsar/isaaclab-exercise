import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

### Example use ###
"""
generate_two_segments([0,0,1], [1,0,0], [1,0,1], 3, 3)
"""

def minimum_jerk_trajectory_3d(A, B, T, num_points=100):
    """
    Generates a minimum jerk trajectory in 3D space from point A to point B over time T.
    
    :param A: Starting position (array-like [Ax, Ay, Az])
    :param B: Ending position (array-like [Bx, By, Bz])
    :param T: Total time for the motion
    :param num_points: Number of points in the trajectory
    :return: Time values and trajectory values (position at each time step)
    """
    # Generate time values (linearly spaced from 0 to T)
    t = np.linspace(0, T, num_points)
    
    # Calculate the minimum jerk trajectory for each dimension (X, Y, Z)
    trajectory_x = A[0] + (B[0] - A[0]) * (10 * (t / T) ** 3 - 15 * (t / T) ** 4 + 6 * (t / T) ** 5)
    trajectory_y = A[1] + (B[1] - A[1]) * (10 * (t / T) ** 3 - 15 * (t / T) ** 4 + 6 * (t / T) ** 5)
    trajectory_z = A[2] + (B[2] - A[2]) * (10 * (t / T) ** 3 - 15 * (t / T) ** 4 + 6 * (t / T) ** 5)
    
    # Return the time values and the trajectory in 3D space (X, Y, Z)
    return t, trajectory_x, trajectory_y, trajectory_z


def generate_two_segments(A, B, C, T1, T2, num_points):
    # Step 1: Generate the first segment from A to B
    t1, trajectory_x1, trajectory_y1, trajectory_z1 = minimum_jerk_trajectory_3d(A, B, T1, num_points//2)
    t2, trajectory_x2, trajectory_y2, trajectory_z2 = minimum_jerk_trajectory_3d(B, C, T2, num_points//2)

    # Combine the trajectories (first segment + second segment)
    t_combined = np.concatenate((t1, t2 + T1))  # Shift t2 to start after t1
    trajectory_x_combined = np.concatenate((trajectory_x1, trajectory_x2))
    trajectory_y_combined = np.concatenate((trajectory_y1, trajectory_y2))
    trajectory_z_combined = np.concatenate((trajectory_z1, trajectory_z2))

    # # Plotting the combined trajectory in 3D
    # fig = plt.figure(figsize=(10, 6))
    # ax = fig.add_subplot(111, projection='3d')

    # # Plot the combined trajectory
    # ax.plot(trajectory_x_combined, trajectory_y_combined, trajectory_z_combined, label="Combined Minimum Jerk Trajectory", color="b")
    # ax.scatter([A[0], B[0]], [A[1], B[1]], [A[2], B[2]], color='r', label="Start and End Points")

    # # Labels and title
    # ax.set_title("Combined Minimum Jerk Trajectory in 3D Space")
    # ax.set_xlabel("X")
    # ax.set_ylabel("Y")
    # ax.set_zlabel("Z")
    # ax.legend()

    # plt.show(block=False)

    min_jerk_trajectory = np.stack((trajectory_x_combined, 
                                    trajectory_y_combined, 
                                    trajectory_z_combined)).T
    return min_jerk_trajectory


def generate_minimum_jerk(waypoints, durations, num_points):
    """
    Build a minimum‐jerk trajectory through an arbitrary list of 3D waypoints.
    
    Args:
        waypoints: list of (3,) arrays or tuples, e.g. [(x0,y0,z0), (x1,y1,z1), ...]
        durations: list of floats, durations[i] is time to go from waypoints[i] to waypoints[i+1]
        num_points: int, total number of trajectory points across *all* segments

    Returns:
        trajectory: (num_points, 3) array of xyz positions
    """
    assert len(waypoints) >= 2, "Need at least two waypoints"
    assert len(durations) == len(waypoints) - 1, "durations must match segments"

    total_time = sum(durations)
    # allocate points per segment (at least 2 each to include endpoints)
    pts_per_seg = [
        max(2, int(round(num_points * (dur / total_time))))
        for dur in durations
    ]
    # adjust if rounding error
    diff = sum(pts_per_seg) - num_points
    # if we have too many, shave off from the longest segment(s)
    # if too few, add to the longest segment(s)
    if diff != 0:
        # sort segments by duration descending
        order = sorted(range(len(durations)), key=lambda i: durations[i], reverse=(diff>0))
        for i in order:
            if diff == 0:
                break
            # for negative diff, we need to add points; for positive, remove
            pts_per_seg[i] -= np.sign(diff)
            diff -= np.sign(diff)

    segments = []
    t_offset = 0.0
    for i, (A, B) in enumerate(zip(waypoints[:-1], waypoints[1:])):
        Ti = durations[i]
        Ni = pts_per_seg[i]
        # call your existing 3D minimum‐jerk
        t_seg, x, y, z = minimum_jerk_trajectory_3d(A, B, Ti, Ni)
        pts = np.stack((x, y, z), axis=1)
        # drop the first point for all but the very first segment so we don't duplicate
        if i > 0:
            pts = pts[1:]
        segments.append(pts)
        t_offset += Ti

    trajectory = np.concatenate(segments, axis=0)
    return trajectory