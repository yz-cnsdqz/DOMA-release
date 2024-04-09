import numpy as np


def rotation_matrix_z(angle):
    """Create a rotation matrix for a rotation around the z-axis."""
    cos_a, sin_a = np.cos(angle), np.sin(angle)
    return np.array([
        [cos_a, -sin_a, 0],
        [sin_a, cos_a, 0],
        [0, 0, 1]
    ])

# Number of points and time steps
num_points = 3000
num_time_steps = 20

# Generate initial random 3D points
points = np.random.rand(num_points, 3)*2-1  # 1000 points with 3 coordinates (x, y, z)

# Angle of rotation per time step (in radians), e.g., pi/20 radians per step
angle_step = np.pi / 20

# Create a sequence of points over time
sequence_of_points = np.zeros((num_time_steps, num_points, 3))

# Apply the rotation over each time step
for t in range(num_time_steps):
    # Compute the rotation matrix for the current time step
    rotation_mat = rotation_matrix_z(t * angle_step)
    # Apply the rotation to all points
    sequence_of_points[t] = points.dot(rotation_mat.T)
np.save('rotation.npy', sequence_of_points)


