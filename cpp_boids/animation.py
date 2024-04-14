import os
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d as mplot3d
from matplotlib.animation import FuncAnimation
from plyfile import PlyData
import numpy as np
# Directory containing PLY files
ply_dir = "boid_ply_data"

# Get list of PLY files in the directory
def extract_number(filename):
    return int(filename.split('.')[0])  # Extract the numeric part before the first dot and convert it to an integer

ply_files = sorted([f for f in os.listdir(ply_dir) if f.endswith('.ply')], key=extract_number)

# Set up plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.set_xlim([-100, 100])  # Adjust as needed
ax.set_ylim([-100, 100])  # Adjust as needed
ax.set_zlim([-100, 100])  # Adjust as needed
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')



# Create animation
interval_ms = 1000 / 30  # 1000 milliseconds in 1 second
interval = int(interval_ms)  # Convert to integer (milliseconds)


# Function to update plot for each frame
def update(frame):
    ply_file = ply_files[frame]
    ply_data = PlyData.read(os.path.join(ply_dir, ply_file))
    vertices = ply_data['vertex']
    x = vertices['x']
    y = vertices['y']
    z = vertices['z']
    colors = np.array([x, y, z]).T
    colors -= np.min(colors, axis=0)
    colors /= np.max(colors, axis=0)

    # Update scatter plot data
    sc.set_offsets(np.column_stack([x, y]))
    sc.set_3d_properties(z, 'z')
    sc.set_color(colors)
    ax.set_title(f'Frame {frame}')


# Set up empty scatter plot
sc = ax.scatter([], [], [], marker='.', s=1)

# Create animation
ani = FuncAnimation(fig, update, frames=len(ply_files), interval=interval)

# Save animation as GIF
ani.save('animation_smooth_coherent.gif', writer='pillow')

# check frame order
# for i in range(len(ply_files)):
#     print(ply_files[i])
