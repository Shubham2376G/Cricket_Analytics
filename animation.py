import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


# Load pose estimation JSON
def load_pose_data(json_path):
    with open(json_path, 'r') as f:
        data = json.load(f)
    return data


# Define skeleton connections
SKELETON = [[0, 1], [1, 2], [2, 3], [0, 4], [4, 5],
            [5, 6], [0, 7], [7, 8], [8, 9], [9, 10],
            [8, 11], [11, 12], [12, 13], [8, 14], [14, 15], [15, 16]]


def extract_frames(data):
    frames = []
    for frame in data["frames"]:
        joints = np.array([[j["x"], j["y"], j["z"]] for j in frame["joints"]])
        frames.append(joints)
    return frames


# Function to create cricket pitch
def create_cricket_pitch(ax):
    # Cricket pitch dimensions in meters (standard is 20.12m x 3.05m)
    # Scale to fit our coordinate system
    pitch_length = 6.6  # Length along y-axis
    pitch_width = 1  # Width along x-axis

    # Define the corners of the pitch (rectangular surface)
    pitch_corners = np.array([
        [-pitch_width / 2, -pitch_length / 2, 0],  # Bottom left
        [pitch_width / 2, -pitch_length / 2, 0],  # Bottom right
        [pitch_width / 2, pitch_length / 2, 0],  # Top right
        [-pitch_width / 2, pitch_length / 2, 0]  # Top left
    ])

    # Create vertices for the pitch surface
    vertices = [
        [pitch_corners[0], pitch_corners[1], pitch_corners[2], pitch_corners[3]]
    ]

    # Create a Poly3DCollection
    pitch = Poly3DCollection(vertices, alpha=0.7)
    pitch.set_facecolor('tan')  # Light brown color for the pitch
    pitch.set_edgecolor('k')
    ax.add_collection3d(pitch)

    # Add crease lines (bowling crease at each end)
    # Bowling crease at one end
    bowling_crease1_x = [-pitch_width / 2 - 0.05, pitch_width / 2 + 0.05]
    bowling_crease1_y = [-pitch_length / 2, -pitch_length / 2]
    bowling_crease1_z = [0.001, 0.001]  # Slightly above the pitch
    ax.plot(bowling_crease1_x, bowling_crease1_y, bowling_crease1_z, 'white', linewidth=2)

    # Bowling crease at other end
    bowling_crease2_x = [-pitch_width / 2 - 0.05, pitch_width / 2 + 0.05]
    bowling_crease2_y = [pitch_length / 2, pitch_length / 2]
    bowling_crease2_z = [0.001, 0.001]
    ax.plot(bowling_crease2_x, bowling_crease2_y, bowling_crease2_z, 'white', linewidth=2)

    # Add popping creases
    popping_crease1_x = [-pitch_width / 2, -pitch_width / 2]
    popping_crease1_y = [-pitch_length / 2 - 0.05, -pitch_length / 2 + 0.15]
    popping_crease1_z = [0.001, 0.001]
    ax.plot(popping_crease1_x, popping_crease1_y, popping_crease1_z, 'white', linewidth=2)

    popping_crease2_x = [pitch_width / 2, pitch_width / 2]
    popping_crease2_y = [-pitch_length / 2 - 0.05, -pitch_length / 2 + 0.15]
    popping_crease2_z = [0.001, 0.001]
    ax.plot(popping_crease2_x, popping_crease2_y, popping_crease2_z, 'white', linewidth=2)

    # Add popping creases at the other end
    popping_crease3_x = [-pitch_width / 2, -pitch_width / 2]
    popping_crease3_y = [pitch_length / 2 + 0.05, pitch_length / 2 - 0.15]
    popping_crease3_z = [0.001, 0.001]
    ax.plot(popping_crease3_x, popping_crease3_y, popping_crease3_z, 'white', linewidth=2)

    popping_crease4_x = [pitch_width / 2, pitch_width / 2]
    popping_crease4_y = [pitch_length / 2 + 0.05, pitch_length / 2 - 0.15]
    popping_crease4_z = [0.001, 0.001]
    ax.plot(popping_crease4_x, popping_crease4_y, popping_crease4_z, 'white', linewidth=2)

    # Add a green field around the pitch
    field_size = 1.2
    field_corners = np.array([
        [-field_size, -field_size, -0.001],  # Slightly below the pitch
        [field_size, -field_size, -0.001],
        [field_size, field_size, -0.001],
        [-field_size, field_size, -0.001]
    ])

    field_vertices = [
        [field_corners[0], field_corners[1], field_corners[2], field_corners[3]]
    ]

    field = Poly3DCollection(field_vertices, alpha=0.5)
    field.set_facecolor('green')
    field.set_edgecolor('darkgreen')
    ax.add_collection3d(field)


# Animate the 3D skeleton
def animate_3d_pose(frames, elev=30, azim=60):
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Set appropriate limits
    ax.set_xlim([-5.8, 5.8])
    ax.set_ylim([-5.8, 5.8])
    ax.set_zlim([0, 1.2])
    ax.view_init(elev=elev, azim=azim)

    # Adding axis labels
    ax.set_xlabel("X Axis", fontsize=12)
    ax.set_ylabel("Y Axis", fontsize=12)
    ax.set_zlabel("Z Axis", fontsize=12)
    ax.set_title("Cricket Batsman 3D Pose Animation", fontsize=14)

    # Add cricket pitch to the scene
    create_cricket_pitch(ax)

    # Initialize lines for skeleton
    lines = [ax.plot([], [], [], 'ro-', markersize=3, linewidth=2)[0] for _ in SKELETON]

    # Add joint markers with different colors for better visibility
    markers = [ax.plot([], [], [], 'o', markersize=5, color='blue')[0] for _ in range(17)]

    def update(frame_idx):
        joints = frames[frame_idx]

        # Update skeleton lines
        for line, (i, j) in zip(lines, SKELETON):
            line.set_data([joints[i, 0], joints[j, 0]], [joints[i, 1], joints[j, 1]])
            line.set_3d_properties([joints[i, 2], joints[j, 2]])

        # Update joint markers
        for i, marker in enumerate(markers):
            marker.set_data([joints[i, 0]], [joints[i, 1]])
            marker.set_3d_properties([joints[i, 2]])

        return lines + markers

    ani = animation.FuncAnimation(fig, update, frames=len(frames), interval=50, blit=False)
    plt.tight_layout()

    # Return both the figure and animation for flexibility
    return fig, ani


# Main execution
def main():
    json_path = "demo/3d_pose/sim/all_poses_3d.json"  # Update with your JSON file path
    data = load_pose_data(json_path)
    frames = extract_frames(data)
    fig, ani = animate_3d_pose(frames, elev=15, azim=70)

    # Show the animation
    plt.show()

    # Save the animation (uncomment to save)
    # ani.save("cricket_pose_animation.mp4", writer="ffmpeg", fps=20)


if __name__ == "__main__":
    main()