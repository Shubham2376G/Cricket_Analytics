from scenedetect import SceneManager, open_video, detect, ContentDetector
from scenedetect.video_splitter import split_video_ffmpeg
from utils.segment import segment_video, calculate_ssim, extract_frames, classify_video, classify_all_videos
from utils.detection import detect
from utils.pose_estimationv1 import *
import os

HOME = os.getcwd()
print(HOME)

# # Rally segmentation
# segment_video("demo/Cric Test2.mp4", "demo/segments_2")
# classify_all_videos("demo/segments_2")
#
# #Player detection: Batsman - Bowler
# detect("demo/3d_pose/sim", "demo/3d_pose/sim/bounding_boxes")

#Pose estimation: Batsman - Bowler
pose_estimation("cric.mp4","cric1.json","output2")
convert_json_to_h36m_npz(json_path, output_dir)


import json
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation


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


# Animate the 3D skeleton
def animate_3d_pose(frames, elev=30, azim=60):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlim([-0.5, 0.5])
    ax.set_ylim([-0.5, 0.5])
    ax.set_zlim([0, 1])
    ax.view_init(elev=elev, azim=azim)
    # Adding axis labels
    ax.set_xlabel("X Axis", fontsize=12)
    ax.set_ylabel("Y Axis", fontsize=12)
    ax.set_zlabel("Z Axis", fontsize=12)

    # Initialize lines for skeleton
    lines = [ax.plot([], [], [], 'ro-')[0] for _ in SKELETON]

    def update(frame_idx):
        joints = frames[frame_idx]
        for line, (i, j) in zip(lines, SKELETON):
            line.set_data([joints[i, 0], joints[j, 0]], [joints[i, 1], joints[j, 1]])
            line.set_3d_properties([joints[i, 2], joints[j, 2]])
        return lines

    ani = animation.FuncAnimation(fig, update, frames=len(frames), interval=50, blit=False)
    plt.show()
    ani.save("pose_estimation.mp4", writer="ffmpeg", fps=20)

# Main execution
json_path = "demo/3d_pose/sim/all_poses_3d.json"  # Update this with your JSON file path
data = load_pose_data(json_path)
frames = extract_frames(data)
animate_3d_pose(frames, elev=15, azim=70)
