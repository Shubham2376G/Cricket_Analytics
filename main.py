from scenedetect import SceneManager, open_video, detect, ContentDetector
from scenedetect.video_splitter import split_video_ffmpeg
from utils.segment import segment_video, calculate_ssim, extract_frames, classify_video, classify_all_videos
from utils.detection import detect
from utils.pose_estimationv1 import pose_estimation
import os

HOME = os.getcwd()
print(HOME)

# Rally segmentation
segment_video("demo/Cric Test2.mp4", "demo/segments_2")
classify_all_videos("demo/segments_2")

#Player detection: Batsman - Bowler
detect("demo/segments_2/rally_videos", "demo/segments_2/rally_videos/bounding_boxes")

#Pose estimation: Batsman - Bowler
pose_estimation("demo/segments_2/rally_videos/bounding_boxes/detections.json", "demo/output")