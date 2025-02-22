from scenedetect import SceneManager, open_video, detect, ContentDetector
from scenedetect.video_splitter import split_video_ffmpeg
from utils.segment import segment_video
from utils.detection import detect
from utils.pose_estimationv1 import pose_estimationv1
import os

HOME = os.getcwd()
print(HOME)

# Rally segmentation
segment_video("demo/Cric Test2.mp4", "demo/segments_2")

#Player detection: Batsman - Bowler
detect("demo/segments_2", "demo/segments_2/bounding_boxes")

#Pose estimation: Batsman - Bowler
pose_estimationv1("demo/segments_2/bounding_boxes/detections.json", "demo/segments_2/rallies/output")