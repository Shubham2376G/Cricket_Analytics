from scenedetect import SceneManager, open_video, detect, ContentDetector
from scenedetect.video_splitter import split_video_ffmpeg
import cv2
import os
import shutil
import numpy as np
from skimage.metrics import structural_similarity as ssim

def segment_video(video_path, output_dir):


    # Open video and set up scene manager
    video = open_video(video_path)
    scene_manager = SceneManager()
    scene_manager.add_detector(ContentDetector())

    # Detect scenes
    scene_manager.detect_scenes(video)
    scene_list = scene_manager.get_scene_list()

    # Print detected scenes
    print(f"Detected {len(scene_list)} scenes.")
    for i, (start, end) in enumerate(scene_list):
        print(f"Scene {i + 1}: Start {start.get_timecode()} - End {end.get_timecode()}")

    # Split video based on detected scenes
    split_video_ffmpeg(video_path, scene_list, output_dir)
    print("Segmentation complete. Clips saved in:", output_dir)


def calculate_ssim(image1, image2):
    """Compute SSIM similarity between two images."""
    return ssim(image1, image2)


def extract_frames(video_path, frame_interval=30):
    """Extract frames from video at a given interval."""
    cap = cv2.VideoCapture(video_path)
    frames = []
    frame_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if frame_count % frame_interval == 0:
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray_frame = cv2.resize(gray_frame, (100, 100))  # Resize
            frames.append(gray_frame)
        frame_count += 1

    cap.release()
    return frames


def classify_video(video_path, threshold=0.5, reference_img_path="assets/reference.png"):
    """Classify video based on frame similarity with reference image."""
    # Load reference image
    reference_img = cv2.imread(reference_img_path, cv2.IMREAD_GRAYSCALE)
    reference_img = cv2.resize(reference_img, (100, 100))  # Resize to standard size
    frames = extract_frames(video_path)
    similarities = [calculate_ssim(reference_img, frame) for frame in frames]

    avg_similarity = np.mean(similarities) if similarities else 0
    return avg_similarity > threshold


def classify_all_videos(input_dir):

    rally_dir = f"{input_dir}/rally_videos/"
    no_rally_dir = f"{input_dir}/no_rally_videos/"

    os.makedirs(rally_dir, exist_ok=True)
    os.makedirs(no_rally_dir, exist_ok=True)

    # Process all videos
    for video_file in os.listdir(input_dir):
        video_path = os.path.join(input_dir, video_file)

        # Skip directories
        if os.path.isdir(video_path):
            continue

        if classify_video(video_path):
            shutil.move(video_path, os.path.join(rally_dir, video_file))
            print(f"Moved {video_file} to rally_videos/")
        else:
            shutil.move(video_path, os.path.join(no_rally_dir, video_file))
            print(f"Moved {video_file} to no_rally_videos/")

