import os
import cv2
import random
import supervision as sv
from ultralytics import YOLO
import json
from transformers import pipeline
import torch
from PIL import Image


def pose_estimationv1(json_file, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    pose_model = pipeline("image-classification", model="runs/ViTPose")

    with open(json_file, "r") as file:
        data = json.load(file)

    pose_results = []
    for frame_data in data:
        frame_id = frame_data["frame"]
        detections = frame_data["detections"]

        for detection in detections:
            x1, y1, x2, y2 = detection["x1"], detection["y1"], detection["x2"], detection["y2"]
            label = detection["label"]

            cropped_image = Image.open(json_file).crop((x1, y1, x2, y2))
            keypoints = pose_model(cropped_image)

            pose_results.append({"frame": frame_id, "label": label, "keypoints": keypoints})

    pose_output_path = os.path.join(output_dir, "pose_estimations.json")
    with open(pose_output_path, "w") as pose_file:
        json.dump(pose_results, pose_file, indent=4)

    print("Pose estimation complete. Check the output directory for results.")