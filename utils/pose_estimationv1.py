import os
import cv2
import random
import supervision as sv
from ultralytics import YOLO
import json
import torch
import numpy as np
from PIL import Image
from transformers import AutoProcessor, VitPoseForPoseEstimation

device = "cuda" if torch.cuda.is_available() else "cpu"

def pose_estimation(json_file, image_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    image_processor = AutoProcessor.from_pretrained("usyd-community/vitpose-base-simple")
    model = VitPoseForPoseEstimation.from_pretrained("usyd-community/vitpose-base-simple", device_map=device)

    with open(json_file, "r") as file:
        data = json.load(file)

    pose_results = []
    for frame_data in data:
        frame_id = frame_data["frame"]
        detections = frame_data["detections"]
        image_path = os.path.join(image_dir, f"frame_{frame_id}.jpg")
        image = Image.open(image_path).convert("RGB")

        person_boxes = []
        for detection in detections:
            if detection["label"] == "person":
                x1, y1, x2, y2 = detection["x1"], detection["y1"], detection["x2"], detection["y2"]
                person_boxes.append([x1, y1, x2 - x1, y2 - y1])  # Convert to COCO format (x, y, w, h)

        if not person_boxes:
            continue

        person_boxes = np.array(person_boxes)
        inputs = image_processor(image, boxes=[person_boxes], return_tensors="pt").to(device)

        with torch.no_grad():
            outputs = model(**inputs)

        pose_estimates = image_processor.post_process_pose_estimation(outputs, boxes=[person_boxes], threshold=0.3)
        image_pose_result = pose_estimates[0]

        frame_poses = []
        for i, person_pose in enumerate(image_pose_result):
            keypoints = []
            for keypoint, label, score in zip(person_pose["keypoints"], person_pose["labels"], person_pose["scores"]):
                keypoints.append({
                    "name": model.config.id2label[label.item()],
                    "x": keypoint[0].item(),
                    "y": keypoint[1].item(),
                    "score": score.item()
                })
            frame_poses.append({"person": i, "keypoints": keypoints})

        pose_results.append({"frame": frame_id, "poses": frame_poses})

    pose_output_path = os.path.join(output_dir, "pose_estimations.json")
    with open(pose_output_path, "w") as pose_file:
        json.dump(pose_results, pose_file, indent=4)

    print("Pose estimation complete. Check the output directory for results.")
