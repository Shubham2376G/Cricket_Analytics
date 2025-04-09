import os
import cv2
import random
import supervision as sv
from ultralytics import YOLO
import json


def detect(input_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    model = YOLO('runs/detect/train/weights/best.pt')

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for video_name in os.listdir(input_dir):
        if not video_name.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
            continue

        video_path = os.path.join(input_dir, video_name)
        output_path = os.path.join(output_dir, f"processed_{video_name}")
        json_output_path = os.path.join(output_dir, f"processed_{video_name}.json")

        cap = cv2.VideoCapture(video_path)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        frame_data = []
        frame_count = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            results = model(frame, verbose=False)[0]
            detections = sv.Detections.from_ultralytics(results).with_nms()

            box_annotator = sv.BoxAnnotator()
            label_annotator = sv.LabelAnnotator()

            annotated_frame = frame.copy()
            annotated_frame = box_annotator.annotate(scene=annotated_frame, detections=detections)
            annotated_frame = label_annotator.annotate(scene=annotated_frame, detections=detections)

            out.write(annotated_frame)

            frame_detections = []
            for detection in detections:
                print(detection)  # Check the structure of detection
                bbox, _, _, _, _, metadata = detection  # Unpack the tuple
                x1, y1, x2, y2 = bbox  # Extract the bounding box coordinates
                label = metadata.get("class_name", "Unknown")  # Extract the label safely

                frame_detections.append({
                    "x1": float(x1),
                    "y1": float(y1),
                    "x2": float(x2),
                    "y2": float(y2),
                    "label": label
                })

            frame_data.append({"frame": frame_count, "detections": frame_detections})
            frame_count += 1

        cap.release()
        out.release()

        with open(json_output_path, "w") as json_file:
            json.dump(frame_data, json_file, indent=4)

    print("Processing complete. Check the output directory for results.")
