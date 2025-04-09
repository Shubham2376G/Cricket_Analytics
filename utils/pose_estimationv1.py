import os
import json
import torch
import numpy as np
import cv2
from PIL import Image
from transformers import AutoProcessor, VitPoseForPoseEstimation

device = "cuda" if torch.cuda.is_available() else "cpu"


def extract_frames(video_path, frame_dir):
    os.makedirs(frame_dir, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    frame_id = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame_path = os.path.join(frame_dir, f"frame_{frame_id}.jpg")
        cv2.imwrite(frame_path, frame)
        frame_id += 1

    cap.release()
    return frame_id


def pose_estimation(video_path, json_file, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    frame_dir = os.path.join(output_dir, "frames")
    frame_count = extract_frames(video_path, frame_dir)

    image_processor = AutoProcessor.from_pretrained("runs/ViTPose")
    model = VitPoseForPoseEstimation.from_pretrained("runs/ViTPose").to(device)

    with open(json_file, "r") as file:
        data = json.load(file)

    pose_results = []
    for frame_data in data:
        frame_id = frame_data["frame"]
        detections = frame_data["detections"]
        image_path = os.path.join(frame_dir, f"frame_{frame_id}.jpg")

        if not os.path.exists(image_path):
            continue

        image = Image.open(image_path).convert("RGB")
        person_boxes = []
        for detection in detections:
            if detection["label"] == "BM 1":
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




# Constants from your provided code
h36m_coco_order = [9, 11, 14, 12, 15, 13, 16, 4, 1, 5, 2, 6, 3]
coco_order = [0, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
spple_keypoints = [10, 8, 0, 7]

def convert_json_to_h36m_npz(json_path, output_dir):
    """
    Converts pose estimations from JSON format to H36M NPZ format,
    handling missing frames by interpolation.
    """
    os.makedirs(output_dir, exist_ok=True)

    with open(json_path, 'r') as f:
        pose_data = json.load(f)

    # Get total number of frames (detect missing ones)
    all_frames = [frame_data['frame'] for frame_data in pose_data]
    num_frames = max(all_frames) + 1 +16
    print(num_frames)

    num_keypoints = 17  # COCO format has 17 keypoints per person
    keypoints_coco = np.full((1, num_frames, num_keypoints, 2), np.nan)
    scores_coco = np.full((1, num_frames, num_keypoints), np.nan)

    for frame_data in pose_data:
        frame_idx = frame_data['frame']

        person_0_data = next((p for p in frame_data['poses'] if p['person'] == 0), None)
        if person_0_data:
            for kp in person_0_data['keypoints']:
                kp_idx = get_keypoint_index(kp['name'])
                keypoints_coco[0, frame_idx, kp_idx, 0] = kp['x']
                keypoints_coco[0, frame_idx, kp_idx, 1] = kp['y']
                scores_coco[0, frame_idx, kp_idx] = kp['score']

    # Interpolate missing frames
    for kp in range(num_keypoints):
        valid_indices = ~np.isnan(keypoints_coco[0, :, kp, 0])
        if np.sum(valid_indices) > 1:  # Ensure interpolation is possible
            keypoints_coco[0, :, kp, 0] = np.interp(
                np.arange(num_frames),
                np.where(valid_indices)[0],
                keypoints_coco[0, valid_indices, kp, 0]
            )
            keypoints_coco[0, :, kp, 1] = np.interp(
                np.arange(num_frames),
                np.where(valid_indices)[0],
                keypoints_coco[0, valid_indices, kp, 1]
            )
            scores_coco[0, :, kp] = np.interp(
                np.arange(num_frames),
                np.where(valid_indices)[0],
                scores_coco[0, valid_indices, kp]
            )
        else:
            keypoints_coco[0, :, kp, :] = 0  # Fill with zero if no valid data

    # Convert to H36M format
    h36m_kpts, h36m_scores, valid_frames = h36m_coco_format(keypoints_coco, scores_coco)
    final_h36m_kpts = revise_kpts(h36m_kpts, h36m_scores, valid_frames)

    output_path = os.path.join(output_dir, 'keypoints.npz')
    np.savez_compressed(output_path, reconstruction=final_h36m_kpts)

    print(f"Fixed missing frames. NPZ saved at {output_path}, Shape: {final_h36m_kpts.shape}")
    return output_path

def get_keypoint_index(keypoint_name):
    """
    Maps keypoint name to index in COCO format

    Args:
        keypoint_name (str): Name of the keypoint

    Returns:
        int: Index of the keypoint in the COCO format
    """
    keypoint_map = {
        "Nose": 0,
        "L_Eye": 1,
        "R_Eye": 2,
        "L_Ear": 3,
        "R_Ear": 4,
        "L_Shoulder": 5,
        "R_Shoulder": 6,
        "L_Elbow": 7,
        "R_Elbow": 8,
        "L_Wrist": 9,
        "R_Wrist": 10,
        "L_Hip": 11,
        "R_Hip": 12,
        "L_Knee": 13,
        "R_Knee": 14,
        "L_Ankle": 15,
        "R_Ankle": 16
    }

    return keypoint_map[keypoint_name]

def coco_h36m(keypoints):
    temporal = keypoints.shape[0]
    keypoints_h36m = np.zeros_like(keypoints, dtype=np.float32)
    htps_keypoints = np.zeros((temporal, 4, 2), dtype=np.float32)

    # htps_keypoints: head, thorax, pelvis, spine
    htps_keypoints[:, 0, 0] = np.mean(keypoints[:, 1:5, 0], axis=1, dtype=np.float32)
    htps_keypoints[:, 0, 1] = np.sum(keypoints[:, 1:3, 1], axis=1, dtype=np.float32) - keypoints[:, 0, 1]
    htps_keypoints[:, 1, :] = np.mean(keypoints[:, 5:7, :], axis=1, dtype=np.float32)
    htps_keypoints[:, 1, :] += (keypoints[:, 0, :] - htps_keypoints[:, 1, :]) / 3

    htps_keypoints[:, 2, :] = np.mean(keypoints[:, 11:13, :], axis=1, dtype=np.float32)
    htps_keypoints[:, 3, :] = np.mean(keypoints[:, [5, 6, 11, 12], :], axis=1, dtype=np.float32)

    keypoints_h36m[:, spple_keypoints, :] = htps_keypoints
    keypoints_h36m[:, h36m_coco_order, :] = keypoints[:, coco_order, :]

    keypoints_h36m[:, 9, :] -= (keypoints_h36m[:, 9, :] - np.mean(keypoints[:, 5:7, :], axis=1, dtype=np.float32)) / 4
    keypoints_h36m[:, 7, 0] += 2*(keypoints_h36m[:, 7, 0] - np.mean(keypoints_h36m[:, [0, 8], 0], axis=1, dtype=np.float32))
    keypoints_h36m[:, 8, 1] -= (np.mean(keypoints[:, 1:3, 1], axis=1, dtype=np.float32) - keypoints[:, 0, 1])*2/3

    valid_frames = np.where(np.sum(keypoints_h36m.reshape(-1, 34), axis=1) != 0)[0]

    return keypoints_h36m, valid_frames

def h36m_coco_format(keypoints, scores):
    assert len(keypoints.shape) == 4 and len(scores.shape) == 3

    h36m_kpts = []
    h36m_scores = []
    valid_frames = []

    for i in range(keypoints.shape[0]):
        kpts = keypoints[i]
        score = scores[i]

        new_score = np.zeros_like(score, dtype=np.float32)

        if np.sum(kpts) != 0.:
            kpts, valid_frame = coco_h36m(kpts)
            h36m_kpts.append(kpts)
            valid_frames.append(valid_frame)

            new_score[:, h36m_coco_order] = score[:, coco_order]
            new_score[:, 0] = np.mean(score[:, [11, 12]], axis=1, dtype=np.float32)
            new_score[:, 8] = np.mean(score[:, [5, 6]], axis=1, dtype=np.float32)
            new_score[:, 7] = np.mean(new_score[:, [0, 8]], axis=1, dtype=np.float32)
            new_score[:, 10] = np.mean(score[:, [1, 2, 3, 4]], axis=1, dtype=np.float32)

            h36m_scores.append(new_score)

    h36m_kpts = np.asarray(h36m_kpts, dtype=np.float32)
    h36m_scores = np.asarray(h36m_scores, dtype=np.float32)

    return h36m_kpts, h36m_scores, valid_frames

def revise_kpts(h36m_kpts, h36m_scores, valid_frames):
    new_h36m_kpts = np.zeros_like(h36m_kpts)
    for index, frames in enumerate(valid_frames):
        kpts = h36m_kpts[index, frames]
        score = h36m_scores[index, frames]

        index_frame = np.where(np.sum(score < 0.3, axis=1) > 0)[0]

        for frame in index_frame:
            less_threshold_joints = np.where(score[frame] < 0.3)[0]

            intersect = [i for i in [2, 3, 5, 6] if i in less_threshold_joints]

            if [2, 3, 5, 6] == intersect:
                kpts[frame, [2, 3, 5, 6]] = kpts[frame, [1, 1, 4, 4]]
            elif [2, 3, 6] == intersect:
                kpts[frame, [2, 3, 6]] = kpts[frame, [1, 1, 5]]
            elif [3, 5, 6] == intersect:
                kpts[frame, [3, 5, 6]] = kpts[frame, [2, 4, 4]]
            elif [3, 6] == intersect:
                kpts[frame, [3, 6]] = kpts[frame, [2, 5]]
            elif [3] == intersect:
                kpts[frame, 3] = kpts[frame, 2]
            elif [6] == intersect:
                kpts[frame, 6] = kpts[frame, 5]
            else:
                continue

        new_h36m_kpts[index, frames] = kpts

    return new_h36m_kpts
