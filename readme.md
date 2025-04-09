# Cricket Analytics Pipeline 🏏

This repository contains a modular and configurable video analytics pipeline for cricket, including rally segmentation, player detection, pose estimation, and 3D animation.

---

## 📂 Project Structure

```
Cricket_Analytics/
│
├── args/                   # Argparser logic
├── assets/                 # Assets like helper data
├── dataset/                # Dataset storage or loading logic
├── demo/                   # Input and output folder for testing
├── models/                 # Model checkpoints and architecture files
├── notebooks/              # Exploratory notebooks
├── utils/                  # Utility scripts (detailed below)
├── videos/                 # Video storage folder
│
├── main.py                 # Entry point (modularized pipeline)
├── requirements.txt        # Python dependencies
└── readme.md               
```

---

## 🚀 Getting Started

```bash
# Install dependencies
pip install -r requirements.txt
```
```bash
# Run the pipeline
python main.py --input demo2/cric.mp4 --rally_segment --player_det --pose_est --animate
```

### Arguments:

- `--input`: Path to the input video.
- `--rally_segment`: Perform rally segmentation and classifies clips into rally and non rally.
- `--player_det`: Detects bowler and the batsman and create a output video file along with keypoints json file.
- `--pose_est`: Perform pose estimation on detected player videos and creates json file and npz file for 3D estimation.
- `--animate`: Performs the 3D pose predictions and saves frames.

---

## 🧠 Modules

### Rally Segmentation
- `segment_video(video_path, output_dir)`
- `classify_all_videos(folder_path)`

### Player Detection
- `detect(input_folder, output_folder)`

### Pose Estimation
- `pose_estimation(video_path, json_path, output_dir, label)`
- `convert_json_to_h36m_npz(json_path, output_dir, label)`

### Animation
- `get_pose3D(video_path, keypoints.npz_path, output_dir)`

---

## 🧰 `utils/` Directory

Descriptions for each file (edit as needed):

| Filename                     | Description                           |
|------------------------------|---------------------------------------|
| `animation_plotly.py`        | 3D animation plotting using Plotly    |
| `background_subtractor.py`   | Background subtraction utility        |
| `ball_bounce_detector.py`    | Detects ball bounce frames            |
| `ball_detect_dynamic.py`     | Dynamic ball detection logic          |
| `bowl_angle_LH.py`           | Bowling angle analysis (Left-handed)  |
| `bowl_angle_RH.py`           | Bowling angle analysis (Right-handed) |
| `contour_detector.py`        | Contour-based object detection        |
| `detection.py`               | General detection utilities           |
| `full_animation.py`          | Full 3D animation visualization       |
| `hsv_filter.py`              | HSV filter tuning utility             |
| `object_tracker.py`          | Object tracking logic                 |
| `pose_estimationv1.py`       | Pose estimation pipeline              |
| `pose_estimationv1collab.py` | Pose estimation for Colab             |
| `segment.py`                 | Rally segmentation logic              |

---

## 🚪 Example

```bash
python main.py --input demo/sample_match.mp4 --rally_segment --pose_est
```

---


