# Cricket Analytics Pipeline 🏏


## 📂 Project Structure

```
Cricket_Analytics/
│
├── src/
│   └── cricket_analytics/
│       ├── segmentation/       # Rally segmentation (segment.py)
│       ├── detection/          # Player/ball/object detection
│       │                       #   detection.py, object_tracker.py,
│       │                       #   ball_detect_dynamic.py, ball_bounce_detector.py,
│       │                       #   contour_detector.py, hsv_filter.py,
│       │                       #   background_subtractor.py
│       ├── pose/               # Pose estimation
│       │                       #   pose_estimationv1.py
│       ├── animation/          # 3D animation & plotting
│       │                       #   animation_plotly.py, full_animation.py
│       └── bowling/            # Bowling angle analysis
│                               #   bowl_angle_LH.py, bowl_angle_RH.py
│
├── args/                   # Argparser logic
├── assets/                 # Helper data / assets
├── dataset/                # Dataset storage or loading logic
├── demo/                   # Input and output folder for testing
├── models/                 # Model checkpoints and architecture files
├── notebooks/              # Jupyter notebooks
│
├── main.py                 # Entry point (modularized pipeline)
├── requirements.txt        # Python dependencies
├── LICENSE                 # All-rights-reserved license
├── .gitignore
└── README.md
```
---

## 🎬 Results
 
> Pretrained model weights are **not distributed** in this repository
> (patent-pending components). The clips/images below showcase the
> pipeline's output on sample footage.
 
| Stage | Output |
|---|---|
| Rally Segmentation | *[add a short clip/gif showing rally vs. non-rally classification]* |
| Player Detection | *[add a frame/clip with bowler + batsman bounding boxes]* |
| Pose Estimation | *[add a frame/clip with 2D keypoints overlaid]* |
| 3D Animation | *[add a frame/gif of the 3D pose animation]* |
 
 
---
 
## ⚙️ Pipeline Overview
 
The `main.py` entry point runs a configurable set of stages via flags:
 
```bash
python main.py --input demo/cric.mp4 --rally_segment --player_det --pose_est --animate
```

### Arguments

| Flag | Description |
|---|---|
| `--input` | Path to the input video |
| `--rally_segment` | Perform rally segmentation and classify clips into rally / non-rally |
| `--player_det` | Detect bowler and batsman; outputs video + keypoints JSON |
| `--pose_est` | Perform pose estimation on detected player clips; outputs JSON + NPZ for 3D estimation |
| `--animate` | Generate 3D pose predictions and save animation frames |

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
- `get_pose3D(video_path, keypoints_npz_path, output_dir)`

---

## 🧰 Module Reference

| File | Description |
|---|---|
| `segmentation/segment.py` | Rally segmentation logic |
| `detection/detection.py` | General detection utilities |
| `detection/object_tracker.py` | Object tracking logic |
| `detection/ball_detect_dynamic.py` | Dynamic ball detection logic |
| `detection/ball_bounce_detector.py` | Detects ball bounce frames |
| `detection/contour_detector.py` | Contour-based object detection |
| `detection/hsv_filter.py` | HSV filter tuning utility |
| `detection/background_subtractor.py` | Background subtraction utility |
| `pose/pose_estimationv1.py` | Pose estimation pipeline |
| `animation/animation_plotly.py` | 3D animation plotting using Plotly |
| `animation/full_animation.py` | Full 3D animation visualization |
| `bowling/bowl_angle_LH.py` | Bowling angle analysis (left-handed) |
| `bowling/bowl_angle_RH.py` | Bowling angle analysis (right-handed) |

---

## 🏁 Example

```bash
python main.py --input demo/sample_match.mp4 --rally_segment --pose_est
```

---


