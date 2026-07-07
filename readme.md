# Cricket Analytics Pipeline 🏏

> **⚠️ Patent Pending — All Rights Reserved.**
> This repository is public for **portfolio/demonstration purposes only**.
> No part of this code may be used, copied, modified, or redistributed
> without written permission. See [LICENSE](./LICENSE) and [NOTICE.md](./NOTICE.md).

A modular and configurable video analytics pipeline for cricket, including
rally segmentation, player detection, pose estimation, and 3D animation.

---

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
│       │                       #   pose_estimationv1.py, pose_estimationv1collab.py
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
├── NOTICE.md               # Patent-pending notice
├── .gitignore
└── README.md
```

> Note: `videos/`, `results/`, and downloaded model checkpoints are
> gitignored — they're generated/downloaded locally, not tracked in the repo.

---

## 🚀 Getting Started

### Download pretrained model for pose estimation

The pretrained model can be found [here](https://drive.google.com/file/d/1kJzGsorMPydMSKOZzFKNYXGWjpmIB4ge/view?usp=sharing). Download it and place it in `models/ViTPose/`.

```bash
# Install dependencies
pip install -r requirements.txt
```

```bash
# Run the pipeline
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
| `pose/pose_estimationv1collab.py` | Pose estimation pipeline (Colab variant) |
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

## License & Patent Notice

This project is **not open source**. It is shared publicly for portfolio
and demonstration purposes only, and one or more components are subject
to a pending patent application. See [LICENSE](./LICENSE) and
[NOTICE.md](./NOTICE.md) for full terms. For collaboration or licensing
inquiries, please contact the author directly.
