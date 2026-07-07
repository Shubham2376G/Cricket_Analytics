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
<div align="center">

<table>
  <tr>
    <td align="center" colspan="2">
      <b>Player Detection</b><br><br>
      <img src="assets/detection.gif" width="400">
    </td>
  </tr>
  <tr>
    <td align="center" width="100%">
      <b>3D Animation</b><br><br>
      <img src="assets/bowler_3d.gif" width="400">
    </td>
      </tr>
  <tr>
    <td align="center" width="100%">
      <b>Simulation</b><br><br>
      <img src="assets/simulation.gif" width="400">
    </td>
  </tr>
</table>

</div>
 
---
 
## ⚙️ Pipeline Overview
 
The `main.py` entry point runs a configurable set of stages via flags:
 
```bash
python main.py --input demo/sample.mp4 --rally_segment --player_det --pose_est --animate
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


## 🤝 Let's Connect
 
If you'd like to discuss this project, potential collaborations, or
licensing - feel free to reach out.
 
<p>
  <a href="https://www.linkedin.com/in/shubham-aggarwal-a63b40276">
    <img src="https://img.shields.io/badge/Follow%20on-LinkedIn-0A66C2?style=for-the-badge&logo=linkedin&logoColor=white" />
  </a>
</p>
