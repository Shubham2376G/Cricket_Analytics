# Cricket Analytics Pipeline

A video analytics pipeline for cricket that combines rally segmentation,
player detection, pose estimation, and 3D animation to break down player
biomechanics from raw match footage.

---

## 🎬 Results
The clips below showcase the pipeline's output on sample match footage.
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
 
At a high level, the pipeline processes raw match video through four stages:
 
1. **Rally Segmentation** - isolates active rally footage from dead time / replays / crowd shots
2. **Player Detection** - locates and tracks the bowler and batsman across frames
3. **Pose Estimation** - extracts 2D keypoints for the tracked players
4. **3D Animation** - lifts 2D keypoints to 3D and renders an interactive animation for biomechanical analysis (e.g. bowling arm angle)
Implementation details, model architecture, and code are not included in
this repository - see the **Results** section above for output samples,
or reach out below to discuss the approach in more depth.

---

## 🤝 Let's Connect
 
If you'd like to discuss this project, potential collaborations, or
licensing - feel free to reach out.
 
<p>
  <a href="https://www.linkedin.com/in/shubham-aggarwal-a63b40276">
    <img src="https://img.shields.io/badge/Follow%20on-LinkedIn-0A66C2?style=for-the-badge&logo=linkedin&logoColor=white" />
  </a>
</p>
