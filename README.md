# ⚽ Football Analysis using YOLO

> Real-time player tracking, team assignment, ball possession, speed & distance estimation — all from a single match video.

---

## 📽️ Before vs After

### Raw Input
https://github.com/vintiw6/Football-Analysis-using-YOLO/blob/main/input_videos/08fd33_4.mp4

### Analysis Output
![Analyzed Output](output.gif)

---

## 🧠 What It Does

| Feature | Description |
|---------|-------------|
| 🎯 **Player & Ball Detection** | YOLOv5-based detection of players, referees, and the ball across every frame |
| 🎨 **Automatic Team Assignment** | KMeans pixel clustering on shirt colors to assign players to teams — no manual labeling |
| 📊 **Ball Possession Tracking** | Calculates each team's ball control % in real time throughout the match |
| 🎥 **Camera Movement Compensation** | Optical flow tracks camera pan/tilt so player positions stay accurate |
| 🗺️ **Perspective Transformation** | Warps the pitch view so distances are measured in real-world meters, not pixels |
| 💨 **Speed & Distance** | Per-player speed (km/h) and total distance covered, updated every 5 frames |

---

## 🖼️ Output Preview

![Analysis Screenshot](output_videos/screenshot.png)

---

## 🏗️ Architecture

```
input video
    │
    ▼
┌─────────────────┐
│  YOLOv5 Detect  │  ── players, referees, ball
└────────┬────────┘
         │
    ┌────▼────────────┐
    │  ByteTrack      │  ── persistent IDs across frames
    └────┬────────────┘
         │
    ┌────▼────────────┐     ┌──────────────────┐
    │  KMeans Cluster │ ──► │  Team Assignment │
    └────┬────────────┘     └──────────────────┘
         │
    ┌────▼────────────┐
    │  Optical Flow   │  ── camera movement correction
    └────┬────────────┘
         │
    ┌────▼────────────────────┐
    │  Perspective Transform  │  ── pixel → meters
    └────┬────────────────────┘
         │
    ┌────▼───────────────────────┐
    │  Speed & Distance Estimator│
    └────┬───────────────────────┘
         │
         ▼
   annotated output video
```

---

## 🚀 Quick Start

### 1. Clone & install

```bash
git clone https://github.com/vintiw6/Football-Analysis-using-YOLO
cd Football-Analysis-using-YOLO
python -m venv venv
venv\Scripts\activate        # Windows
pip install -r requirements.txt
```

### 2. Add your video

Drop an MP4 into `input_videos/` and update the path in `main.py`:

```python
video_frames = read_video('input_videos/your_video.mp4')
```

### 4. Run

```bash
python main.py
```

Output saved to `output_videos/`.

---

## 📦 Requirements

```
Python 3.x
ultralytics
supervision
opencv-python
numpy
matplotlib
pandas
scikit-learn
```

Install all at once:
```bash
pip install ultralytics supervision opencv-python numpy matplotlib pandas scikit-learn
```

---

## 📁 Project Structure

```
Football-Analysis-using-YOLO/
│
├── main.py                          # Entry point
├── models/
│   └── best.pt                      # Trained YOLO weights
│
├── input_videos/                    # Drop your footage here
├── output_videos/                   # Annotated output
├── stubs/                           # Cached detections (speeds up reruns)
│
├── trackers/
│   └── tracker.py                   # Detection + annotation logic
├── team_assigner/
│   └── team_assigner.py             # KMeans shirt color clustering
├── player_ball_assigner/
│   └── player_ball_assigner.py      # Ball possession assignment
├── camera_movement_estimator/
│   └── camera_movement_estimator.py # Optical flow camera tracking
├── view_transformer/
│   └── view_transformer.py          # Perspective warp
├── speed_and_distance_estimator/
│   └── speed_and_distance_estimator.py
└── utils/                           # Helper functions
```

---

## 🔧 Tips

- **Stub caching** — first run saves detections to `stubs/`. Delete `.pkl` files if you change the input video.
- **Confidence threshold** — edit `conf=0.35` in `tracker.py` to tune detection sensitivity.
- **GPU acceleration** — CUDA is supported automatically if PyTorch detects a GPU. Install the CUDA version of PyTorch for a significant speedup.

---

## 📚 Concepts Covered

- Object detection with YOLO
- Multi-object tracking with ByteTrack
- Unsupervised clustering (KMeans)
- Optical flow (Lucas-Kanade)
- Homographic perspective transformation
- Real-world metric estimation from video

---

## 🙏 Credits

Base project inspired by [Abdullah Tarek's football analysis tutorial](https://github.com/abdullahtarek/football_analysis).  
Extended with modernized annotations, GPU training pipeline, and dedicated ball detection model.
