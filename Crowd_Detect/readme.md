````markdown
# Crowd Detection using YOLOv8 and DBSCAN

This project detects and tracks crowds in videos using a YOLOv8 object detection model combined with DBSCAN clustering. It identifies "crowd events" when multiple people are found in close proximity over consecutive frames and logs these events to a CSV file.

---

## 📌 Features

- Person detection using YOLOv8
- Crowd detection using DBSCAN clustering
- Logs frame-wise and crowd-event-wise data
- Outputs an annotated video with bounding boxes and crowd highlights
- Configurable via command-line arguments

---

## 📦 Requirements

Install dependencies using:

```bash
pip install -r requirements.txt
````

---

## 🚀 Usage

Run the script using:

```bash
python detect.py --video dataset_video.mp4 --output output.mp4 --results_csv frame_detected.csv
```

### Optional Arguments

| Argument        | Description                                                           | Default               |
| --------------- | --------------------------------------------------------------------- | --------------------- |
| `--video`       | **(Required)** Path to input video file                               | —                     |
| `--output`      | Path to save output annotated video                                   | `None`                |
| `--results_csv` | Path to save per-frame detection results                              | `frame_results.csv`   |
| `--conf`        | Confidence threshold for YOLOv8 detection                             | `0.5`                 |
| `--nms`         | Non-maximum suppression threshold                                     | `0.4`                 |
| `--min_size`    | Minimum number of persons in proximity to be considered a crowd       | `3`                   |
| `--min_frames`  | Minimum consecutive frames a cluster must persist to count as a crowd | `10`                  |
| `--proximity`   | Maximum distance (in pixels) between people in a crowd                | `100`                 |
| `--model`       | Path to custom YOLOv8 model (.pt file)                                | `None` (uses yolov8n) |

---

## 📝 Output

* **Annotated video** (if `--output` is provided)
* `crowd_events.csv`: Records detected crowd events with frame number, person count, and timestamp
* `frame_results.csv`: Frame-wise count of detected persons

---

## 🧠 How It Works

1. **Detection**: YOLOv8 detects all persons (class 0) in a frame.
2. **Clustering**: DBSCAN clusters people based on spatial proximity.
3. **Tracking**: If a cluster persists for `min_frames`, it is marked as a crowd.
4. **Logging**: Crowd events and frame data are saved to CSV files.

---

## 📂 Example

```bash
python detect.py --video dataset_video.mp4 --output output.mp4 --results_csv frame_detected.csv
```

---

## 📸 Visualization

* Red rectangles: Individual detected persons
* Person Count in the far top left corner

---

## 📁 File Structure

```
.
├── detect.py
├── README.md
├── requirements.txt
├── frame_detected.csv (This file is returned as output when passed in the terminal)
├── dataset_video.mp4
└── output.mp4
```

---

## 🧩 Dependencies

* OpenCV
* NumPy
* pandas
* scikit-learn
* ultralytics

````



