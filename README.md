# Sheep Detection and Counting

This project uses a YOLO (You Only Look Once) model for detecting and counting sheep in a video. It also tracks the sheep's movement and counts them as they cross a defined line in the frame.

## Table of Contents
- [Features](#features)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Usage](#usage)
- [Configuration](#configuration)
- [Acknowledgments](#acknowledgments)

## Features
- Detects sheep in a video using the YOLOv8 model.
- Tracks sheep movement across frames.
- Counts the number of sheep crossing a defined line.
- Displays video with annotated detections and counts.

## Prerequisites
Make sure you have the following installed:
- Python 3.8 or later
- `pip` package installer

## Installation
1. **Clone the repository:**
    ```bash
    git clone https://github.com/your-username/sheep-detection.git
    cd sheep-detection
    ```

2. **Install required packages:**
    ```bash
    pip install -r requirements.txt
    ```

## Usage
1. **Prepare your environment:**
    - Place your YOLOv8 weights file in the `weights` directory.
    - Place the input video in the `videos` directory and name it `sheep.mp4`.
    - Ensure the `sheep_mask.png` and `sheep_counter_bg.png` images are in the `images` directory.

2. **Run the script:**
    ```bash
    python main.py
    ```

3. **View the output:**
    - The processed video with detections will be saved as `output.mp4` in the `videos` directory.
    - During processing, the script will display the video with annotated detections and count overlays.

## Configuration
- **Model and Device:**
    - The YOLO model weights are expected to be located at `./weights/yolov8x.pt`.
    - The script automatically selects the device (`mps` if available, otherwise `cpu`).

- **Video Input/Output:**
    - Input video path: `./videos/sheep.mp4`
    - Output video path: `./videos/output.mp4`
    - Video resolution: `1280x720`

- **Counter Line:**
    - The coordinates of the line for counting sheep crossing: `[430, 567, 1276, 206]`.

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments
- **[YOLOv8 by Ultralytics](https://github.com/autogyro/yolo-V8)**
- **[SORT: Simple Online and Realtime Tracking](https://github.com/abewley/sort)**
- **[cvzone](https://github.com/cvzone/cvzone)**

This project is inspired by the need to track and count animals for various applications in agriculture and wildlife monitoring.

---

*Note: Customize the repository URL, paths, and any other project-specific details as needed.*
