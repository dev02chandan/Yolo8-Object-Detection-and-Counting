# Video Processing and Object Counting

This project includes scripts for processing videos and counting objects using the YOLO8v models.

## Project Output Video

See the output of our object detection and counting process in action below:

<video src='runs/20240405_155550/output_video.mp4'/>

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/dev02chandan/Yolo8-Object-Detection-and-Counting.git
```

### 2. Create a Virtual Environment

#### Windows

```bash
cd Yolo8-Object-Detection-and-Counting
python -m venv venv
.\venv\Scripts\activate
```

#### Mac/Linux

```bash
cd Yolo8-Object-Detection-and-Counting
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

## Usage

To count objects in a video, run the following command:

```bash
python src/main.py --video_path "videos/video1.mp4" --model_path "yolov8m.pt" --classes_to_count 39 67 63 56 2 66
```

This will process the video by reducing its frame rate, detecting, and counting objects, and then outputting the results in a new directory within **runs/** containing the count in a JSON file and the processed video.


## Acknowledgement

This code is contributed mainly by Prof. Kapil Rathor
