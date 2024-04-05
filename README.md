# Video Processing and Object Counting

This project includes scripts for processing videos and counting objects using the YOLO8v models.

## Installation

### 1. Clone the Repository

```bash
git clone <repository-url>
```

### 2. Create a Virtual Environment

#### Windows

```bash
cd <repository-name>
python -m venv venv
.\venv\Scripts\activate
```

#### Mac/Linux

```bash
cd <repository-name>
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
python src/main.py --video_path "Videos/video1.mp4" --model_path "yolov8m.pt" --classes_to_count 0 1 2 3
```

This will process the video by reducing its frame rate, detecting, and counting objects, and then outputting the results in a new directory within **runs/** containing the count in a JSON file and the processed video.

