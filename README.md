# Video Processing and Object Counting

This project includes scripts for processing videos and counting objects using the YOLO8v models. 

Here is a preview of how it works:


https://github.com/dev02chandan/Yolo8-Object-Detection-and-Counting/assets/73015720/7b04da0f-255a-4de9-9885-8c74d52f1e50

**Result:**
```bash
{
    "laptop": 2,
    "bottle": 2,
    "cell phone": 2,
    "chair": 1,
    "car": 1,
    "keyboard": 1
}
```



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

OR 

**Directly use the code in colab:**

https://colab.research.google.com/drive/18a8Y3gtt-byXX5jfdjhJ3w6GmhgxGoss?usp=sharing

## Observations on testing

1. Duplicate items were counted, due to reflections, or circular / shaky movement of camera.
2. Giving the exact classes that you want to count improves the results. 
3. Low confidence duplicate items are frequently detected.

## Acknowledgement

This code is contributed mainly by Prof. Kapil Rathor
