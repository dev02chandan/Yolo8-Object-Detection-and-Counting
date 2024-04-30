# Video Processing and Object Counting with Streamlit frontend

This project includes scripts for processing videos and counting objects using the YOLO8v models. 

## Preview


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

**Frontend:**

![app _ Streamlit - http___localhost_8501_](https://github.com/dev02chandan/Yolo8-Object-Detection-and-Counting/assets/73015720/71556e94-ef70-41dd-a95d-f61e123f47bc)


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

**Use the Frontend:**

```bash
streamlit run src/app.py
```
and go to the localhost link provided in the terminal.

OR 

**Run directly in terminal:** 

```bash
python src/main.py --video_path "videos/video1.mp4" --model_path "yolov8m.pt" --classes_to_count 39 67 63 56 2 66
```

This will process the video by reducing its frame rate, detecting, and counting objects, and then outputting the results in a new directory within **runs/** containing the count in a JSON file and the processed video.

OR 

**Directly use the code in colab:**

[Link to Notebook](https://colab.research.google.com/drive/18a8Y3gtt-byXX5jfdjhJ3w6GmhgxGoss?usp=sharing)

## Training on other objects using Roboflow


Follow the Roboflow notebook below, and train Yolov8 model on custom objects. 
Steps to create the dataset and labelling are also in the notebook below: 

[Link to Notebook](https://colab.research.google.com/github/roboflow-ai/notebooks/blob/main/notebooks/train-yolov8-object-detection-on-custom-dataset.ipynb#scrollTo=ovQgOj_xSNDg)

> **_NOTE:_**  When you train on new objects, the Yolo model will forget the old objects. Your dataset should include all the objects that you want to train the model for. This is called ***Catastrophic Forgetting***. Overcoming this is called ***Continual Learning*** and this area is still a field of research.

## Observations on testing

1. Duplicate items were counted, due to reflections, or circular / shaky movement of camera. Avoiding such things while taking videos can improve results.
2. Giving the exact classes that you want to count improves the results. 
3. Low confidence duplicate items are frequently detected. You can increase or decrease the confidence as per your requirements (line 66 in object_counting.py)

## Acknowledgement

Special Thanks to Prof. Kapil Rathore Sir.
