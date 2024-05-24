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
!git clone -b custom_10_classes https://github.com/dev02chandan/Yolo8-Object-Detection-and-Counting.git
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

**Directly use the code in colab with ngrok and streamlit:**

[Link to Notebook](https://colab.research.google.com/drive/12rv6tvAls7hzXeWPkVWCj9GrnIAr8P6_?usp=sharing)
(Requires an Ngrok token)

## Training on other objects using Roboflow

Follow the Roboflow notebook below, and train Yolov8 model on custom objects. 
Steps to create the dataset and labelling are also in the notebook below: 

[Link to Notebook](https://colab.research.google.com/github/roboflow-ai/notebooks/blob/main/notebooks/train-yolov8-object-detection-on-custom-dataset.ipynb#scrollTo=ovQgOj_xSNDg)

> **_NOTE:_**  When you train on new objects, the Yolo model will forget the old objects. Your dataset should include all the objects that you want to train the model for. This is called ***Catastrophic Forgetting***. Overcoming this is called ***Continual Learning*** and this area is still a field of research.

You can also try transfer learning on the Yolo Model by checking the following notebook:
[Link to Kaggle Notebook](https://www.kaggle.com/dev02chandan/transfer-learning-by-freezing-layers-yolov8)

## Object Counting Project Guidelines

To achieve the best results when using the object counting project, please follow these guidelines:

### 1. Proper Placement of Objects
- **Keep Objects Separate**: Ensure objects are not overlapping or touching each other. Each object should be clearly distinguishable.
- **Avoid Stacking**: Do not stack objects on top of each other. Place them in a single layer.

### 2. Lighting Conditions
- **Good Lighting**: Ensure the area is well-lit. Avoid shadows and dark areas.
- **Avoid Reflections**: Minimize reflections on surfaces, especially if objects are shiny or metallic.

### 3. Background and Environment
- **Neutral Background**: Use a plain and neutral-colored background to avoid distractions and false detections.
- **Clutter-Free**: Keep the background free of other objects that are not part of the count.

### 4. Camera Setup
- **Stable Camera**: Ensure the camera is stable and not shaking. Use a tripod if necessary.
- **Proper Angle**: Position the camera directly above the objects if possible. Avoid extreme angles that may obscure the view.

### 5. Object Condition
- **Visible and Intact**: Ensure that the objects are intact and clearly visible. Partial or damaged objects may not be detected correctly.
- **Standard Position**: Place objects in a standard, upright position if applicable (e.g., cups should not be upside down).

### 6. Avoiding Obstructions
- **Clear View**: Ensure no part of the objects is obscured by other items, hands, or any other obstruction during the counting process.

### 7. Handling Reflections
- **Matte Surfaces**: Use matte surfaces to place objects if possible, as they reduce reflections.
- **Anti-Reflective Coating**: Consider using anti-reflective coatings on surfaces to minimize glare.

### 8. Regular Calibration
- **Calibrate Regularly**: If your setup changes, re-calibrate the system to ensure continued accuracy. This includes changes in lighting, camera position, or background.

By adhering to these guidelines, you can significantly improve the accuracy and reliability of the object counting system. If you encounter any issues, reviewing these guidelines can help troubleshoot and resolve common problems.


## Acknowledgement

Special Thanks to Prof. Kapil Rathore Sir.
