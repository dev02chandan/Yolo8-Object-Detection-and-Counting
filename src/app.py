import streamlit as st
from shutil import copyfile
from object_counting import process_video_and_count
import tempfile
import os

# App title
st.title('Object Detection and Counting with YOLOv8')



# Video file uploader
uploaded_video = st.file_uploader("Upload a mp4 video...", type=["mp4"])

# Global classNames for use across scripts
classNames = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
              "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
              "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
              "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
              "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
              "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
              "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed",
              "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone",
              "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
              "teddy bear", "hair drier", "toothbrush"
              ]

# Selection of objects to detect
selected_classes = st.multiselect(
    'Select object classes to count',

    options=classNames,

    default=["chair", "laptop"]
)

# Convert class names to class IDs
class_ids = [classNames.index(cls) for cls in selected_classes if cls in classNames]

if uploaded_video is not None and len(selected_classes) > 0:
    with st.spinner('Processing...'):

        # Save uploaded video to a temporary file
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')  # Ensure suffix matches file type
        tfile.write(uploaded_video.getvalue())  # Use getvalue() to read the content of the uploaded file
        video_path = tfile.name

        # Process video
        run_dir = "runs/temp"
        os.makedirs(run_dir, exist_ok=True)
        object_counts, output_video_path = process_video_and_count(video_path, 'yolov8m.pt', class_ids, run_dir)

        st.video(output_video_path)

        # Display object counts
        st.write(f"Object counts: {object_counts}")

