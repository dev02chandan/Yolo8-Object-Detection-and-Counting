import streamlit as st
from shutil import copyfile
from object_counting import process_video_and_count, process_image_and_count
import tempfile
import os

# App title
st.image("videos/logo.png", use_column_width=False)
st.title('Object Detection and Counting')

# Option to select input type
input_type = st.radio("Select the input type:", ("Video", "Image"))

# File uploader based on input type
if input_type == "Video":
    uploaded_file = st.file_uploader("Upload a mp4 video...", type=["mp4"])
    file_type = "video"
elif input_type == "Image":
    uploaded_file = st.file_uploader("Upload an image...", type=["jpg", "png"])
    file_type = "image"

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
    default= ["cup", "cutter", "fork", "knife", "painting", "pan", "plant", "plate", 'scissor', 'spoon']
)

# Convert class names to class IDs
class_ids = [classNames.index(cls) for cls in selected_classes if cls in classNames]

if uploaded_file is not None and len(selected_classes) > 0:
    with st.spinner('Processing...'):
        # Save uploaded file to a temporary file
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix=f'.{uploaded_file.type.split("/")[-1]}')
        tfile.write(uploaded_file.getvalue())
        file_path = tfile.name

        # Process file based on type
        run_dir = "runs/temp"
        os.makedirs(run_dir, exist_ok=True)

        if file_type == "video":
            object_counts, output_path = process_video_and_count(file_path, 'best_ver4.pt', class_ids, run_dir)
            st.video(output_path)
        elif file_type == "image":
            object_counts, output_path = process_image_and_count(file_path, 'best_ver4.pt', class_ids, run_dir)
            st.image(output_path)

        # Display object counts
        st.write(f"Object counts: {object_counts}")
