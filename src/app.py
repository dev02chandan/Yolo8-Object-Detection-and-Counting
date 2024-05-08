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

classNames = ['airplane', 'apple', 'backpack', 'banana', 'baseball bat', 'baseball glove', 'bear', 'bed', 'bench', 'bicycle', 'bird', 'boat', 'book', 'bottle', 'bowl', 'broccoli', 'bus', 'cake', 'car', 'carrot', 'cat', 'cell phone', 'chair', 'clock', 'couch', 'cow', 'cup', 'cutter', 'dining table', 'dog', 'donut', 'elephant', 'fire hydrant', 'fork', 'frisbee', 'giraffe', 'hair drier', 'handbag', 'horse', 'hot dog', 'keyboard', 'kite', 'knife', 'laptop', 'microwave', 'motorcycle', 'mouse', 'orange', 'oven', 'painting', 'pan', 'parking meter', 'person', 'pizza', 'plate', 'potted plant', 'pottedplant', 'refrigerator', 'remote', 'sandwich', 'scissors', 'sheep', 'sink', 'skateboard', 'skis', 'snowboard', 'spoon', 'sports ball', 'stop sign', 'suitcase', 'surfboard', 'teddy bear', 'tennis racket', 'tie', 'toaster', 'toilet', 'toothbrush', 'traffic light', 'train', 'truck', 'tv', 'umbrella', 'vase', 'wine glass', 'zebra']

# Selection of objects to detect
selected_classes = st.multiselect(
    'Select object classes to count',
    options=classNames,
    default= ["cup", "cutter", "fork", "knife", "painting", "pan", "pottedplant", "plate", 'scissors', 'spoon']
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
