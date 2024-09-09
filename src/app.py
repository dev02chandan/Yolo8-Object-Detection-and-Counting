import streamlit as st
from shutil import copyfile
from process_video import process_video_and_count
from process_image import process_image_and_count
from process_livestream import process_livestream_and_count, VideoProcessor
import tempfile
import os
import time
import torch
from collections import defaultdict
from streamlit_webrtc import webrtc_streamer, WebRtcMode

# App title
st.image("logo.png", use_column_width=False)
st.title('Fire and Smoke Detection')

# Option to select input type
input_type = st.radio("Select the input type:", ("Video", "Image"))

# File uploader based on input type
uploaded_file = None
if input_type == "Video":
    uploaded_file = st.file_uploader("Upload a mp4 video...", type=["mp4"])
    file_type = "video"
else:
    uploaded_file = st.file_uploader("Upload an image...", type=["jpg", "png"])
    file_type = "image"

classNames = ['Fire', 'default', 'smoke']

# # Selection of objects to detect
# selected_classes = st.multiselect(
#     'Select object classes to count',
#     options=classNames,
#     default=['Fire', 'default', 'smoke']
# )
selected_classes = ['Fire']

# Convert class names to class IDs
class_ids = [classNames.index(cls) for cls in selected_classes if cls in classNames]

selected_model = 'Fire and Smoke Model - 20 epochs - 6k dataset.pt'

# Check if CUDA is available
cuda_available = torch.cuda.is_available()
device = 'cuda:0' if cuda_available else 'cpu'
half = cuda_available

# Set default values for parameters
iou = 0.6
conf = 0.25
imgsz = 600
vid_stride = 2
augment = False

# Initialize tracked objects
if "tracked_objects" not in st.session_state:
    st.session_state["tracked_objects"] = defaultdict(lambda: defaultdict(int))

if uploaded_file is not None and len(selected_classes) > 0 and selected_model:
    with st.spinner('Processing...'):
        if file_type == "video" or file_type == "image":
            # Save uploaded file to a temporary file
            tfile = tempfile.NamedTemporaryFile(delete=False, suffix=f'.{uploaded_file.type.split("/")[-1]}')
            tfile.write(uploaded_file.getvalue())
            file_path = tfile.name
        else:
            file_path = None  # Not needed for live stream

        run_dir = "runs/temp"
        os.makedirs(run_dir, exist_ok=True)

        # Process file based on type
        if file_type == "video":
            object_counts, output_path = process_video_and_count(file_path, selected_model, class_ids, run_dir, iou, conf, imgsz, tracker="botsort.yaml", vid_stride=vid_stride, device=device)
            st.session_state["tracked_objects"] = object_counts
            # st.write(f"Object counts: {dict(object_counts)}")

        elif file_type == "image":
            object_counts, output_path = process_image_and_count(file_path, selected_model, class_ids, run_dir, iou=iou, conf=conf, imgsz=imgsz, augment=augment, device=device)
            st.session_state["tracked_objects"] = object_counts
            st.image(output_path)
            # st.write(f"Object counts: {dict(object_counts)}")