import streamlit as st
from shutil import copyfile
from object_counting import process_video_and_count, process_image_and_count
import tempfile
import os
import time

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

classNames = ['cup', 'cutter', 'fork', 'knife', 'painting', 'pan', 'plant', 'plate', 'scissor', 'spoon']

# Selection of objects to detect
selected_classes = st.multiselect(
    'Select object classes to count',
    options=classNames,
    default=['cup', 'fork', 'spoon', 'knife']
)

# Convert class names to class IDs
class_ids = [classNames.index(cls) for cls in selected_classes if cls in classNames]

# Model selection
model_options = ["50 epochs balanced.pt", "Transfer learning 50 epochs balanced.pt"]
selected_models = st.multiselect(
    'Select models to test',
    options=model_options,
    default=["50 epochs balanced.pt"]
)

if uploaded_file is not None and len(selected_classes) > 0 and len(selected_models) > 0:
    with st.spinner('Processing...'):
        # Save uploaded file to a temporary file
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix=f'.{uploaded_file.type.split("/")[-1]}')
        tfile.write(uploaded_file.getvalue())
        file_path = tfile.name

        # Process file based on type
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        
        results = []
        for model in selected_models:
            run_dir = f"runs/{model.split('.')[0]}_{timestamp}"
            os.makedirs(run_dir, exist_ok=True)
            
            if file_type == "video":
                object_counts, output_path = process_video_and_count(file_path, model, class_ids, run_dir)
                results.append((model, object_counts, output_path, "video"))
            elif file_type == "image":
                object_counts, output_path = process_image_and_count(file_path, model, class_ids, run_dir)
                results.append((model, object_counts, output_path, "image"))

        # Display results
        for model, object_counts, output_path, output_type in results:
            st.write(f"Results for model: {model}")
            st.write(f"Object counts: {object_counts}")
            if output_type == "video":
                st.video(output_path)
            elif output_type == "image":
                st.image(output_path)
