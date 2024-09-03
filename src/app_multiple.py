import streamlit as st
from process_image import process_image_and_count
import tempfile
import os
import torch

# App title
st.title('Object Counting in Multiple Images')

# Instructions
st.write("Upload multiple images, and the app will count selected objects in each image.")

# File uploader for multiple images
uploaded_files = st.file_uploader("Upload images...", type=["jpg", "png"], accept_multiple_files=True)

classNames = ['cup', 'cutter', 'fork', 'knife', 'painting', 'pan', 'plant', 'plate', 'scissor', 'spoon']

# Selection of objects to detect
selected_classes = st.multiselect(
    'Select object classes to count',
    options=classNames,
    default=['cup', 'cutter', 'fork', 'knife', 'plate', 'scissor', 'spoon']
)

# Convert class names to class IDs
class_ids = [classNames.index(cls) for cls in selected_classes if cls in classNames]

selected_model = '50_epochs_balanced.pt'

# Check if CUDA is available
cuda_available = torch.cuda.is_available()
device = 'cuda:0' if cuda_available else 'cpu'
half = cuda_available

# Set default values for parameters
iou = 0.6
conf = 0.6
imgsz = 1280
augment = True

# Process images if files are uploaded
if uploaded_files and len(selected_classes) > 0 and selected_model:
    st.write(f"Processing {len(uploaded_files)} image(s)...")
    
    for uploaded_file in uploaded_files:
        with st.spinner(f'Processing {uploaded_file.name}...'):
            # Save uploaded file to a temporary file
            tfile = tempfile.NamedTemporaryFile(delete=False, suffix=f'.{uploaded_file.type.split("/")[-1]}')
            tfile.write(uploaded_file.getvalue())
            file_path = tfile.name

            run_dir = "runs/temp"
            os.makedirs(run_dir, exist_ok=True)

            # Process the image and count objects
            object_counts, output_path = process_image_and_count(
                file_path, selected_model, class_ids, run_dir,
                iou=iou, conf=conf, imgsz=imgsz, augment=augment, device=device
            )

            # Display the results for each image
            st.write(f"Object counts in {uploaded_file.name}: {object_counts}")

            # Display the processed image
            st.image(output_path, caption=f"Processed Image: {uploaded_file.name}")

            # Optionally, you could save the output image permanently by moving it to a desired directory
            # e.g., copyfile(output_path, f"output_images/{uploaded_file.name}")

