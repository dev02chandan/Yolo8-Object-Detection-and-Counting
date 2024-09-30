import cv2
import os
import tempfile
from process_image import process_image_and_count
import streamlit as st
from collections import defaultdict


def process_video_and_count(
    video_path,
    model_path,
    classes_to_count,
    run_dir,
    iou=0.6,
    conf=0.2,
    imgsz=640,
    vid_stride=1,
    augment=False,
    device="cpu",
):
    """
    Process the video to count objects, extract one frame per second,
    and use process_image_and_count on each frame.
    Display each frame and its corresponding count.
    Return the maximum object count found across all frames.
    """

    # Ensure run directory exists
    os.makedirs(run_dir, exist_ok=True)

    # Open the video file
    video_cap = cv2.VideoCapture(video_path)

    if not video_cap.isOpened():
        raise FileNotFoundError(f"Could not open video file: {video_path}")

    # Get the video frame rate (fps)
    fps = video_cap.get(cv2.CAP_PROP_FPS)
    frame_count = 0
    max_object_count = 0

    # Store counts for all frames
    all_frame_counts = []

    # Read frames from the video one at a time
    while True:
        # Move to the next frame (1 frame per second)
        video_cap.set(cv2.CAP_PROP_POS_FRAMES, frame_count * fps * vid_stride)
        ret, frame = video_cap.read()

        if not ret:
            break  # Exit loop if there are no more frames to process

        frame_count += 1

        # Save frame to a temporary file
        temp_frame_file = tempfile.NamedTemporaryFile(delete=False, suffix=".jpg")
        cv2.imwrite(temp_frame_file.name, frame)

        # Process this frame using the process_image_and_count function
        with st.spinner(f"Processing frame {frame_count}..."):
            object_counts, annotated_frame_path = process_image_and_count(
                temp_frame_file.name,
                model_path,
                classes_to_count,
                run_dir,
                iou=iou,
                conf=conf,
                imgsz=imgsz,
                augment=augment,
                device=device,
            )

        # Display the frame and its count in Streamlit
        st.image(
            annotated_frame_path,
            caption=f"Frame {frame_count} - Counts: {object_counts}",
        )

        # Track the maximum object count for this frame
        frame_total_count = sum(object_counts.values())
        all_frame_counts.append(frame_total_count)
        max_object_count = max(max_object_count, frame_total_count)

        # Clean up temp frame file
        temp_frame_file.close()

    # Release the video capture object
    video_cap.release()

    # Return the maximum object count found across all frames
    st.write(f"Maximum object count found in a single frame: {max_object_count}")
    return max_object_count
