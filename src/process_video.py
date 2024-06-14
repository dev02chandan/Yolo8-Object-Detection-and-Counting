import cv2
import json
import os
import tempfile
import shutil
from collections import defaultdict
from ultralytics import YOLO
import streamlit as st

classNames = ['cup', 'cutter', 'fork', 'knife', 'painting', 'pan', 'plant', 'plate', 'scissor', 'spoon']

def count_objects(list1):
    count = {}
    for obj in list1:
        if obj not in count:
            count[obj] = 0
        count[obj] += 1
    return count

def count(set_obj):
    final_obj_list = []
    Obj_list = list(set_obj)
    for i in Obj_list:
        temp = i.split('_')
        final_obj_list.append(temp[0])
    return count_objects(final_obj_list)

def concatenate_videos(temp_video_paths, output_path, frame_width, frame_height, fourcc):
    out = cv2.VideoWriter(output_path, fourcc, 30.0, (frame_width, frame_height))
    for temp_video_path in temp_video_paths:
        cap = cv2.VideoCapture(temp_video_path)
        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                break
            out.write(frame)
        cap.release()
    out.release()

def process_video_and_count(video_path, model_path, classes_to_count, run_dir, iou, conf, imgsz, tracker, vid_stride, device='cpu', chunk_size=10):
    """
    Process the video in chunks to count objects, draw bounding boxes,
    and display processed frames with a progress bar.

    Args:
        chunk_size (int): Number of frames to process in a chunk. Defaults to 10.

    Returns:
        dict: Object counts.
        str: Path to the output video.
    """

    model = YOLO(model_path).to(device)

    cap = cv2.VideoCapture(video_path)

    output_video_path = os.path.join(run_dir, "output_video.mp4")

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'avc1')

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    object_counts = defaultdict(int)
    Final_obj = set()
    tracked_objects = defaultdict(lambda: defaultdict(int))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    progress_bar = st.progress(0)  # Initialize progress bar

    frame_counter = 0
    temp_video_paths = []

    # Create a container to manage displayed content
    video_container = st.empty()

    # Create a directory for temporary files
    temp_dir = os.path.join(run_dir, "tempdel")
    os.makedirs(temp_dir, exist_ok=True)

    while cap.isOpened():
        frames = []
        for _ in range(chunk_size):
            success, frame = cap.read()
            if not success:
                break
            frames.append(frame)
        if not frames:
            break

        with tempfile.NamedTemporaryFile(dir=temp_dir, delete=False, suffix='.mp4') as temp_video_file:
            temp_video_path = temp_video_file.name
            temp_video_paths.append(temp_video_path)
            temp_out = cv2.VideoWriter(temp_video_path, fourcc, 30.0, (frame_width, frame_height))

            # Process the chunk
            results = model.track(frames, classes=classes_to_count, persist=True, tracker=tracker, conf=conf, iou=iou, imgsz=imgsz, vid_stride=vid_stride, device=device, half=(device != 'cpu'))

            # Process results and display
            for r, frame in zip(results, frames):
                annotated_frame = r.plot()
                temp_out.write(annotated_frame)  # Write annotated frame to temporary video

                for box in r.boxes:
                    if box.id is not None and box.cls[0] in classes_to_count:
                        track_id = box.id.int().tolist()[0]
                        class_id = int(box.cls[0])
                        class_name = classNames[class_id]

                        # Count objects
                        tracked_objects[track_id][class_name] += 1

            temp_out.release()

        # Concatenate and display the video up to the current chunk
        concatenate_videos(temp_video_paths, output_video_path, frame_width, frame_height, fourcc)
        with video_container:
            st.video(output_video_path)

        # Update progress bar
        frame_counter += len(frames)
        progress_bar.progress(min(frame_counter / total_frames, 1.0))  # Ensure progress bar reaches 100%

    cap.release()

    # Final concatenation of all chunks
    concatenate_videos(temp_video_paths, output_video_path, frame_width, frame_height, fourcc)

    # Display the final video
    with video_container:
        st.video(output_video_path)

    # Determine most frequent class per track
    for track_id, classes in tracked_objects.items():
        most_common_class = max(classes, key=classes.get)
        Final_obj.add(most_common_class + '_' + str(track_id))

    # Save object counts to JSON
    json_path = os.path.join(run_dir, "object_counts.json")
    with open(json_path, 'w') as f:
        json.dump(count(Final_obj), f, indent=4)

    # Complete progress bar
    progress_bar.progress(1.0)

    # Clean up temporary files
    shutil.rmtree(temp_dir)

    return count(Final_obj), output_video_path
