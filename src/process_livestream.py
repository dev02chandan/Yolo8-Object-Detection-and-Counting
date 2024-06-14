import cv2
import time
from collections import defaultdict
from ultralytics import YOLO
import streamlit as st
from tempfile import NamedTemporaryFile
import os
import json
from process_video import count, count_objects

classNames = ['cup', 'cutter', 'fork', 'knife', 'painting', 'pan', 'plant', 'plate', 'scissor', 'spoon']

def process_livestream_and_count(model_path, classes_to_count, run_dir, iou=0.6, conf=0.6, imgsz=1280, tracker="botsort.yaml", device='cpu'):
    model = YOLO(model_path).to(device)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        st.error("Failed to open webcam.")
        return

    output_video_path = os.path.join(run_dir, "livestream_output.mp4")

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'avc1')
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    with NamedTemporaryFile(delete=False, suffix='.mp4', dir=run_dir) as temp_video_file:
        output_video_path = temp_video_file.name

    out = cv2.VideoWriter(output_video_path, fourcc, 30.0, (frame_width, frame_height))

    tracked_objects = defaultdict(lambda: defaultdict(int))

    stframe = st.empty()
    if 'stop_button' not in st.session_state:
        st.session_state['stop_button'] = False

    def stop_button_callback():
        st.session_state['stop_button'] = True

    st.button('Stop Live Stream', on_click=stop_button_callback, key='unique_stop_button')

    while not st.session_state['stop_button']:
        ret, frame = cap.read()
        if not ret:
            break

        results = model.track(frame, classes=classes_to_count, persist=True, tracker=tracker, conf=conf, iou=iou, imgsz=imgsz, device=device, half=(device != 'cpu'), stream_buffer=False)

        for r in results:
            annotated_frame = r.plot()
            stframe.image(annotated_frame, channels="BGR")

            for box in r.boxes:
                if box.id is not None and int(box.cls[0]) in classes_to_count:
                    track_id = box.id.int().tolist()[0]
                    class_id = int(box.cls[0])
                    class_name = classNames[class_id]

                    # Log detected class and track ID
                    st.write(f"Detected: {class_name} with ID: {track_id}")

                    # Ensure the tracked_objects dictionary is updated
                    if class_name not in tracked_objects[track_id]:
                        tracked_objects[track_id][class_name] = 0
                    tracked_objects[track_id][class_name] += 1

                    # Additional logging to debug
                    st.write(f"Tracked objects (updated): {tracked_objects}")

        out.write(annotated_frame)

    cap.release()
    out.release()

    # Determine most frequent class per track
    Final_obj = set()
    for track_id, classes in tracked_objects.items():
        most_common_class = max(classes, key=classes.get)
        Final_obj.add(most_common_class + '_' + str(track_id))

    json_path = os.path.join(run_dir, "object_counts.json")
    with open(json_path, 'w') as f:
        final_counts = count(Final_obj)
        json.dump(final_counts, f, indent=4)

    # Log final counts for debugging
    st.write(f"Tracked objects: {tracked_objects}")
    st.write(f"Final object counts: {final_counts}")

    st.write(f"Object counts: {final_counts}")

    return final_counts, output_video_path
