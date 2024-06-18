import cv2
from collections import defaultdict
from ultralytics import YOLO
import streamlit as st

classNames = ['cup', 'cutter', 'fork', 'knife', 'painting', 'pan', 'plant', 'plate', 'scissor', 'spoon']

def initialize_session_state():
    if 'tracked_objects' not in st.session_state:
        st.session_state['tracked_objects'] = defaultdict(lambda: defaultdict(int))
    if 'stop_button' not in st.session_state:
        st.session_state['stop_button'] = False

def reset_session_state():
    st.session_state['tracked_objects'] = defaultdict(lambda: defaultdict(int))
    st.session_state['stop_button'] = False

def process_livestream_and_count(model_path, classes_to_count, iou=0.6, conf=0.6, imgsz=1280, tracker="botsort.yaml", device='cpu'):
    initialize_session_state()
    model = YOLO(model_path).to(device)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        st.error("Failed to open webcam.")
        return

    stframe = st.empty()

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
                    print(f"Detected: {class_name} with ID: {track_id}")

                    # Update the session state's tracked_objects dictionary
                    st.session_state['tracked_objects'][track_id][class_name] += 1

                    # Additional logging to debug
                    print(f"Tracked objects (updated): {dict(st.session_state['tracked_objects'])}")

    cap.release()

    # Determine the most frequently detected class for each track_id
    final_counts = defaultdict(int)
    for track_id, classes in st.session_state['tracked_objects'].items():
        most_common_class = max(classes, key=classes.get)
        final_counts[most_common_class] += 1

    # Log final counts for debugging
    print(f"Final object counts: {dict(final_counts)}")

    st.write(f"Object counts: {dict(final_counts)}")

    return dict(final_counts)
