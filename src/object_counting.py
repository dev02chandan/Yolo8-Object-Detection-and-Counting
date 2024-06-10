import cv2
import json
import os
import torch
import tempfile
import logging
from collections import defaultdict
from ultralytics import YOLO

# Set up logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

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

def process_video_and_count(video_path, model_path, classes_to_count, iou, conf, imgsz, run_dir, tracker, vid_stride, device='cpu'):
    """
    Process the video to count objects, draw bounding boxes around detected objects,
    and save an annotated video along with a JSON file containing the counts.
    """
    logging.debug(f"Processing video: {video_path}")
    logging.debug(f"Model path: {model_path}")
    logging.debug(f"Classes to count: {classes_to_count}")
    logging.debug(f"IOU: {iou}, Confidence: {conf}, Image size: {imgsz}")
    logging.debug(f"Run directory: {run_dir}")
    logging.debug(f"Tracker: {tracker}, Video stride: {vid_stride}, Device: {device}")

    # Ensure run directory exists
    if not os.path.exists(run_dir):
        try:
            os.makedirs(run_dir)
            logging.debug(f"Run directory created: {run_dir}")
        except Exception as e:
            logging.error(f"Failed to create run directory {run_dir}: {e}")
            raise

    # Load the YOLO model
    try:
        model = YOLO(model_path).to(device)
        logging.debug(f"Model loaded: {model_path} on device: {device}")
    except Exception as e:
        logging.error(f"Failed to load model {model_path}: {e}")
        raise

    # Open the video file
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video file {video_path}")
    logging.debug(f"Video file opened: {video_path}")

    # Use a temporary file for the output video
    with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4', dir=run_dir) as tmpfile:
        output_video_path = tmpfile.name
    logging.debug(f"Temporary output video path: {output_video_path}")

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'avc1')
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out = cv2.VideoWriter(output_video_path, fourcc, 30.0, (frame_width, frame_height))
    if not out.isOpened():
        raise IOError(f"Cannot open video writer with path {output_video_path}")
    logging.debug(f"VideoWriter opened: {output_video_path}")

    object_counts = defaultdict(int)
    Final_obj = set()
    tracked_objects = defaultdict(lambda: defaultdict(int))
    frame_counter = 0

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        logging.debug(f"Processing frame {frame_counter}")
        results = model.track(frame,
                              classes=classes_to_count,
                              persist=True,
                              tracker=tracker,
                              conf=conf,
                              iou=iou,
                              imgsz=imgsz,
                              half=(device != 'cpu'),
                              stream=True,
                              augment=True,
                              max_det=300,
                              vid_stride=vid_stride,
                              agnostic_nms=True,
                              device=device
                              )

        annotated_frames = []

        for r in results:
            for box in r.boxes:
                if box.id is not None and box.cls[0] in classes_to_count:
                    track_id = box.id.int().tolist()[0]
                    class_id = int(box.cls[0])
                    class_name = classNames[class_id]

                    # Count the number of frames each class is detected for this track_id
                    tracked_objects[track_id][class_name] += 1

            annotated_frame = r.plot()
            annotated_frames.append(annotated_frame)

        if annotated_frames:
            logging.debug(f"Writing frame {frame_counter}")
            out.write(annotated_frames[0])  # Ensure only one frame is written per loop iteration

        frame_counter += 1

    cap.release()
    out.release()

    # Determine the most frequently detected class for each track_id
    for track_id, classes in tracked_objects.items():
        most_common_class = max(classes, key=classes.get)
        Final_obj.add(most_common_class + '_' + str(track_id))

    # Save object counts to a JSON file
    json_path = os.path.join(run_dir, "object_counts.json")
    try:
        with open(json_path, 'w') as f:
            json.dump(count(Final_obj), f, indent=4)
        logging.debug(f"Object counts saved: {json_path}")
    except Exception as e:
        logging.error(f"Failed to save object counts to {json_path}: {e}")
        raise

    return count(Final_obj), output_video_path

def process_image_and_count(image_path, model_path, classes_to_count, run_dir, iou=0.6, conf=0.2, imgsz=640, augment=False, device='cpu'):
    """
    Process the image to count objects, draw bounding boxes around detected objects,
    and save an annotated image along with a JSON file containing the counts.
    """

    # Ensure run directory exists
    os.makedirs(run_dir, exist_ok=True)

    # Load the YOLO model
    model = YOLO(model_path).to(device)

    # Load the image file
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"No image found at {image_path}")

    # Process the image
    results = model.track(image, classes=classes_to_count, persist=True, conf=conf, iou=iou, imgsz=imgsz, augment=augment, stream=False, device=device, half=(device != 'cpu'))

    object_counts = defaultdict(int)
    Final_obj = set()
    tracked_objects = defaultdict(lambda: defaultdict(int))

    annotated_images = []

    for r in results:
        for box in r.boxes:
            if box.id is not None and box.cls[0] in classes_to_count:
                track_id = box.id.int().tolist()[0]
                class_id = int(box.cls[0])
                class_name = classNames[class_id]

                # Count the number of frames each class is detected for this track_id
                tracked_objects[track_id][class_name] += 1

        annotated_image = r.plot()
        annotated_images.append(annotated_image)

    # Determine the most frequently detected class for each track_id
    for track_id, classes in tracked_objects.items():
        most_common_class = max(classes, key=classes.get)
        Final_obj.add(most_common_class + '_' + str(track_id))

    # Determine output image path
    output_image_path = os.path.join(run_dir, "output_image.jpg")
    if annotated_images:
        # Save the annotated image
        cv2.imwrite(output_image_path, annotated_images[0])

    # Save object counts to a JSON file
    json_path = os.path.join(run_dir, "object_counts.json")
    with open(json_path, 'w') as f:
        json.dump(list(count(Final_obj)), f, indent=4)

    return count(Final_obj), output_image_path
