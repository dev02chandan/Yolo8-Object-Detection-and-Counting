import cv2
import json
import os
import tempfile
import shutil
from collections import defaultdict
from ultralytics import YOLO
import streamlit as st

classNames = ["diamond"]


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
        temp = i.split("_")
        final_obj_list.append(temp[0])
    return count_objects(final_obj_list)


def process_image_and_count(
    image_path,
    model_path,
    classes_to_count,
    run_dir,
    iou=0.6,
    conf=0.2,
    imgsz=640,
    augment=False,
    device="cpu",
):
    """
    Process the image to count objects, draw bounding boxes around detected objects (without annotations),
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
    results = model.track(
        image,
        classes=classes_to_count,
        persist=True,
        conf=conf,
        iou=iou,
        imgsz=imgsz,
        augment=augment,
        stream=False,
        device=device,
        half=(device != "cpu"),
    )

    object_counts = defaultdict(int)
    Final_obj = set()
    tracked_objects = defaultdict(lambda: defaultdict(int))

    # Loop through results and draw bounding boxes on the image
    for r in results:
        for box in r.boxes:
            if box.id is not None and box.cls[0] in classes_to_count:
                track_id = box.id.int().tolist()[0]
                class_id = int(box.cls[0])
                class_name = classNames[class_id]

                # Count the number of frames each class is detected for this track_id
                tracked_objects[track_id][class_name] += 1

                # Extract the bounding box coordinates
                xyxy = box.xyxy[0].tolist()  # [xmin, ymin, xmax, ymax]
                xmin, ymin, xmax, ymax = map(int, xyxy)

                # Draw a bounding box on the image (without annotation)
                color = (0, 255, 0)  # Green for bounding box
                thickness = 2
                cv2.rectangle(image, (xmin, ymin), (xmax, ymax), color, thickness)

    # Determine the most frequently detected class for each track_id
    for track_id, classes in tracked_objects.items():
        most_common_class = max(classes, key=classes.get)
        Final_obj.add(most_common_class + "_" + str(track_id))

    # Determine output image path
    output_image_path = os.path.join(run_dir, "output_image.jpg")

    # Save the image with bounding boxes
    cv2.imwrite(output_image_path, image)

    # Save object counts to a JSON file
    json_path = os.path.join(run_dir, "object_counts.json")
    with open(json_path, "w") as f:
        json.dump(list(count(Final_obj)), f, indent=4)

    return count(Final_obj), output_image_path
