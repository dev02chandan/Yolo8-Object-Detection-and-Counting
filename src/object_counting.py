import cv2
import json
import os
from collections import defaultdict
from ultralytics import YOLO


classNames = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
              "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
              "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
              "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
              "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
              "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
              "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed",
              "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone",
              "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
              "teddy bear", "hair drier", "toothbrush"
              ]

def count_objects(list1):
  count = {}
  for obj in list1:
    if obj not in count:
      count[obj] = 0
    count[obj] += 1
  return count

def count(set_obj):
    final_obj_list =[]
    Obj_list = list(set_obj)
    for i in Obj_list:
        temp = i.split('_')
        final_obj_list.append(temp[0])
    return count_objects(final_obj_list)

def process_video_and_count(video_path, model_path, classes_to_count, run_dir):
    """
    Process the video to count objects, draw bounding boxes around detected objects,
    and save an annotated video along with a JSON file containing the counts.
    """

    # Load the YOLO model
    model = YOLO(model_path)

    # Open the video file
    cap = cv2.VideoCapture(video_path)

    # Determine output video path
    output_video_path = os.path.join(run_dir, "output_video.mp4")

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'avc1')
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    out = cv2.VideoWriter(output_video_path, fourcc, 30.0, (frame_width, frame_height))

    object_counts = defaultdict(int)
    Final_obj = set()

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        # Process the frame
        results = model.track(frame, classes= classes_to_count,persist=True, tracker="botsort.yaml", conf=0.6, iou=0.6, stream=True)
        annotated_frames = []
        
        for r in results:
            for box in r.boxes:
                if box.id is not None and box.cls[0] in classes_to_count:
                    track_id = box.id.int().cpu().tolist()[0]
                    class_id = int(box.cls[0])
                    class_name = classNames[class_id]
                    Final_obj.add(classNames[class_id]+'_'+str(track_id))
                    
            annotated_frame = r.plot()
            annotated_frames.append(annotated_frame)
        
        # Write the frame with annotations to the output video
        out.write(annotated_frame)

    cap.release()
    out.release()

    # Save object counts to a JSON file
    json_path = os.path.join(run_dir, "object_counts.json")
    with open(json_path, 'w') as f:
        json.dump(count(Final_obj), f, indent=4)

    return count(Final_obj), output_video_path


def process_image_and_count(image_path, model_path, classes_to_count, run_dir):
    """
    Process the image to count objects, draw bounding boxes around detected objects,
    and save an annotated image along with a JSON file containing the counts.
    """

    # Load the YOLO model
    model = YOLO(model_path)

    # Load the image file
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"No image found at {image_path}")

    # Process the image
    results = model.track(image, classes=classes_to_count, persist=True, conf=0.2, iou=0.6, stream=False)
    
    object_counts = defaultdict(int)
    Final_obj = set()

    annotated_images = []

    for r in results:
        for box in r.boxes:
            if box.id is not None and box.cls[0] in classes_to_count:
                track_id = box.id.int().cpu().tolist()[0]
                class_id = int(box.cls[0])
                class_name = classNames[class_id]
                Final_obj.add(class_name + '_' + str(track_id))
                
        annotated_image = r.plot()
        annotated_images.append(annotated_image)

    # Determine output image path
    output_image_path = os.path.join(run_dir, "output_image.jpg")
    if annotated_images:
        # Save the annotated image
        cv2.imwrite(output_image_path, annotated_images[0])

    # Save object counts to a JSON file
    json_path = os.path.join(run_dir, "object_counts.json")
    with open(json_path, 'w') as f:
        json.dump(count(Final_obj), f, indent=4)

    return count(Final_obj), output_image_path
