import argparse
import os
import json
from datetime import datetime
from object_counting import process_video_and_count

def main():
    parser = argparse.ArgumentParser(description="Count objects in a video.")
    parser.add_argument('--video_path', type=str, required=True, help="Path to the video file.")
    parser.add_argument('--model_path', type=str, default='yolov8s.pt', help="Path to the YOLO model file.")
    parser.add_argument('--classes_to_count', nargs='+', type=int, help="List of class IDs to count.", default=[i for i in range(1, 81)])

    args = parser.parse_args()

    # Generate a new run directory
    run_dir = os.path.join("runs", datetime.now().strftime("%Y%m%d_%H%M%S"))
    os.makedirs(run_dir, exist_ok=True)

    # Process video and count objects
    object_counts, output_video_path = process_video_and_count(args.video_path, args.model_path, args.classes_to_count, run_dir)

    # Save object counts to JSON
    json_path = os.path.join(run_dir, "object_counts.json")
    with open(json_path, "w") as json_file:
        json.dump(object_counts, json_file, indent=4)

    print(f"Object counts saved to {json_path}")
    print(f"Processed video saved to {output_video_path}")

if __name__ == "__main__":
    main()