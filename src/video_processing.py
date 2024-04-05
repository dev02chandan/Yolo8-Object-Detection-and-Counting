# video_processing.py

import cv2

def reduce_frame_rate(input_video_path, output_video_path, new_fps=10):
    # Open the original video file
    cap = cv2.VideoCapture(input_video_path)

    # Get the original frame rate
    fps = cap.get(cv2.CAP_PROP_FPS)

    # Create a VideoWriter object to write the new video
    out = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc(*'mp4v'), new_fps, 
                          (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))))

    # Read frames from the original video and write them to the new video
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        out.write(frame)

    # Release the VideoCapture and VideoWriter objects
    cap.release()
    out.release()
