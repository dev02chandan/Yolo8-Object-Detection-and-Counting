from streamlit_webrtc import VideoProcessorBase, webrtc_streamer, WebRtcMode
import av
import torch
from ultralytics import YOLO
from collections import defaultdict

classNames = ['cup', 'cutter', 'fork', 'knife', 'painting', 'pan', 'plant', 'plate', 'scissor', 'spoon']

class VideoProcessor(VideoProcessorBase):
    def __init__(self, model_path, classes_to_count, iou, conf, imgsz, tracker, device):
        self.model = YOLO(model_path).to(device)
        self.classes_to_count = classes_to_count
        self.iou = iou
        self.conf = conf
        self.imgsz = imgsz
        self.tracker = tracker
        self.device = device
        self.tracked_objects = defaultdict(lambda: defaultdict(int))

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        results = self.model.track(img, classes=self.classes_to_count, persist=True, tracker=self.tracker, conf=self.conf, iou=self.iou, imgsz=self.imgsz, device=self.device, half=(self.device != 'cpu'))

        for r in results:
            for box in r.boxes:
                if box.id is not None and box.cls[0] in self.classes_to_count:
                    track_id = box.id.int().tolist()[0]
                    class_id = int(box.cls[0])
                    class_name = classNames[class_id]

                    # Count objects with respect to their track ID
                    self.tracked_objects[track_id][class_name] += 1

        annotated_frame = results[0].plot()
        return av.VideoFrame.from_ndarray(annotated_frame, format="bgr24")

def process_livestream_and_count(model_path, classes_to_count, iou, conf, imgsz, tracker, device):
    ctx = webrtc_streamer(
        key="example",
        mode=WebRtcMode.SENDRECV,
        video_processor_factory=lambda: VideoProcessor(model_path, classes_to_count, iou, conf, imgsz, tracker, device),
        media_stream_constraints={"video": True, "audio": False},
    )

    return ctx.video_processor.tracked_objects if ctx.video_processor else {}
