"""
model.py

Defines a ModelThread that pulls frames from the queue, runs YOLO inference,
and places (frame, detections) results into another queue.
"""

import threading
import queue
import torch
from ultralytics import YOLO

class ModelThread(threading.Thread):
    """
    Continuously pulls frames from frame_queue, runs YOLO inference,
    and puts (frame, detections) into result_queue.
    """
    def __init__(self, frame_queue, result_queue, stop_event,
                 model_path="best.pt", min_conf=0.5):
        super().__init__()
        self.frame_queue = frame_queue
        self.result_queue = result_queue
        self.stop_event = stop_event
        self.min_conf = min_conf

        # Decide which device to use: GPU (CUDA), MPS for Mac, or CPU.
        if torch.cuda.is_available():
            self.device = "cuda"
        elif torch.backends.mps.is_available():
            self.device = "mps"
        else:
            self.device = "cpu"
        print(f"Using device: {self.device}")

        # Initialize the YOLO model
        self.model = YOLO(model_path)
        self.model.conf = self.min_conf

    def run(self):
        while not self.stop_event.is_set():
            try:
                frame = self.frame_queue.get(timeout=0.01)
            except queue.Empty:
                continue

            # YOLO inference
            results = self.model(
                frame,
                conf=self.min_conf,
                device=self.device,
                verbose=True
            )
            detections = results[0].boxes  # YOLO detections in first batch

            # Put (frame, detections) into the result queue
            try:
                self.result_queue.put_nowait((frame, detections))
            except queue.Full:
                pass