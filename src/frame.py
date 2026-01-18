"""
frame.py

Defines a FrameGrabber thread that continuously captures frames from the camera
and places them into a queue for processing.
"""

import threading
import cv2
import queue

class FrameGrabber(threading.Thread):
    def __init__(self, frame_queue, stop_event, cam_index=0,
                 width=640, height=480, desired_fps=60):
        """
        :param frame_queue: Queue to place captured frames in.
        :param stop_event: Event to signal when to stop capturing.
        :param cam_index: Index of the camera/video device.
        :param width: Desired capture width.
        :param height: Desired capture height.
        :param desired_fps: Attempt to set camera FPS (may vary by device).
        """
        super().__init__()
        self.frame_queue = frame_queue
        self.stop_event = stop_event
        self.cam_index = cam_index
        self.width = width
        self.height = height
        self.desired_fps = desired_fps
        self.cap = None

    def run(self):
        self.cap = cv2.VideoCapture(self.cam_index)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        self.cap.set(cv2.CAP_PROP_FPS, self.desired_fps)

        while not self.stop_event.is_set():
            success, frame = self.cap.read()
            if not success:
                print("Error: Could not read frame from camera.")
                break

            # If queue is full, skip adding this frame to avoid blocking
            try:
                self.frame_queue.put_nowait(frame)
            except queue.Full:
                pass

        # Cleanup
        if self.cap is not None:
            self.cap.release()