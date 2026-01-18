"""
main.py

Entry point for the application: sets up threads, processes detections,
displays results (with FPS), and handles program shutdown.
"""

import threading

import cv2
import queue
import numpy as np

from typing import Literal, Any, Generator

from config import (
    FRAME_QUEUE_SIZE,
    RESULT_QUEUE_SIZE,
    FRAME_WIDTH,
    FRAME_HEIGHT,
    LR_SPLIT,
    MODEL_MIN_CONFIDENCE,
    label_to_color
)
from env import (
    CAM_INDEX,
    TRIGGER_HEIGHT,
    USE_MICROBIT,
    MICROBIT_SERIAL_PORT,
    MICROBIT_BAND_RATE
)
from claim_downer import Claimdowner
from microbit_control import SerialConnector
from counter import FPSCounter, LabelCounter
from frame import FrameGrabber
from model import ModelThread


def get_split(x: float, width: int, split_ratio: float) -> Literal["left", "right"]:
    """
    Returns 'right' if x >= (split_ratio * width), otherwise 'left'.
    """
    return "right" if x >= split_ratio * width else "left"


def most_similar_color(image_rgb: np.ndarray) -> Literal["red", "green", "yellow", "blue"]:
    """
    Determine whether an RGB image (e.g. of a marble) is closest in color
    to red, green, yellow, or blue.

    Parameters
    ----------
    image_rgb : np.ndarray
        Input image as an H×W×3 array in RGB order, dtype uint8 or float.

    Returns
    -------
    str
        One of: "red", "green", "yellow", "blue".
    """
    if image_rgb.ndim != 3 or image_rgb.shape[2] != 3:
        raise ValueError("Input must be an H×W×3 RGB image.")

    # Ensure float for averaging
    img = image_rgb.astype(np.float32)

    # Compute the mean R, G, B over all pixels
    mean_color = img.reshape(-1, 3).mean(axis=0)  # shape (3,)

    # Define our four reference colors in RGB
    # Red     = (255,   0,   0)
    # Green   = (  0, 255,   0)
    # Yellow  = (255, 255,   0)
    # Blue    = (  0,   0, 255)
    prototypes = {
        "red": np.array([0, 0, 255], dtype=np.float32),
        "green": np.array([0, 255, 0], dtype=np.float32),
        "yellow": np.array([0, 255, 255], dtype=np.float32),
        "blue": np.array([255, 0, 0], dtype=np.float32),
    }

    # Compute Euclidean distance from the mean color to each prototype
    distances = {name: np.linalg.norm(mean_color - rgb)
                 for name, rgb in prototypes.items()}

    # Find the color name with minimum distance
    best_match = min(distances, key=distances.get)
    return best_match


def box_drawing(frame, label: str, conf: float, side: str, box: tuple[int, int, int, int]):
    x_min, y_min, x_max, y_max = box
    # Draw bounding boxes
    color = label_to_color.get(label, (255, 255, 255))
    cv2.rectangle(frame,
                  (x_min, y_min),
                  (x_max, y_max),
                  color, 4)

    # Label text
    text = f"{label} on {side} (conf: {conf:.2f})"
    text_position = (int(x_min), int(y_min) - 10)
    cv2.putText(frame, text, text_position,
                cv2.FONT_HERSHEY_SIMPLEX, 1.3, color, 3)


def box_converter(box) -> tuple[Any, Generator[int, Any, None]]:
    return box.conf[0].item(), (int(i) for i in box.xyxy[0])

def draw_split_lines(frame, split_x, count_y, width, height):
    cv2.line(frame, (split_x, 0), (split_x, height), (0, 0, 255), 3)
    cv2.line(frame, (0, count_y), (width, count_y), (0, 255, 0), 2)

class VCC2025:
    def __init__(self):
        # Queues for frames and inference results
        self.frame_queue = queue.Queue(maxsize=FRAME_QUEUE_SIZE)
        self.result_queue = queue.Queue(maxsize=RESULT_QUEUE_SIZE)
        self.stop_event = threading.Event()

        # Set enable=True if you want to send serial commands to the micro:bit
        self.serialConnector = SerialConnector(serial_port=MICROBIT_SERIAL_PORT, enable=USE_MICROBIT)

        # Calculate positions for vertical split line and horizontal "trigger" line
        self.split_x = int(LR_SPLIT * FRAME_WIDTH)
        self.count_y = int(TRIGGER_HEIGHT * FRAME_HEIGHT)

        self.capture_thread = FrameGrabber(
            self.frame_queue, self.stop_event,
            cam_index=CAM_INDEX,
            width=FRAME_WIDTH,
            height=FRAME_HEIGHT
        )
        self.capture_thread.start()

        self.model_thread = ModelThread(
            self.frame_queue, self.result_queue, self.stop_event,
            model_path="./best.pt",
            min_conf=MODEL_MIN_CONFIDENCE
        )
        self.model_thread.start()

        self.labels = ["red", "green", "yellow", "blue"]
        self.label_counter = LabelCounter(self.labels)
        self.fps_counter = FPSCounter()

        # Initialize classes
        self.claim_downer = Claimdowner()

    def frame_processer(self, frame, fps_display):
        height, width, _ = frame.shape
        # Draw split line (vertical) and trigger line (horizontal)
        draw_split_lines(frame, self.split_x, self.count_y, width, height)

        # Show counters for each label in the top-left
        # The order depends on the YOLO model's .names order
        scale_factor = width / 1600.0
        base_offset = int(scale_factor * 50)

        for i, label in enumerate(self.labels):
            color = label_to_color.get(label, (255, 255, 255))
            counter_text = f"Counter ({label}): {self.label_counter.get_count(label)}"
            text_position = (base_offset, base_offset * (i + 1))

            # Draw a thick black outline
            cv2.putText(frame, counter_text, text_position,
                        cv2.FONT_HERSHEY_DUPLEX, scale_factor, (0, 0, 0), 10)
            # Draw the colored text on top
            cv2.putText(frame, counter_text, text_position,
                        cv2.FONT_HERSHEY_DUPLEX, scale_factor, color, 2)

        # Show FPS in the upper-right corner
        fps_text = f"FPS: {fps_display:.1f}"
        offset_x, offset_y = 200, 50
        # Shadow (black) text
        cv2.putText(frame, fps_text, (width - offset_x, offset_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 0), 5)
        # Main (white) text
        cv2.putText(frame, fps_text, (width - offset_x, offset_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2)

    def main(self):
        marble_passed = False

        while True:
            if cv2.waitKey(1) & 0xFF == ord('q'): break
            # Retrieve results if available
            try:
                frame, detections = self.result_queue.get(True, 0.05)
                height, width, _ = frame.shape
                fps_display = self.fps_counter.update_and_get()
                self.frame_processer(frame, fps_display)
            except queue.Empty:
                continue

            # Process each detection
            for box in detections:
                conf, box_cor = box_converter(box)
                if conf < MODEL_MIN_CONFIDENCE:
                    continue  # skip low-confidence

                x_min, y_min, x_max, y_max = box_cor

                side = get_split((x_min + x_max) / 2, width, LR_SPLIT)

                marble_img = frame[y_min:y_max, x_min:x_max]
                label = most_similar_color(marble_img)

                if y_max >= self.count_y >= y_min - height * 0.05:
                    self.claim_downer.found(label)
                    if self.claim_downer.total_count() >= 1:
                        max_count_label = self.claim_downer.max_count_label()
                        # Send a short color/side code. e.g.: 'r' or 'b'
                        self.serialConnector.send_serial(max_count_label[0].lower(), side[0].lower())
                    marble_passed = True
                elif marble_passed:
                    self.label_counter.update(label)
                    self.claim_downer.reset()
                    marble_passed = False

                box_drawing(frame, label, conf, side, box_cor)

            cv2.imshow("Marble Detection", frame)

    def __del__(self):
        # Signal the threads to stop and wait for them
        self.stop_event.set()
        self.capture_thread.join()
        self.model_thread.join()

        # Clean up
        self.serialConnector.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    vcc = VCC2025()
    vcc.main()
