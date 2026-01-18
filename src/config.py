"""
config.py

Holds global configuration constants used throughout the project.
"""

FRAME_QUEUE_SIZE = 5   # Max frames in the queue to avoid large latencies
RESULT_QUEUE_SIZE = 5  # Max results in the queue

FRAME_WIDTH = 1920
FRAME_HEIGHT = 1080

# Ratio for splitting left/right. For example, 0.5 draws a vertical line at 50% width.
LR_SPLIT = 0.5

MODEL_MIN_CONFIDENCE = 0.75

# Dictionary mapping model class names to BGR colors (OpenCV uses BGR).
# Make sure these keys match what your YOLO model outputs in model.names.
label_to_color = {
    "Blue Marble":   (255, 0, 0),
    "Green Marble":  (0, 255, 0),
    "Red Marble":    (0, 0, 255),
    "Yellow Marble": (0, 255, 255)
}