import cv2
import numpy as np

def merge_rectangles(rects, merge_threshold=50):
    """
    Merges overlapping or nearby rectangles.

    Parameters:
        rects (list of tuples): List of rectangles, each defined by (x, y, w, h).
        merge_threshold (int): Distance threshold to merge nearby rectangles.

    Returns:
        merged_rects (list of tuples): List of merged rectangles as (x, y, w, h).
    """
    if not rects:
        return []

    rects = np.array(rects)
    merged = []
    while len(rects) > 0:
        # Take the first rectangle and compare with others
        current = rects[0]
        rest = rects[1:]

        # Compute the center of the current rectangle
        current_center = np.array([current[0] + current[2]/2, current[1] + current[3]/2])

        to_merge = []
        indices_to_keep = []
        for i, rect in enumerate(rest):
            rect_center = np.array([rect[0] + rect[2]/2, rect[1] + rect[3]/2])
            distance = np.linalg.norm(current_center - rect_center)
            if distance < merge_threshold:
                to_merge.append(rect)
            else:
                indices_to_keep.append(i)

        # Merge current rectangle with the ones in to_merge
        if to_merge:
            all_rects = np.vstack([current, to_merge])
        else:
            all_rects = current.reshape(1, 4)

        # Compute the bounding rectangle for all merged rectangles
        x_min = np.min(all_rects[:, 0])
        y_min = np.min(all_rects[:, 1])
        x_max = np.max(all_rects[:, 0] + all_rects[:, 2])
        y_max = np.max(all_rects[:, 1] + all_rects[:, 3])
        merged_rect = (int(x_min), int(y_min), int(x_max - x_min), int(y_max - y_min))
        merged.append(merged_rect)

        # Update the rects array
        rects = rest[indices_to_keep]

    return merged

def find_motion(image):
    """
    Detects motion in the given image and returns bounding boxes around moving objects.

    Parameters:
        image (numpy.ndarray): The input image/frame in BGR format.

    Returns:
        List of tuples: Each tuple contains (x1, y1, x2, y2) coordinates of the bounding box.
    """
    # Initialize background subtractor and kernel as function attributes
    if not hasattr(find_motion, "bg_subtractor"):
        find_motion.bg_subtractor = cv2.createBackgroundSubtractorMOG2(history=200, varThreshold=18)
    
    if not hasattr(find_motion, "kernel"):
        find_motion.kernel = np.ones((7, 7), np.uint8)
    
    # Define thresholds
    MIN_CONTOUR_AREA = 500
    MIN_RECT_WIDTH = 50
    MIN_RECT_HEIGHT = 50
    MERGE_THRESHOLD = 70  # Adjust based on your use case

    # Convert frame to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian Blur
    gray_blur = cv2.GaussianBlur(gray, (7, 7), 3)

    # Apply background subtraction
    fg_mask = find_motion.bg_subtractor.apply(gray_blur)

    # Morphological operations
    eroded = cv2.erode(fg_mask, find_motion.kernel, iterations=1)
    eroded_blur = cv2.GaussianBlur(eroded, (5, 5), 3)
    dilated = cv2.dilate(eroded_blur, find_motion.kernel, iterations=15)

    # Threshold to get binary image
    _, filtered = cv2.threshold(dilated, 50, 255, cv2.THRESH_BINARY)

    # Find contours in the filtered image
    contours, _ = cv2.findContours(filtered, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # List to hold bounding rectangles
    bounding_rects = []

    # Iterate over contours and collect bounding rectangles
    for contour in contours:
        if cv2.contourArea(contour) < MIN_CONTOUR_AREA:
            continue  # Skip small contours that may be noise

        # Get bounding rectangle coordinates
        x, y, w, h = cv2.boundingRect(contour)
        bounding_rects.append((x, y, w, h))

    # Merge overlapping or nearby rectangles
    merged_rects = merge_rectangles(bounding_rects, merge_threshold=MERGE_THRESHOLD)

    # Further filter merged rectangles based on size
    significant_rects = []
    for rect in merged_rects:
        x, y, w, h = rect
        if w >= MIN_RECT_WIDTH and h >= MIN_RECT_HEIGHT:
            significant_rects.append((x, y, x + w, y + h))  # Convert to (x1, y1, x2, y2)

    return significant_rects