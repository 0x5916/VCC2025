import cv2
import numpy as np

# Initialize the background subtractor
bg_subtractor = cv2.createBackgroundSubtractorMOG2(history=150, varThreshold=20)

# Capture video from webcam or file
cap = cv2.VideoCapture(0)  # Change to video file path if needed

# Define a minimum area threshold to filter out small contours (adjust as needed)
MIN_CONTOUR_AREA = 500

# Define minimum size for merged rectangles
MIN_RECT_WIDTH = 50   # Minimum width of the rectangle
MIN_RECT_HEIGHT = 50  # Minimum height of the rectangle
# Alternatively, you can use area:
# MIN_RECT_AREA = 2500  # Minimum area (width * height)

# Function to merge overlapping or nearby rectangles
def merge_rectangles(rects, merge_threshold=50):
    """
    Merges overlapping or nearby rectangles.

    Parameters:
        rects (list of tuples): List of rectangles, each defined by (x, y, w, h).
        merge_threshold (int): Distance threshold to merge nearby rectangles.

    Returns:
        merged_rects (list of tuples): List of merged rectangles.
    """
    if not rects:
        return []

    # Convert to array for easier manipulation
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

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Optionally resize the frame for faster processing
    # frame = cv2.resize(frame, (640, 480))

    # Convert frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian Blur
    gray_blur = cv2.GaussianBlur(gray, (7, 7), 3)

    # Apply background subtraction
    fg_mask = bg_subtractor.apply(gray_blur)

    # Morphological operations
    kernel = np.ones((7, 7), np.uint8)
    eroded = cv2.erode(fg_mask, kernel, iterations=1)
    eroded_blur = cv2.GaussianBlur(eroded, (5, 5), 3)
    dilated = cv2.dilate(eroded_blur, np.ones((19, 19), np.uint8), iterations=15)

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
    merged_rects = merge_rectangles(bounding_rects, merge_threshold=70)  # Adjust threshold as needed

    # Further filter merged rectangles based on size
    significant_rects = []
    for rect in merged_rects:
        x, y, w, h = rect
        # Option 1: Filter based on width and height
        if w >= MIN_RECT_WIDTH and h >= MIN_RECT_HEIGHT:
            significant_rects.append(rect)
        # Option 2: Alternatively, filter based on area
        # if (w * h) >= MIN_RECT_AREA:
        #     significant_rects.append(rect)

    # Optional: Handle frames where all merged rectangles are small
    if significant_rects:
        # Draw significant rectangles on the original frame
        for rect in significant_rects:
            x, y, w, h = rect
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            # Optionally, add labels or other annotations
            # cv2.putText(frame, "Motion", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX,
            #             0.5, (0, 255, 0), 2)
    else:
        # Optionally, perform actions when no significant motion is detected
        # For example, display a message or keep the frame unannotated
        pass  # No rectangles drawn

    # Show the frames
    cv2.imshow("Masked Image", fg_mask)
    cv2.imshow("Eroded", eroded)
    cv2.imshow("Dilated", dilated)
    cv2.imshow("Filtered", filtered)
    cv2.imshow("Motion Tracking", frame)

    # Exit when 'Esc' key is pressed
    if cv2.waitKey(30) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()