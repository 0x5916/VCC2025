import cv2
import os
import time

def main():
    save_path = "./dataset/backgorund"
    os.makedirs(save_path, exist_ok=True)  # Ensure the directory exists
    
    cap = cv2.VideoCapture(0)  # Open camera at index 0
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
    
    if not cap.isOpened():
        print("Error: Could not open camera.")
        return
    
    print("Press 'c' to capture an image, 'q' to exit.")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame.")
            break
        
        cv2.imshow("Camera", frame)
        
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('c'):
            img_name = os.path.join(save_path, f"captured_{time.time()}.jpg")
            cv2.imwrite(img_name, frame)
            print(f"Image saved to {img_name}")
        elif key == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()