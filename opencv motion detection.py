import cv2
import numpy as np
import time
import threading
from queue import Queue

def nothing(x):
    pass

def compare_frames(prev_frame, current_frame, threshold):
    """
    Compare two frames and return the thresholded difference image and detected motion contours.
    
    Args:
        prev_frame: Previous grayscale frame
        current_frame: Current color frame
        threshold: Threshold value for motion detection
    
    Returns:
        tuple: (processed current frame in grayscale, threshold image, motion contours)
    """
    # Process current frame
    gray = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (21, 21), 0)
    
    # Calculate difference between current and previous frame
    frame_diff = cv2.absdiff(prev_frame, gray)
    
    # Apply threshold to get binary image
    thresh = cv2.threshold(frame_diff, threshold, 255, cv2.THRESH_BINARY)[1]
    
    # Dilate threshold image to fill in holes
    thresh = cv2.dilate(thresh, None, iterations=2)
    
    # Find contours of moving objects
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    return gray, thresh, contours

import cv2
import numpy as np
import time
import threading
from queue import Queue, Empty, Full

class CameraThread(threading.Thread):
    def __init__(self, camera_id):
        super().__init__()
        self.camera_id = camera_id
        self.cap = cv2.VideoCapture(camera_id)
        self.frame_queue = Queue(maxsize=2)
        self.running = True
        self._stop_lock = threading.Lock()
        self._camera_lock = threading.Lock()

    def run(self):
        while True:
            with self._stop_lock:
                if not self.running:
                    break
                
            with self._camera_lock:
                if self.cap is None:
                    break
                ret, frame = self.cap.read()
                
            if ret:
                if self.frame_queue.full():
                    try:
                        self.frame_queue.get_nowait()
                    except Empty:
                        pass
                try:
                    self.frame_queue.put_nowait(frame)
                except Full:
                    pass
            time.sleep(0.001)

    def get_frame(self):
        try:
            return self.frame_queue.get_nowait()
        except Empty:
            return None

    def stop(self):
        with self._stop_lock:
            self.running = False
        
        # Give the thread a moment to exit its read loop
        time.sleep(0.1)
        
        with self._camera_lock:
            if self.cap is not None:
                self.cap.release()
                self.cap = None

def track_motion(camera1_id, camera2_id):
    # Start camera threads
    camera1 = CameraThread(camera1_id)
    camera2 = CameraThread(camera2_id)
    camera1.start()
    camera2.start()

    # Wait for first frames
    while True:
        frame1 = camera1.get_frame()
        frame2 = camera2.get_frame()
        if frame1 is not None and frame2 is not None:
            break
        time.sleep(0.1)

    # Initialize previous frames
    prev_frame1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    prev_frame1 = cv2.GaussianBlur(prev_frame1, (21, 21), 0)
    
    prev_frame2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
    prev_frame2 = cv2.GaussianBlur(prev_frame2, (21, 21), 0)
    
    # Get frame dimensions
    height1 = prev_frame1.shape[0]
    width1 = prev_frame1.shape[1]
    height2 = prev_frame2.shape[0]
    width2 = prev_frame2.shape[1]
    
    # Store reference frame from camera 2
    reference_frame2 = prev_frame2.copy()
    reference_frame2_counter = 0
    display_frame2 = None

    while True:
        # Get current frames from queues
        current_frame1 = camera1.get_frame()
        current_frame2 = camera2.get_frame()
        
        if current_frame1 is None or current_frame2 is None:
            continue

        # Get current threshold values
        threshold1 = cv2.getTrackbarPos('Threshold', 'Camera 1')
        min_size1 = cv2.getTrackbarPos('Min Size', 'Camera 1')
        
        threshold2 = cv2.getTrackbarPos('Threshold', 'Camera 2')
        min_size2 = cv2.getTrackbarPos('Min Size', 'Camera 2')
        
        line_position_percent = cv2.getTrackbarPos('Trigger', 'Camera 1')
        line_position = int((line_position_percent / 100) * width1)
        
        # Compare frames for both cameras
        gray1, _, contours1 = compare_frames(prev_frame1, current_frame1, threshold1)
        gray2, _, contours2 = compare_frames(prev_frame2, current_frame2, threshold2)
        
        # Draw vertical line
        cv2.line(current_frame1, (line_position, 0), (line_position, height1), (0, 255, 255), 2)
        
        right_motion = False
        display_frame2 = current_frame2.copy()
        
        # Process contours for camera 1
        for contour in contours1:
            if cv2.contourArea(contour) < min_size1:
                continue
                
            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(current_frame1, (x, y), (x + w, y + h), (0, 255, 0), 2)
            
            if x > line_position:
                right_motion = True
                
            cv2.putText(current_frame1, f'Motion ({(x+w//2)}, {(y+h//2)})', (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # Check for motion in camera 2
        motion_detected2 = len([cnt for cnt in contours2 if cv2.contourArea(cnt) > min_size2]) > 0
        
        # Update reference frame when no motion in camera 2
        if not motion_detected2:
            if reference_frame2_counter > 5:
                reference_frame2 = gray2.copy()
                cv2.putText(display_frame2, "Reference Frame Captured", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            else:
                reference_frame2_counter += 1
        else:
            reference_frame2_counter = 0

        # Display status text
        status_text = "Triggered" if right_motion else "Waiting"
        cv2.putText(current_frame1, status_text, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255) if right_motion else (0, 255, 0), 2)

        # Check for motion in camera 2 against reference frame
        if right_motion:
            _, _, contours3 = compare_frames(reference_frame2, current_frame2, threshold2)

            largest_area3 = 0
            largest_contour3 = None
            for contour in contours3:
                contour_area = cv2.contourArea(contour)
                if contour_area < min_size2:
                    continue
                
                if contour_area > largest_area3:
                    largest_area3 = contour_area
                    largest_contour3 = contour
            
            if largest_contour3 is not None:
                x, y, w, h = cv2.boundingRect(largest_contour3)
                cv2.rectangle(display_frame2, (x, y), (x + w, y + h), (0, 255, 0), 2)

                center = ((x+w//2), (y+h//2))
                cv2.putText(display_frame2, f'Motion {center}', (x, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                
                cv2.imshow('Camera 1', current_frame1)
                cv2.imshow('Camera 2', display_frame2)
                cv2.waitKey(1)

                # Stop cameras and return result
                camera1.stop()
                camera2.stop()
                camera1.join()
                camera2.join()
                return center
        
        # Show frames
        cv2.imshow('Camera 1', current_frame1)
        cv2.imshow('Camera 2', display_frame2)
        
        # Update previous frames
        prev_frame1 = gray1
        prev_frame2 = gray2
        
        # Break loop with 'Escape' key
        if cv2.waitKey(1) & 0xFF == 27:
            camera1.stop()
            camera2.stop()
            camera1.join()
            camera2.join()
            return None

if __name__ == "__main__":
    # Create windows and sliders
    cv2.namedWindow('Camera 1')
    cv2.createTrackbar('Threshold', 'Camera 1', 25, 100, nothing)
    cv2.createTrackbar('Min Size', 'Camera 1', 25, 100, nothing)
    cv2.createTrackbar('Trigger', 'Camera 1', 50, 100, nothing)

    cv2.namedWindow('Camera 2')
    cv2.createTrackbar('Threshold', 'Camera 2', 50, 100, nothing)
    cv2.createTrackbar('Min Size', 'Camera 2', 25, 100, nothing)

    # Load standby image
    standby_image = cv2.imread("./standby.png")
    cv2.imshow('Camera 1', standby_image)
    cv2.imshow('Camera 2', standby_image)

    while True:
        key = cv2.waitKey(1000)

        if key == 13:  # Enter key
            position = track_motion(0, 2)  # Camera IDs
            
            if position:
                print(f"{position}")
            else:
                print("No bounding boxes were returned")
                
        elif key == 27:  # Escape key
            cv2.destroyAllWindows()
            break