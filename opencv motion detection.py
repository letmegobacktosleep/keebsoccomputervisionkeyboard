import cv2
import numpy as np
import time

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

def track_motion(cap1, cap2):
    
    # Read first frames
    _, prev_frame1 = cap1.read()
    prev_frame1 = cv2.cvtColor(prev_frame1, cv2.COLOR_BGR2GRAY)
    prev_frame1 = cv2.GaussianBlur(prev_frame1, (21, 21), 0)
    
    _, prev_frame2 = cap2.read()
    prev_frame2 = cv2.cvtColor(prev_frame2, cv2.COLOR_BGR2GRAY)
    prev_frame2 = cv2.GaussianBlur(prev_frame2, (21, 21), 0)
    
    # Get frame dimensions
    height1 = prev_frame1.shape[0]
    width1  = prev_frame1.shape[1]
    height2 = prev_frame2.shape[0]
    width2  = prev_frame2.shape[1]
    
    # Store reference frame from camera 2 (when no motion is detected)
    reference_frame2 = prev_frame2.copy()
    reference_frame2_counter = 0 # Count number of frames since last motion
    display_frame2 = None  # Frame to display for camera 2
    
    while True:
        # Read current frames
        ret1, current_frame1 = cap1.read()
        ret2, current_frame2 = cap2.read()
        if not ret1 or not ret2:
            break
        
        # Get current threshold values
        threshold1 = cv2.getTrackbarPos('Threshold Value', 'Camera 1')
        min_size1  = cv2.getTrackbarPos('Minimum Size', 'Camera 1')

        threshold2 = cv2.getTrackbarPos('Threshold Value', 'Camera 2')
        min_size2  = cv2.getTrackbarPos('Minimum Size', 'Camera 2')

        # Get current line position values
        line_position_percent = cv2.getTrackbarPos('Line Position %', 'Camera 1')
        line_position = int((line_position_percent / 100) * width1)
        
        # Compare frames for both cameras
        gray1, _, contours1 = compare_frames(prev_frame1, current_frame1, threshold1)
        gray2, _, contours2 = compare_frames(prev_frame2, current_frame2, threshold2)
        
        # Draw vertical line at adjustable position (camera 1)
        cv2.line(current_frame1, (line_position, 0), (line_position, height1), (0, 255, 255), 2)
        
        right_motion = False
        display_frame2 = current_frame2.copy()  # Reset display frame
        
        # Process contours for camera 1
        for contour in contours1:
            if cv2.contourArea(contour) < min_size1:
                continue
                
            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(current_frame1, (x, y), (x + w, y + h), (0, 255, 0), 2)
            
            # Check if motion is right of the line
            if (x + w/2) > line_position:
                right_motion = True
                
            cv2.putText(current_frame1, f'Motion ({x}, {y})', (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # Check for motion in camera 2
        motion_detected2 = len([cnt for cnt in contours2 if cv2.contourArea(cnt) > min_size2]) > 0
        
        # Update reference frame when no motion in camera 2
        if not motion_detected2:
            # Update if it has been more than 10 frames since last motion
            if reference_frame2_counter > 10:
                reference_frame2 = gray2.copy()
                cv2.putText(display_frame2, "Reference Frame Captured", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            else:
                reference_frame2_counter += 1 # there has been motion in the last few frames
        else:
            reference_frame2_counter = 0

        # Check for motion in camera 2, against reference frame
        if right_motion:

            _, _, contours3 = compare_frames(reference_frame2, current_frame2, threshold2)

            # Process contours for camera 2
            largest_area3 = 0
            largest_contour3 = None
            for contour in contours3:
                contour_area = cv2.contourArea(contour)
                if contour_area < min_size2:
                    continue
                
                if contour_area > largest_area3:
                    largest_area3 = contour_area
                    largest_contour3 = contour
            
            # Draw bounding box on largest contour and return center of box
            if largest_contour3 is not None:
                x, y, w, h = cv2.boundingRect(largest_contour3)
                cv2.rectangle(display_frame2, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    
                cv2.putText(display_frame2, f'Motion ({x}, {y})', (x, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                
                # Show frames
                cv2.imshow('Camera 1', current_frame1)
                cv2.imshow('Camera 2', display_frame2)
                cv2.waitKey(1)
                
                # Return center of box
                return ((x+w//2), (y+h//2))
        
        # Display status text
        status_text = "Right Motion: YES" if right_motion else "Right Motion: NO"
        cv2.putText(current_frame1, status_text, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255) if right_motion else (0, 255, 0), 2)
        
        # Show frames
        cv2.imshow('Camera 1', current_frame1)
        cv2.imshow('Camera 2', display_frame2)
        # cv2.imshow('Threshold', thresh1)
        
        # Update previous frames
        prev_frame1 = gray1
        prev_frame2 = gray2
        
        # Break loop with 'Escape' key
        if cv2.waitKey(1) & 0xFF == 27:
            return None
    
    

if __name__ == "__main__":

    # Create window and sliders
    cv2.namedWindow('Camera 1')
    cv2.createTrackbar('Threshold Value', 'Camera 1', 25, 100, nothing)
    cv2.createTrackbar('Minimum Size', 'Camera 1', 25, 100, nothing)
    cv2.createTrackbar('Line Position %', 'Camera 1', 50, 100, nothing)

    cv2.namedWindow('Camera 2')
    cv2.createTrackbar('Threshold Value', 'Camera 2', 10, 100, nothing)
    cv2.createTrackbar('Minimum Size', 'Camera 2', 25, 100, nothing)

    # Load standby image
    standby_image = cv2.imread("./standby.png")
    cv2.imshow('Camera 1', standby_image)
    cv2.imshow('Camera 2', standby_image)

    while True:

        key = cv2.waitKey(1000)

        # Start motion detection with 'Enter' key
        if key == 13:

            # Start video captures
            cap1 = cv2.VideoCapture(0)  # First camera
            cap2 = cv2.VideoCapture(2)  # Second camera

            if not cap1.isOpened() or not cap2.isOpened():
                print("Error: Couldn't open one or both cameras")
                continue

            # Track motion
            position = track_motion(cap1, cap2)

            # Release resources
            cap1.release()
            cap2.release()
            
            # Check if a bounding box was returned
            if position:
                print(f"{position}")
            else:
                print("No bounding boxes were returned")
        
        # Kill program with 'Escape' key
        elif key == 27:
            cv2.destroyAllWindows()
            break