import time
import threading
from queue import Queue
import cv2
import numpy as np

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

class GridDrawer:
    def __init__(self):
        self.points = []
        self.max_points = 4
        self.grid_size_x = 5  # Default grid size X
        self.grid_size_y = 5  # Default grid size Y
        self.transform_matrix = None
        self.inverse_matrix = None
        self.highlighted_cell = None
        self.counter = 0
        
    def mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            if len(self.points) < self.max_points:
                self.points.append((x, y))
                if len(self.points) == self.max_points:
                    self.counter = 0
            else:
                self.points[self.counter] = (x, y)
                self.counter = (self.counter + 1) % self.max_points
        #elif event == cv2.EVENT_MOUSEMOVE:
        #    if len(self.points) == self.max_points:
        #        self.highlighted_cell = self.get_grid_coordinate(x, y)
    
    def get_grid_coordinate(self, x, y):
        """Convert screen coordinates to grid coordinates."""
        if self.inverse_matrix is None or len(self.points) != self.max_points:
            return None
            
        point = np.array([x, y, 1]).reshape(1, -1)
        transformed = point.dot(self.inverse_matrix.T)
        if transformed[0, 2] == 0:
            return None
            
        grid_x = transformed[0, 0] / transformed[0, 2]
        grid_y = transformed[0, 1] / transformed[0, 2]
        
        # Convert to grid coordinates using separate x and y cell sizes
        cell_size_x = 500 / self.grid_size_x
        cell_size_y = 500 / self.grid_size_y
        
        grid_col = int(grid_x / cell_size_x)
        grid_row = int(grid_y / cell_size_y)
        
        if 0 <= grid_col < self.grid_size_x and 0 <= grid_row < self.grid_size_y:
            self.highlighted_cell = (grid_row, grid_col)
            return self.highlighted_cell
        
        # Not within any grid cells
        return None
    
    def project_grid(self, frame):
        if len(self.points) != 4:
            return frame
            
        cell_size_x = 500 / self.grid_size_x
        cell_size_y = 500 / self.grid_size_y
        
        src_points = np.float32([[0, 0], [500, 0], [500, 500], [0, 500]])
        dst_points = np.float32(self.points)
        
        self.transform_matrix = cv2.getPerspectiveTransform(src_points, dst_points)
        self.inverse_matrix = cv2.getPerspectiveTransform(dst_points, src_points)
        
        grid_img = np.zeros((501, 501, 3), dtype=np.uint8)
        
        # Draw vertical lines (x-axis divisions)
        for i in range(0, 501, int(cell_size_x)):
            cv2.line(grid_img, (i, 0), (i, 500), (0, 255, 0), 2)
            
        # Draw horizontal lines (y-axis divisions)
        for i in range(0, 501, int(cell_size_y)):
            cv2.line(grid_img, (0, i), (500, i), (0, 255, 0), 2)
        
        # Highlight cell if mouse is over grid
        if self.highlighted_cell is not None:
            row, col = self.highlighted_cell
            top_left = (int(col * cell_size_x), int(row * cell_size_y))
            bottom_right = (int((col + 1) * cell_size_x), int((row + 1) * cell_size_y))
            cv2.rectangle(grid_img, top_left, bottom_right, (0, 0, 255, 128), -1)
        
        warped_grid = cv2.warpPerspective(grid_img, self.transform_matrix, 
                                        (frame.shape[1], frame.shape[0]))
        
        blend = cv2.addWeighted(frame, 1, warped_grid, 0.7, 0)
        return blend
    
    def draw_on_frame(self, frame):
        display_frame = frame.copy()
        
        if len(self.points) == self.max_points:
            display_frame = self.project_grid(display_frame)
        
        # Draw points
        for i, point in enumerate(self.points):
            color = (0, 255, 255) if (len(self.points) == self.max_points and i == self.counter) else (0, 0, 255)
            cv2.circle(display_frame, point, 5, color, -1)
            cv2.putText(display_frame, str(i), 
                       (point[0] + 10, point[1] + 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
        # Draw lines between points
        if len(self.points) > 1:
            for i in range(len(self.points) - 1):
                cv2.line(display_frame, self.points[i], self.points[i + 1], 
                        (255, 0, 0), 2)
                
            if len(self.points) == self.max_points:
                cv2.line(display_frame, self.points[-1], self.points[0], 
                        (255, 0, 0), 2)
        
        # Display status information
        remaining = self.max_points - len(self.points)
        if remaining > 0:
            message = f"Click {remaining} more points"
        else:
            message = f"Next update: Point {self.counter} | Grid: {self.grid_size_x}x{self.grid_size_y}"
            if self.highlighted_cell is not None:
                row, col = self.highlighted_cell
                message += f" | Cell: ({row}, {col})"
                
        cv2.putText(display_frame, message, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        return display_frame

def track_motion(camera1_id, camera2_id, grid):
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
        line_position = int((line_position_percent / 100) * height1)
        
        # Compare frames for both cameras
        gray1, _, contours1 = compare_frames(prev_frame1, current_frame1, threshold1)
        gray2, _, contours2 = compare_frames(prev_frame2, current_frame2, threshold2)
        
        # Draw horizontal line
        cv2.line(current_frame1, (0, line_position), (width1, line_position), (0, 255, 255), 2)
        
        triggered = False
        display_frame2 = current_frame2.copy()
        
        # Process contours for camera 1
        for contour in contours1:
            if cv2.contourArea(contour) < min_size1:
                continue
                
            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(current_frame1, (x, y), (x + w, y + h), (0, 255, 0), 2)
            
            center = ((x+w//2), (y+h//2))
            if center[1] > line_position:
                triggered = True
                
            cv2.putText(current_frame1, f'Motion ({(x+w//2)}, {(y+h//2)})', (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # Check for motion in camera 2
        motion_detected2 = len([cnt for cnt in contours2 if cv2.contourArea(cnt) > min_size2]) > 0
        
        # Update reference frame when no motion in camera 2
        if not motion_detected2:
            if reference_frame2_counter > 5:
                reference_frame2 = gray2.copy()
                cv2.putText(display_frame2, "Reference Frame Captured", (10, 70),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            else:
                reference_frame2_counter += 1
        else:
            reference_frame2_counter = 0

        # Display status text
        status_text = "Triggered" if triggered else "Waiting"
        cv2.putText(current_frame1, status_text, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255) if triggered else (0, 255, 0), 2)

        # Check for motion in camera 2 against reference frame
        if triggered:
            _, _, contours3 = compare_frames(reference_frame2, current_frame2, threshold2)

            # Find the largest contour
            largest_area3 = 0
            largest_contour3 = None
            for contour in contours3:
                contour_area = cv2.contourArea(contour)
                if contour_area < min_size2:
                    continue
                
                if contour_area > largest_area3:
                    largest_area3 = contour_area
                    largest_contour3 = contour
            
            # If there is a contour with area greater than the minimum
            if largest_contour3 is not None:
                x, y, w, h = cv2.boundingRect(largest_contour3)
                cv2.rectangle(display_frame2, (x, y), (x + w, y + h), (0, 255, 0), 2)

                center = ((x+w//2), (y+h//2))
                cv2.putText(display_frame2, f'Motion {center}', (x, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                
                # Get the grid cell
                grid_cell = grid.get_grid_coordinate(center[0], center[1])

                if grid_cell is not None:
                    # Project grid onto camera 2
                    display_frame2 = grid.draw_on_frame(display_frame2)
                    
                    cv2.imshow('Camera 1', current_frame1)
                    cv2.imshow('Camera 2', display_frame2)
                    cv2.waitKey(1)

                    # Stop cameras and return result
                    camera1.stop()
                    camera2.stop()
                    camera1.join()
                    camera2.join()
                    return grid_cell
            
        # Project grid onto camera 2
        display_frame2 = grid.draw_on_frame(display_frame2)
        
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

    # Create mouse callback
    grid = GridDrawer()
    cv2.setMouseCallback("Camera 2", grid.mouse_callback)

    # Load standby image
    standby_image = cv2.imread("./standby.png")
    cv2.imshow('Camera 1', standby_image)
    cv2.imshow('Camera 2', standby_image)

    while True:
        key = cv2.waitKey(1000)

        if key == 13:  # Enter key
            position = track_motion(0, 2, grid)  # Camera IDs & grid class
            
            if position:
                print(f"{position}")
            else:
                print("Motion tracking was aborted")
                
        elif key == 27:  # Escape key
            cv2.destroyAllWindows()
            break