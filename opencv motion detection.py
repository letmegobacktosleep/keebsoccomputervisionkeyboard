# Import built-in libraries
import os
import time
import threading
from queue import Queue
# Import 3rd party libraries
import cv2
import numpy as np
from pynput import keyboard
from pynput.keyboard import Key, Controller

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
            time.sleep(0.01)

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
    def __init__(self, grid_size_x=5, grid_size_y=5):
        self.points = []
        self.max_points = 4
        self.grid_size_x = grid_size_x
        self.grid_size_y = grid_size_y
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
    
    def get_grid_coordinate(self, x, y):
        """Convert screen coordinates to grid coordinates using linear interpolation."""
        if len(self.points) != 4:
            return None
            
        # Helper function for inverse lerp
        def inverse_lerp(start, end, point):
            # Returns how far point is between start and end (0 to 1)
            if abs(end - start) < 0.0001:  # Avoid division by zero
                return 0
            return (point - start) / (end - start)
        
        def point_to_line_distance(p, a, b):
            # Returns distance from point p to line segment ab
            ax, ay = a
            bx, by = b
            px, py = p
            
            # Vector from a to b
            abx = bx - ax
            aby = by - ay
            
            # Vector from a to p
            apx = px - ax
            apy = py - ay
            
            # Length of ab squared
            ab_squared = abx * abx + aby * aby
            
            if ab_squared == 0:
                return ((px - ax) ** 2 + (py - ay) ** 2) ** 0.5
                
            # Calculate dot product
            ap_ab = apx * abx + apy * aby
            
            # Get normalized distance along line
            t = ap_ab / ab_squared
            
            if t < 0:
                return ((px - ax) ** 2 + (py - ay) ** 2) ** 0.5
            elif t > 1:
                return ((px - bx) ** 2 + (py - by) ** 2) ** 0.5
                
            # Project point onto line
            proj_x = ax + t * abx
            proj_y = ay + t * aby
            
            return ((px - proj_x) ** 2 + (py - proj_y) ** 2) ** 0.5
        
        # Check if point is inside polygon using ray casting
        poly = np.array(self.points)
        if cv2.pointPolygonTest(poly, (x, y), False) < 0:
            return None
        
        # Find closest edge points
        top = (self.points[0], self.points[1])
        right = (self.points[1], self.points[2])
        bottom = (self.points[2], self.points[3])
        left = (self.points[0], self.points[3])
        
        # Get vertical position (y coordinate)
        dist_top = point_to_line_distance((x, y), top[0], top[1])
        dist_bottom = point_to_line_distance((x, y), bottom[0], bottom[1])
        y_ratio = dist_top / (dist_top + dist_bottom)
        
        # Get horizontal position (x coordinate)
        dist_left = point_to_line_distance((x, y), left[0], left[1])
        dist_right = point_to_line_distance((x, y), right[0], right[1])
        x_ratio = dist_left / (dist_left + dist_right)
        
        # Convert to grid coordinates
        col = int(x_ratio * self.grid_size_x)
        row = int(y_ratio * self.grid_size_y)
        
        # Clamp values to grid bounds
        col = max(0, min(col, self.grid_size_x - 1))
        row = max(0, min(row, self.grid_size_y - 1))
        
        self.highlighted_cell = (row, col)
        return self.highlighted_cell
    
    def project_grid(self, frame):
        if len(self.points) != 4:
            return frame
            
        display_frame = frame.copy()
        
        # Helper function for linear interpolation
        def lerp(start, end, t):
            return tuple(int(a + (b - a) * t) for a, b in zip(start, end))
        
        # Get edges of polygon
        top = (self.points[0], self.points[1])
        right = (self.points[1], self.points[2])
        bottom = (self.points[2], self.points[3])
        left = (self.points[0], self.points[3])
        
        # Draw vertical lines
        for i in range(self.grid_size_x + 1):
            t = i / self.grid_size_x
            # Get points on top and bottom edges
            p1 = lerp(top[0], top[1], t)
            p2 = lerp(bottom[1], bottom[0], t)  # Note reversed order for bottom
            cv2.line(display_frame, p1, p2, (0, 255, 0), 2)
        
        # Draw horizontal lines
        for i in range(self.grid_size_y + 1):
            t = i / self.grid_size_y
            # Get points on left and right edges
            p1 = lerp(left[0], left[1], t)
            p2 = lerp(right[0], right[1], t)
            cv2.line(display_frame, p1, p2, (0, 255, 0), 2)
        
        # Highlight cell if center of motion was in the cell
        if self.highlighted_cell is not None:
            row, col = self.highlighted_cell
            
            # Calculate corners of highlighted cell
            t1 = col / self.grid_size_x
            t2 = (col + 1) / self.grid_size_x
            s1 = row / self.grid_size_y
            s2 = (row + 1) / self.grid_size_y
            
            # Get the four corners of the cell
            top_left = lerp(lerp(left[0], left[1], s1), lerp(right[0], right[1], s1), t1)
            top_right = lerp(lerp(left[0], left[1], s1), lerp(right[0], right[1], s1), t2)
            bottom_right = lerp(lerp(left[0], left[1], s2), lerp(right[0], right[1], s2), t2)
            bottom_left = lerp(lerp(left[0], left[1], s2), lerp(right[0], right[1], s2), t1)
            
            # Create a separate overlay image
            overlay = display_frame.copy()

            # Draw the filled polygon on the overlay
            corners = np.array([top_left, top_right, bottom_right, bottom_left], dtype=np.int32)
            cv2.fillPoly(overlay, [corners], (0, 0, 255))

            # Define transparency (alpha). 0.0 is fully transparent, 1.0 is fully opaque
            alpha = 0.5  # This gives 50% transparency

            # Blend the overlay with the original image
            cv2.addWeighted(overlay, alpha, display_frame, 1 - alpha, 0, display_frame)
        
        return display_frame
    
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

class KeyboardListener:
    def __init__(self, command_queue):
        self.running = True
        self.command_queue = command_queue
        self.listener = keyboard.Listener(on_press=self.on_press)
        self.listener.start()

    def on_press(self, key):
        try:
            if key == Key.enter:
                # Send command to main thread
                self.command_queue.put('track')
            elif key == Key.esc:
                # Send exit command
                self.command_queue.put('exit')
                self.running = False
                return False  # Stop listener
            elif hasattr(key, 'char') and key.char == '`':
                # Send abort command
                self.command_queue.put('abort')
        except AttributeError:
            pass

    def stop(self):
        self.running = False
        self.listener.stop()

def track_motion(camera1_id, camera2_id, grid, command_queue):
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
        # Check for abort command
        try:
            command = command_queue.get_nowait()
            if command in ['abort', 'exit']:
                print("Motion tracking stopped")
                camera1.stop()
                camera2.stop()
                camera1.join()
                camera2.join()
                return None if command == 'abort' else 'exit'
        except Empty:
            pass

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
        max_size2 = cv2.getTrackbarPos('Max Size', 'Camera 2') * width2 * height2
        
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
                if contour_area < min_size2 or contour_area > max_size2:
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
                
                # Get the grid coordinates
                grid_coordinate = grid.get_grid_coordinate(center[0], center[1])

                if grid_coordinate is not None:
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
                    return grid_coordinate
            
        # Project grid onto camera 2
        display_frame2 = grid.draw_on_frame(display_frame2)

        # Show frames
        cv2.imshow('Camera 1', current_frame1)
        cv2.imshow('Camera 2', display_frame2)
        
        # Update previous frames
        prev_frame1 = gray1
        prev_frame2 = gray2

        # Render frames on windows
        cv2.waitKey(1)

if __name__ == "__main__":
    # Create command queue for thread communication
    command_queue = Queue()

    # Create windows and sliders
    cv2.namedWindow('Camera 1')
    cv2.createTrackbar('Threshold', 'Camera 1', 25, 100, nothing)
    cv2.createTrackbar('Min Size', 'Camera 1', 25, 100, nothing)
    cv2.createTrackbar('Trigger', 'Camera 1', 50, 100, nothing)

    cv2.namedWindow('Camera 2')
    cv2.createTrackbar('Threshold', 'Camera 2', 50, 100, nothing)
    cv2.createTrackbar('Min Size', 'Camera 2', 25, 100, nothing)
    cv2.createTrackbar('Max Size', 'Camera 2', 90, 100, nothing)

    # Define keyboard layout
    keyboard_layout = np.array([
        ['q', 'w', 'e', 'r', 't', 'y', 'u', 'i', 'o', 'p'],
        ['a', 's', 'd', 'f', 'g', 'h', 'j', 'k', 'l', Key.backspace],
        ['z', 'x', 'c', 'v', 'b', 'n', 'm', Key.space, Key.space, Key.space]
    ])

    # Create mouse callback
    grid = GridDrawer(10, 3)
    cv2.setMouseCallback("Camera 2", grid.mouse_callback)

    # Load standby image
    standby_image = cv2.imread("./standby.png")
    cv2.imshow('Camera 1', standby_image)
    cv2.imshow('Camera 2', standby_image)
    cv2.waitKey(1)

    # Create keyboard controller
    virtual_keyboard = Controller()

    # Create and start keyboard listener
    keyboard_listener = KeyboardListener(command_queue)

    # Main loop
    running = True
    while running:
        # Handle OpenCV window updates
        cv2.waitKey(1)

        # Check for commands from keyboard listener
        try:
            command = command_queue.get_nowait()
            if command == 'track':
                result = track_motion(0, 2, grid, command_queue)  # Pass command_queue to track_motion
                if result == 'exit':
                    running = False
                elif result is not None:
                    print(f"{result}")
                    virtual_keyboard.press(keyboard_layout[result])
                    time.sleep(0.2 + int.from_bytes(os.urandom(1), 'big') / 1000)
                    virtual_keyboard.release(keyboard_layout[result])
                    print("why didn't it work")
            elif command == 'exit':
                running = False
        except Empty:
            pass

        time.sleep(0.1)  # Reduce CPU usage

    # Cleanup
    keyboard_listener.stop()
    cv2.destroyAllWindows()