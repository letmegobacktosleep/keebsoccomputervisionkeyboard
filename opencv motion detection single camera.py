# Import built-in libraries
import os
import time
import threading
from queue import Queue, Empty, Full
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
    gray = cv2.GaussianBlur(gray, (11, 11), 0)
    
    # Calculate difference between current and previous frame
    frame_diff = cv2.absdiff(prev_frame, gray)
    
    # Apply threshold to get binary image
    thresh = cv2.threshold(frame_diff, threshold, 255, cv2.THRESH_BINARY)[1]
    
    # Dilate threshold image to fill in holes
    thresh = cv2.dilate(thresh, None, iterations=2)
    
    # Find contours of moving objects
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    return gray, thresh, contours

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
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
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
            return self.frame_queue.get(timeout=0.01)
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

    def __init__(self, rows, cols):
        self.points = []
        self.max_points = 4
        self.cols = cols
        self.rows = rows
        self.transform_matrix = None
        self.inverse_matrix = None
        self.highlighted_cell = None
        self.counter = 0
        
    def mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            if len(self.points) < self.max_points:
                self.points.append((x, y))
                if len(self.points) == self.max_points:
                    self._compute_transform_matrices()
                    self.counter = 0
            else:
                self.points[self.counter] = (x, y)
                self.counter = (self.counter + 1) % self.max_points
                self._compute_transform_matrices()
    
    def _compute_transform_matrices(self):
        """Compute perspective transformation matrices."""
        if len(self.points) != 4:
            return
        
        # Source points (in clockwise order starting from top-left)
        src_points = np.array(self.points, dtype=np.float32)
        
        # Destination points (normalized grid in rectangle form)
        dst_points = np.array([
            [0, 0],                  # Top-left
            [self.cols, 0],          # Top-right
            [self.cols, self.rows],  # Bottom-right
            [0, self.rows]           # Bottom-left
        ], dtype=np.float32)
        
        # Compute the perspective transformation matrix
        self.transform_matrix = cv2.getPerspectiveTransform(src_points, dst_points)
        self.inverse_matrix = cv2.getPerspectiveTransform(dst_points, src_points)
    
    def is_point_in_grid(self, x, y):
        """Check if a point is inside the grid."""
        if len(self.points) != 4:
            return False
            
        # Check if point is inside polygon using ray casting
        poly = np.array(self.points)
        return cv2.pointPolygonTest(poly, (x, y), False) >= 0
    
    def get_grid_coordinate(self, x, y):
        """Convert screen coordinates to grid coordinates using perspective transformation."""
        if len(self.points) != 4 or self.transform_matrix is None:
            return None
            
        # Check if point is inside grid
        if not self.is_point_in_grid(x, y):
            return None
        
        # Transform the screen point to the normalized grid space
        point = np.array([[x, y]], dtype=np.float32)
        transformed_point = cv2.perspectiveTransform(point.reshape(-1, 1, 2), self.transform_matrix)
        transformed_point = transformed_point.reshape(-1, 2)[0]
        
        # Get grid coordinates
        col = int(transformed_point[0])
        row = int(transformed_point[1])
        
        # Clamp values to grid bounds
        col = max(0, min(col, self.cols - 1))
        row = max(0, min(row, self.rows - 1))
        
        self.highlighted_cell = (row, col)
        return self.highlighted_cell
    
    def project_grid(self, frame):
        if len(self.points) != 4 or self.transform_matrix is None:
            return frame
            
        display_frame = frame.copy()
        
        # Create grid lines in the normalized space
        grid_points = []
        
        # Vertical lines
        for i in range(self.cols + 1):
            grid_points.append(np.array([[i, 0], [i, self.rows]], dtype=np.float32))
        
        # Horizontal lines
        for i in range(self.rows + 1):
            grid_points.append(np.array([[0, i], [self.cols, i]], dtype=np.float32))
        
        # Transform grid lines to screen space and draw them
        for line in grid_points:
            # Reshape for perspective transform
            line_reshaped = line.reshape(-1, 1, 2)
            transformed_line = cv2.perspectiveTransform(line_reshaped, self.inverse_matrix)
            
            # Convert to pixel coordinates and draw line
            start_point = tuple(map(int, transformed_line[0][0]))
            end_point = tuple(map(int, transformed_line[1][0]))
            cv2.line(display_frame, start_point, end_point, (255, 255, 0), 2)
        
        # Highlight cell if a cell is selected
        if self.highlighted_cell is not None:
            row, col = self.highlighted_cell
            
            # Calculate the four corners of the cell in normalized space
            cell_corners = np.array([
                [col, row],               # Top-left
                [col + 1, row],           # Top-right
                [col + 1, row + 1],       # Bottom-right
                [col, row + 1]            # Bottom-left
            ], dtype=np.float32).reshape(-1, 1, 2)
            
            # Transform cell corners to screen space
            transformed_corners = cv2.perspectiveTransform(cell_corners, self.inverse_matrix)
            corners = np.array(transformed_corners.reshape(-1, 2), dtype=np.int32)
            
            # Create a separate overlay image
            overlay = display_frame.copy()

            # Draw the filled polygon on the overlay
            cv2.fillPoly(overlay, [corners], (0, 0, 255))

            # Define transparency (alpha)
            alpha = 0.5  # 50% transparency

            # Blend the overlay with the original image
            cv2.addWeighted(overlay, alpha, display_frame, 1 - alpha, 0, display_frame)
        
        return display_frame
    
    def draw_on_frame(self, frame):
        display_frame = frame.copy()
        
        if len(self.points) == self.max_points:
            display_frame = self.project_grid(display_frame)
        
        # Draw lines between points
        if len(self.points) > 1:
            for i in range(len(self.points) - 1):
                cv2.line(display_frame, self.points[i], self.points[i + 1], 
                        (255, 255, 0), 2)
                
            if len(self.points) == self.max_points:
                cv2.line(display_frame, self.points[-1], self.points[0], 
                        (255, 255, 0), 2)
                
        # Draw points
        for i, point in enumerate(self.points):
            color = (0, 255, 255) if (len(self.points) == self.max_points and i == self.counter) else (0, 0, 255)
            cv2.circle(display_frame, point, 7, color, -1)
            cv2.putText(display_frame, str(i), 
                       (point[0] + 10, point[1] + 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        # Display status information
        remaining = self.max_points - len(self.points)
        if remaining > 0:
            message = f"Click {remaining} more points"
        else:
            message = f"Next update: Point {self.counter} | Grid: {self.cols}x{self.rows}"
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
                # Send abort command
                self.command_queue.put('abort')
            elif hasattr(key, 'char') and key.char == '`':
                # Send exit command
                self.command_queue.put('exit')
                self.running = False
                return False  # Stop listener
        except AttributeError:
            pass

    def stop(self):
        self.running = False
        self.listener.stop()

def track_motion(camera2_id, grid, command_queue, loop_forever=False):
    # Start camera threads
    camera2 = CameraThread(camera2_id)
    camera2.start()

    # Wait for first frames
    while True:
        frame = camera2.get_frame()
        if frame is not None:
            break
        time.sleep(0.1)

    # Initialize previous frames
    prev_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    prev_frame = cv2.GaussianBlur(prev_frame, (21, 21), 0)
    
    # Get frame dimensions
    height2 = prev_frame.shape[0]
    width2 = prev_frame.shape[1]
    
    # Store reference frame from camera
    reference_frame = prev_frame.copy()
    reference_frame_counter = 0
    display_frame = None

    # Movement path
    movement_path = []

    while True:
        # Check for abort command
        try:
            command = command_queue.get_nowait()
            if command in ['abort', 'exit']:
                print("Motion tracking stopped")
                camera2.stop()
                camera2.join()
                return command
        except Empty:
            pass

        # Get current frames from queues
        current_frame = camera2.get_frame()
        
        if current_frame is None:
            continue

        # Get current threshold values
        threshold = cv2.getTrackbarPos('Threshold', 'Camera')
        min_area = cv2.getTrackbarPos('Min Area', 'Camera') ** 2
        min_size = cv2.getTrackbarPos('Min Size', 'Camera') ** 2
        max_size = cv2.getTrackbarPos('Max Size', 'Camera') ** 2
        r_frame_timeout = cv2.getTrackbarPos('Ref Timeout', 'Camera')
        n_frames_motion = cv2.getTrackbarPos('Min Frames', 'Camera')
        
        # Compare frames for both cameras
        gray2, _, contours1 = compare_frames(prev_frame, current_frame, threshold)

        # Copy frame for camera
        display_frame = current_frame.copy()
        
        # Check for motion in camera
        motion_detected2 = False
        for contour in contours1:
            # Calculate contour area and center
            area = cv2.contourArea(contour)
            if area < min_area:
                continue
                
            M = cv2.moments(contour)
            if M["m00"] == 0:
                continue
                
            # Get center of contour
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            
            # Check if center is within grid
            if grid.is_point_in_grid(cx, cy):
                # Show rectangle
                x, y, w, h = cv2.boundingRect(contour)
                cv2.rectangle(display_frame, (x, y), (x + w, y + h), (0, 255, 0), 1)
                
                # Set to true
                motion_detected2 = True
        
        # When no motion in camera
        if not motion_detected2:
            
            # No motion in last few frames
            if reference_frame_counter > r_frame_timeout:

                # Update reference frame
                reference_frame = gray2.copy()
                cv2.putText(display_frame, "Reference Frame Captured", (10, 70),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                
                # If there are was motion in more than a few frames
                if (len(movement_path) > n_frames_motion):

                    # Get highest coordinate (lowest Y value)
                    center = movement_path[0]
                    for i in range(len(movement_path)):
                        if movement_path[i][1] < center[1]:
                            center = movement_path[i]
                    
                    # Get the grid coordinates
                    grid_coordinate = grid.get_grid_coordinate(*center)

                    if grid_coordinate is not None:
                        # Press corresponding key in keyboard_layout
                        virtual_keyboard.press(keyboard_layout[grid_coordinate])
                        time.sleep(0.2 + int.from_bytes(os.urandom(1), 'big') / 2000)
                        virtual_keyboard.release(keyboard_layout[grid_coordinate])

                        if not loop_forever:

                            # Project grid onto camera
                            display_frame = grid.draw_on_frame(display_frame)
                            cv2.imshow('Camera', display_frame)
                            cv2.waitKey(1)

                            # Stop cameras and return result
                            camera2.stop()
                            camera2.join()
                            return grid_coordinate
                
                # Reset the movement path
                movement_path = []
                        
            else:
                reference_frame_counter += 1
        else:
            reference_frame_counter = 0

        # Check for motion in camera against reference frame
        _, _, contours2 = compare_frames(reference_frame, current_frame, threshold)

        # Find the largest contour
        largest_area2 = 0
        largest_contour2 = None
        for contour in contours2:
            contour_area = cv2.contourArea(contour)
            if contour_area < min_size or contour_area > max_size:
                continue
            
            # Calculate center of contour
            M = cv2.moments(contour)
            if M["m00"] == 0:
                continue
                
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            
            # Is bigger than previous largest contour & within grid bounds
            if contour_area > largest_area2 and grid.is_point_in_grid(cx, cy):
                largest_area2 = contour_area
                largest_contour2 = contour
        
        # If there is a valid contour
        if largest_contour2 is not None:
            x, y, w, h = cv2.boundingRect(largest_contour2)
            
            # Calculate center
            M = cv2.moments(largest_contour2)
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            center = (cx, cy)

            # Draw box
            cv2.rectangle(display_frame, (x, y), (x + w, y + h), (0, 0, 255), 1)
            cv2.putText(display_frame, f'{center}', (x, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        
            # Add to movement path
            movement_path.append(center)

        # Draw movement path
        if len(movement_path) > 0:
            for i in range(len(movement_path) - 1):
                cv2.line(
                    display_frame,
                    movement_path[i],
                    movement_path[i+1],
                    (0, 0, 255),
                    3
                )
            
        # Project grid onto camera
        display_frame = grid.draw_on_frame(display_frame)

        # Show frames
        cv2.imshow('Camera', display_frame)
        
        # Update previous frames
        prev_frame = gray2

        # Render frames on windows
        cv2.waitKey(1)

if __name__ == "__main__":
    # Create command queue for thread communication
    command_queue = Queue()

    # Create windows and sliders
    cv2.namedWindow('Camera')
    cv2.createTrackbar('Threshold', 'Camera', 25, 100, nothing)
    cv2.createTrackbar('Min Area', 'Camera', 4, 100, nothing)  # Renamed from AAAAAAAA to Min Area
    cv2.createTrackbar('Min Size', 'Camera', 4, 100, nothing)
    cv2.createTrackbar('Max Size', 'Camera', 100, 1000, nothing)
    cv2.createTrackbar('Ref Timeout', 'Camera', 5, 10, nothing)
    cv2.createTrackbar('Min Frames', 'Camera', 1, 10, nothing)  

    # Define keyboard layout
    if True:
        keyboard_layout = np.array([
            ['q', 'w', 'e', 'r', 't', 'y', 'u', 'i', 'o', 'p'],
            ['a', 's', 'd', 'f', 'g', 'h', 'j', 'k', 'l', Key.backspace],
            ['z', 'x', 'c', 'v', 'b', 'n', 'm', Key.space, Key.space, Key.backspace]
        ])
    else:
        keyboard_layout = np.array([
            ['7', '8', '9'],
            ['4', '5', '6'],
            ['1', '2', '3'],
            ['0', Key.backspace, Key.backspace]
        ])

    # Create mouse callback
    grid = GridDrawer(*np.shape(keyboard_layout))
    cv2.setMouseCallback("Camera", grid.mouse_callback)

    # Load standby image
    standby_image = cv2.imread("./standby.png")
    cv2.imshow('Camera', standby_image)
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
                result = track_motion(0, grid, command_queue, loop_forever=True)
                if result == 'exit':
                    running = False
                elif result == 'abort':
                    pass
                elif result is not None:
                    print(f"{result}")

            elif command == 'exit':
                running = False
        except Empty:
            pass

        time.sleep(0.1)  # Reduce CPU usage

    # Cleanup
    keyboard_listener.stop()
    cv2.destroyAllWindows()