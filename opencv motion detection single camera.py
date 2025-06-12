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
    """Dummy function for trackbar callbacks."""
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


def get_rightmost_point(contour):
    """
    Get the rightmost x and center y of a contour.
    
    Args:
        contour: OpenCV contour
        
    Returns:
        tuple: (rightmost_x, center_y) coordinates
    """
    # Get rightmost x coordinate
    rightmost_x = contour[contour[:, :, 0].argmax()][0][0]
    
    # Get center y coordinate
    M = cv2.moments(contour)
    center_y = int(M["m01"] / M["m00"]) if M["m00"] != 0 else 0
    
    return (int(rightmost_x), int(center_y))


class CameraThread(threading.Thread):
    """Thread for handling camera capture."""
    
    def __init__(self, camera_id):
        super().__init__()
        self.camera_id = camera_id
        self.cap = cv2.VideoCapture(camera_id)
        self.frame_queue = Queue(maxsize=2)
        self.running = True
        self._stop_lock = threading.Lock()
        self._camera_lock = threading.Lock()

    def run(self):
        """Main thread loop for capturing frames."""
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
                # Remove old frame if queue is full
                if self.frame_queue.full():
                    try:
                        self.frame_queue.get_nowait()
                    except Empty:
                        pass
                
                # Add new frame
                try:
                    self.frame_queue.put_nowait(frame)
                except Full:
                    pass
            
            time.sleep(0.01)

    def get_frame(self):
        """Get the latest frame from the queue."""
        try:
            return self.frame_queue.get(timeout=0.01)
        except Empty:
            return None

    def stop(self):
        """Stop the camera thread and release resources."""
        with self._stop_lock:
            self.running = False
        
        # Give the thread time to exit
        time.sleep(0.1)
        
        with self._camera_lock:
            if self.cap is not None:
                self.cap.release()
                self.cap = None


class GridDrawer:
    """Handles grid drawing and coordinate transformation."""
    
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
        """Handle mouse clicks for grid setup."""
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
        return cv2.pointPolygonTest(poly, (int(x), int(y)), False) >= 0
    
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
        """Project the grid onto the frame."""
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
            line_reshaped = line.reshape(-1, 1, 2)
            transformed_line = cv2.perspectiveTransform(line_reshaped, self.inverse_matrix)
            
            start_point = tuple(map(int, transformed_line[0][0]))
            end_point = tuple(map(int, transformed_line[1][0]))
            cv2.line(display_frame, start_point, end_point, (255, 255, 0), 2)
        
        # Highlight selected cell
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
            
            # Create overlay with transparency
            overlay = display_frame.copy()
            cv2.fillPoly(overlay, [corners], (0, 0, 255))
            alpha = 0.5
            cv2.addWeighted(overlay, alpha, display_frame, 1 - alpha, 0, display_frame)
        
        return display_frame
    
    def draw_on_frame(self, frame):
        """Draw the grid and setup points on the frame."""
        display_frame = frame.copy()
        
        if len(self.points) == self.max_points:
            display_frame = self.project_grid(display_frame)
        
        # Draw lines between points
        if len(self.points) > 1:
            for i in range(len(self.points) - 1):
                cv2.line(display_frame, self.points[i], self.points[i + 1], (255, 255, 0), 2)
                
            if len(self.points) == self.max_points:
                cv2.line(display_frame, self.points[-1], self.points[0], (255, 255, 0), 2)
                
        # Draw points
        for i, point in enumerate(self.points):
            color = (0, 255, 255) if (len(self.points) == self.max_points and i == self.counter) else (0, 0, 255)
            cv2.circle(display_frame, point, 7, color, -1)
            cv2.putText(display_frame, str(i), (point[0] + 10, point[1] + 10),
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
    """Handles keyboard input in a separate thread."""
    
    def __init__(self, command_queue):
        self.running = True
        self.command_queue = command_queue
        self.listener = keyboard.Listener(on_press=self.on_press)
        self.listener.start()

    def on_press(self, key):
        """Handle key press events."""
        try:
            if key == Key.enter:
                self.command_queue.put('track')
            elif key == Key.esc:
                self.command_queue.put('abort')
            elif hasattr(key, 'char') and key.char == '`':
                self.command_queue.put('exit')
                self.running = False
                return False  # Stop listener
        except AttributeError:
            pass

    def stop(self):
        """Stop the keyboard listener."""
        self.running = False
        self.listener.stop()


def track_motion(camera_id, grid, command_queue, loop_forever=False):
    """Main motion tracking function."""
    # Start camera thread
    camera = CameraThread(camera_id)
    camera.start()

    # Wait for first frame
    while True:
        frame = camera.get_frame()
        if frame is not None:
            break
        time.sleep(0.1)

    # Initialize previous frame
    prev_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    prev_frame = cv2.GaussianBlur(prev_frame, (21, 21), 0)
    
    # Store reference frame
    reference_frame = prev_frame.copy()
    reference_frame_counter = 0
    
    # Movement path to store rightmost points
    movement_path = []

    while True:
        # Check for abort command
        try:
            command = command_queue.get_nowait()
            if command in ['abort', 'exit']:
                print("Motion tracking stopped")
                camera.stop()
                camera.join()
                return command
        except Empty:
            pass

        # Get current frame
        current_frame = camera.get_frame()
        if current_frame is None:
            continue

        # Get trackbar values
        threshold = cv2.getTrackbarPos('Threshold', 'Camera')
        min_area = cv2.getTrackbarPos('Min Area', 'Camera') ** 2
        min_size = cv2.getTrackbarPos('Min Size', 'Camera') ** 2
        max_size = cv2.getTrackbarPos('Max Size', 'Camera') ** 2
        ref_timeout = cv2.getTrackbarPos('Ref Timeout', 'Camera')
        min_frames = cv2.getTrackbarPos('Min Frames', 'Camera')
        
        # Compare frames for motion detection
        gray, _, contours = compare_frames(prev_frame, current_frame, threshold)
        display_frame = current_frame.copy()
        
        # Check for motion in current frame vs previous frame
        motion_detected = False
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < min_area:
                continue
                
            # Get rightmost point instead of center
            rightmost_point = get_rightmost_point(contour)
            
            # Check if rightmost point is within grid
            if grid.is_point_in_grid(*rightmost_point):
                x, y, w, h = cv2.boundingRect(contour)
                cv2.rectangle(display_frame, (x, y), (x + w, y + h), (0, 255, 0), 1)
                motion_detected = True
        
        # Handle reference frame updates when no motion
        if not motion_detected:
            if reference_frame_counter > ref_timeout:
                # Update reference frame
                reference_frame = gray.copy()
                cv2.putText(display_frame, "Reference Frame Captured", (10, 70),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                
                # Process movement path if sufficient motion was detected
                if len(movement_path) > min_frames:
                    # Find rightmost point (maximum X coordinate) instead of topmost
                    target_point = movement_path[0]
                    for point in movement_path:
                        if point[0] > target_point[0]:  # Compare X coordinates
                            target_point = point
                    
                    # Get grid coordinates for the rightmost point
                    grid_coordinate = grid.get_grid_coordinate(*target_point)

                    if grid_coordinate is not None:
                        # Press corresponding key
                        virtual_keyboard.press(keyboard_layout[grid_coordinate])
                        time.sleep(0.5 + int.from_bytes(os.urandom(1), 'big') / 1000)
                        virtual_keyboard.release(keyboard_layout[grid_coordinate])

                        if not loop_forever:
                            # Show result and exit
                            display_frame = grid.draw_on_frame(display_frame)
                            cv2.imshow('Camera', display_frame)
                            cv2.waitKey(1)
                            
                            camera.stop()
                            camera.join()
                            return grid_coordinate
                
                # Reset movement path
                movement_path = []
                        
            else:
                reference_frame_counter += 1
        else:
            reference_frame_counter = 0

        # Compare against reference frame for tracking
        _, _, ref_contours = compare_frames(reference_frame, current_frame, threshold)

        # Find the largest valid contour
        largest_area = 0
        largest_contour = None
        
        for contour in ref_contours:
            area = cv2.contourArea(contour)
            if area < min_size or area > max_size:
                continue
            
            # Get rightmost point
            rightmost_point = get_rightmost_point(contour)
            
            # Check if it's the largest contour within grid bounds
            if area > largest_area and grid.is_point_in_grid(*rightmost_point):
                largest_area = area
                largest_contour = contour
        
        # Process the largest contour
        if largest_contour is not None:
            x, y, w, h = cv2.boundingRect(largest_contour)
            rightmost_point = get_rightmost_point(largest_contour)

            # Draw bounding box and rightmost point
            cv2.rectangle(display_frame, (x, y), (x + w, y + h), (0, 0, 255), 1)
            cv2.circle(display_frame, rightmost_point, 5, (255, 0, 0), -1)
            cv2.putText(display_frame, f'{rightmost_point}', (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        
            # Add rightmost point to movement path
            movement_path.append(rightmost_point)

        # Draw movement path
        if len(movement_path) > 1:
            for i in range(len(movement_path) - 1):
                cv2.line(display_frame, movement_path[i], movement_path[i + 1], (0, 0, 255), 3)
            
        # Project grid onto frame
        display_frame = grid.draw_on_frame(display_frame)
        cv2.imshow('Camera', display_frame)
        
        # Update previous frame
        prev_frame = gray
        cv2.waitKey(1)


if __name__ == "__main__":
    # Create command queue for thread communication
    command_queue = Queue()

    # Create windows and trackbars
    cv2.namedWindow('Camera')
    cv2.createTrackbar('Threshold', 'Camera', 25, 100, nothing)
    cv2.createTrackbar('Min Area', 'Camera', 5, 100, nothing)
    cv2.createTrackbar('Min Size', 'Camera', 5, 100, nothing)
    cv2.createTrackbar('Max Size', 'Camera', 100, 1000, nothing)
    cv2.createTrackbar('Ref Timeout', 'Camera', 5, 30, nothing)
    cv2.createTrackbar('Min Frames', 'Camera', 3, 10, nothing)

    # Define keyboard layout
    if False:
        keyboard_layout = np.array([
            ['q', 'w', 'e', 'r', 't', 'y', 'u', 'i', 'o', 'p'],
            ['a', 's', 'd', 'f', 'g', 'h', 'j', 'k', 'l', Key.backspace],
            ['z', 'x', 'c', 'v', 'b', 'n', 'm', Key.space, Key.space, Key.space]
        ])
    else:
        keyboard_layout = np.array([
            ['7', '8', '9'],
            ['4', '5', '6'],
            ['1', '2', '3'],
            ['0', Key.backspace, Key.backspace]
        ])

    # Create grid drawer and set mouse callback
    grid = GridDrawer(*np.shape(keyboard_layout))
    cv2.setMouseCallback("Camera", grid.mouse_callback)

    # Load and display standby image
    try:
        standby_image = cv2.imread("./standby.png")
        if standby_image is not None:
            cv2.imshow('Camera', standby_image)
        else:
            # Create a simple standby image if file doesn't exist
            standby_image = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.putText(standby_image, "Press ENTER to start tracking", (50, 240),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            cv2.imshow('Camera', standby_image)
    except:
        # Fallback if image loading fails
        standby_image = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.putText(standby_image, "Press ENTER to start tracking", (50, 240),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.imshow('Camera', standby_image)
    
    cv2.waitKey(1)

    # Create virtual keyboard controller
    virtual_keyboard = Controller()

    # Create and start keyboard listener
    keyboard_listener = KeyboardListener(command_queue)

    # Main application loop
    running = True
    while running:
        cv2.waitKey(1)

        # Check for commands
        try:
            command = command_queue.get_nowait()
            if command == 'track':
                result = track_motion(1, grid, command_queue, loop_forever=True)
                if result == 'exit':
                    running = False
                elif result == 'abort':
                    pass
                elif result is not None:
                    print(f"Grid coordinate: {result}")

            elif command == 'exit':
                running = False
        except Empty:
            pass

        time.sleep(0.1)

    # Cleanup
    keyboard_listener.stop()
    cv2.destroyAllWindows()