import cv2
import numpy as np

class PolygonDrawer:
    def __init__(self):
        self.points = []
        self.is_drawing = False
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
            self.is_drawing = True
        elif event == cv2.EVENT_MOUSEMOVE:
            if len(self.points) == self.max_points:
                self.highlighted_cell = self.get_grid_coordinate(x, y)
    
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
            return (grid_row, grid_col)
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

def main():
    cap = cv2.VideoCapture(0)
    window_name = "Draw Polygon"
    cv2.namedWindow(window_name)
    drawer = PolygonDrawer()
    cv2.setMouseCallback(window_name, drawer.mouse_callback)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        display_frame = drawer.draw_on_frame(frame)
        cv2.imshow(window_name, display_frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('r'):
            drawer.points = []
            drawer.highlighted_cell = None
            drawer.counter = 0
        elif key == ord('q'):
            break
        # X and Y grid size controls
        elif key == ord('d'):  # Increase X grid size
            drawer.grid_size_x = min(drawer.grid_size_x + 1, 20)
        elif key == ord('a'):  # Decrease X grid size
            drawer.grid_size_x = max(drawer.grid_size_x - 1, 1)
        elif key == ord('w'):  # Increase Y grid size
            drawer.grid_size_y = min(drawer.grid_size_y + 1, 20)
        elif key == ord('s'):  # Decrease Y grid size
            drawer.grid_size_y = max(drawer.grid_size_y - 1, 1)
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()