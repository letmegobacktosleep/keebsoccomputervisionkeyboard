import cv2

# Open the default webcam
cap = cv2.VideoCapture(0)

# Try setting a desired resolution (optional)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

# Check if the camera opened successfully
if not cap.isOpened():
    print("Error: Cannot open webcam")
    exit()

# Read and display frames
while True:
    ret, frame = cap.read()
    
    if not ret:
        print("Error: Can't receive frame (stream end?). Exiting ...")
        break

    # Get the actual resolution
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))

    # Optionally, put the resolution text on the frame
    cv2.putText(frame, f"{width}x{height}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Show the frame
    cv2.imshow('Webcam Feed', frame)

    # Exit on 'q' key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
