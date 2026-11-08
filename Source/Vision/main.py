import time
import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# Hand connection drawing utility
from mediapipe.tasks.python.vision import HandLandmarksConnections

HAND_CONNECTIONS = HandLandmarksConnections.HAND_CONNECTIONS

# Set up the hand landmarker
base_options = python.BaseOptions(model_asset_path="VisionChords/Assets/hand_landmarker.task")
options = vision.HandLandmarkerOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.VIDEO,
    num_hands=2,
)
landmarker = vision.HandLandmarker.create_from_options(options)

# OpenCV video capture
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 600)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 500)

while True:
    success, frame = cap.read()
    if not success:
        continue

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

    # Timestamps must increase with every frame
    timestamp_ms = int(time.time() * 1000)
    result = landmarker.detect_for_video(mp_image, timestamp_ms)

    h, w, _ = frame.shape
    for hand_landmarks in result.hand_landmarks:
        # Convert all landmarks to pixel coordinates once
        points = [(int(lm.x * w), int(lm.y * h)) for lm in hand_landmarks]

        # Draw the lines first
        for connection in HAND_CONNECTIONS:
            cv2.line(frame, points[connection.start], points[connection.end], (255, 255, 255), 2)

        # Then draw the points on top
        for p in points:
            cv2.circle(frame, p, 4, (0, 255, 0), -1)

    cv2.imshow("Vision chords", frame)

    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()