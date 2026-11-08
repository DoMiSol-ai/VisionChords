import time
import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# Hand connection drawing utility
from mediapipe.tasks.python.vision import HandLandmarksConnections

HAND_CONNECTIONS = HandLandmarksConnections.HAND_CONNECTIONS

### Fluidsynth setup:
import fluidsynth
import chords as chords

fs = fluidsynth.Synth(gain=1)

fs.start(driver="coreaudio")
sfid = fs.sfload("VisionChords/Assets/soundfont.sf2")

fs.program_select(0, sfid, 0, 48)
fs.cc(0, 7, 127)  # Channel volume to max
current_chord = 0

### Finger counting utility:
TIPS = [8, 12, 16, 20]
PIPS = [6, 10, 14, 18]

def count_fingers(landmarks):
    return sum(1 for tip, pip in zip(TIPS, PIPS) if landmarks[tip].y < landmarks[pip].y)

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

last_count, stable_frames = -1, 0

while True:
    success, frame = cap.read()
    if not success:
        continue

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

    count = 0

    # Timestamps must increase with every frame
    timestamp_ms = int(time.time() * 1000)
    result = landmarker.detect_for_video(mp_image, timestamp_ms)

    h, w, _ = frame.shape
    for hand_landmarks in result.hand_landmarks:

        hand_landmarks = result.hand_landmarks[0]
        count = count_fingers(hand_landmarks)

        # Convert all landmarks to pixel coordinates once
        points = [(int(lm.x * w), int(lm.y * h)) for lm in hand_landmarks]

        # Draw the lines first
        for connection in HAND_CONNECTIONS:
            cv2.line(frame, points[connection.start], points[connection.end], (255, 255, 255), 2)

        # Then draw the points on top
        for p in points:
            cv2.circle(frame, p, 4, (0, 255, 0), -1)

    # Only switch once the count has been steady for a few frames
    if count == last_count:
        stable_frames += 1
    else:
        last_count, stable_frames = count, 0

    if stable_frames == 3:
        current_chord = chords.set_chord(fs, count)

    cv2.imshow("Vision chords", frame)

    if cv2.waitKey(1) == ord('q'):
        break

fs.delete()
cap.release()
cv2.destroyAllWindows()