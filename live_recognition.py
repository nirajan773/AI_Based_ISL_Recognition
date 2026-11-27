
import cv2
import math
import mediapipe as mp
import pandas as pd
import joblib

from pathlib import Path


# ---------------------------------
# PROJECT PATHS
# ---------------------------------
BASE_DIR = Path(__file__).parent

MODEL_PATH = BASE_DIR / "models" / "isl_random_forest.joblib"
HAND_MODEL_PATH = BASE_DIR / "models" / "hand_landmarker.task"


# ---------------------------------
# LOAD TRAINED CLASSIFIER
# ---------------------------------
print("Loading ISL recognition model...")

model = joblib.load(MODEL_PATH)

print("ISL model loaded successfully!")


# ---------------------------------
# NORMALIZE LANDMARKS
# Same method used during preprocessing
# ---------------------------------
def normalize_landmarks(hand_landmarks):

    wrist = hand_landmarks[0]

    centered = []

    for landmark in hand_landmarks:

        centered.append([
            landmark.x - wrist.x,
            landmark.y - wrist.y,
            landmark.z - wrist.z
        ])

    middle_mcp = centered[9]

    scale = math.sqrt(
        middle_mcp[0] ** 2 +
        middle_mcp[1] ** 2 +
        middle_mcp[2] ** 2
    )

    if scale == 0:
        scale = 1

    features = []

    for point in centered:

        features.extend([
            point[0] / scale,
            point[1] / scale,
            point[2] / scale
        ])

    return features


# ---------------------------------
# HAND CONNECTIONS FOR DRAWING
# ---------------------------------
CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20),
    (0, 17)
]


def draw_hand(frame, hand_landmarks):

    height, width, _ = frame.shape

    points = []

    for landmark in hand_landmarks:

        x = int(landmark.x * width)
        y = int(landmark.y * height)

        points.append((x, y))

        cv2.circle(
            frame,
            (x, y),
            5,
            (0, 255, 0),
            -1
        )

    for start, end in CONNECTIONS:

        cv2.line(
            frame,
            points[start],
            points[end],
            (255, 0, 0),
            2
        )


# ---------------------------------
# START MEDIAPIPE HAND LANDMARKER
# ---------------------------------
BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
RunningMode = mp.tasks.vision.RunningMode

options = HandLandmarkerOptions(
    base_options=BaseOptions(
        model_asset_path=str(HAND_MODEL_PATH)
    ),
    running_mode=RunningMode.VIDEO,
    num_hands=2,
    min_hand_detection_confidence=0.4,
    min_hand_presence_confidence=0.4,
    min_tracking_confidence=0.4
)


# ---------------------------------
# START WEBCAM
# ---------------------------------
cap = cv2.VideoCapture(0)

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

if not cap.isOpened():
    print("ERROR: Could not open webcam.")
    exit()

print("Starting webcam...")
print("Press Q to quit.")


# ---------------------------------
# REAL-TIME RECOGNITION
# ---------------------------------
with HandLandmarker.create_from_options(options) as landmarker:

    frame_timestamp = 0

    while True:

        success, frame = cap.read()

        if not success:
            print("Failed to read webcam frame.")
            break

        # Convert image for MediaPipe
        rgb_frame = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )

        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb_frame
        )

        frame_timestamp += 33

        result = landmarker.detect_for_video(
            mp_image,
            frame_timestamp
        )

        # ---------------------------------
        # PROCESS DETECTED HANDS
        # ---------------------------------
        if result.hand_landmarks:

            hands = result.hand_landmarks

            features = []

            # First detected hand
            features.extend(
                normalize_landmarks(hands[0])
            )

            # Second detected hand
            if len(hands) >= 2:

                features.extend(
                    normalize_landmarks(hands[1])
                )

            else:

                features.extend([0.0] * 63)

            # Draw detected hands
            for hand in hands:
                draw_hand(frame, hand)

            # Prepare input in the same column format
            # used during model training
            if hasattr(model, "feature_names_in_"):
                input_data = pd.DataFrame(
                    [features],
                    columns=model.feature_names_in_
                )
            else:
                input_data = pd.DataFrame([features])

            # Predict alphabet
            probabilities = model.predict_proba(input_data)[0]

            best_index = probabilities.argmax()

            predicted_letter = model.classes_[best_index]

            confidence = probabilities[best_index]

            # Avoid displaying low-confidence predictions
            if confidence >= 0.60:

                display_text = (
                    f"Letter: {predicted_letter} "
                    f"({confidence * 100:.1f}%)"
                )

            else:

                display_text = "Uncertain"

        else:

            display_text = "No hand detected"

        # ---------------------------------
        # DISPLAY RESULT
        # ---------------------------------
        cv2.rectangle(
            frame,
            (10, 10),
            (430, 75),
            (0, 0, 0),
            -1
        )

        cv2.putText(
            frame,
            display_text,
            (20, 52),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (0, 255, 0),
            2
        )

        cv2.imshow(
            "ISL Alphabet Recognition",
            frame
        )

        # Press Q to exit
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break


# ---------------------------------
# CLEANUP
# ---------------------------------
cap.release()
cv2.destroyAllWindows()

print("ISL recognition stopped.")
