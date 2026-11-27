import cv2
import mediapipe as mp
from pathlib import Path


# --------------------------------
# Normalize landmarks
# --------------------------------
import math


def normalize_landmarks(hand_landmarks):
    # Step 1: Use wrist as the reference point
    wrist = hand_landmarks[0]

    centered = []

    for landmark in hand_landmarks:
        x = landmark.x - wrist.x
        y = landmark.y - wrist.y
        z = landmark.z - wrist.z

        centered.append([x, y, z])

    # Step 2: Calculate hand scale
    # Distance from wrist (0) to middle finger MCP (9)
    middle_mcp = centered[9]

    scale = math.sqrt(
        middle_mcp[0] ** 2 +
        middle_mcp[1] ** 2 +
        middle_mcp[2] ** 2
    )

    # Prevent division by zero
    if scale == 0:
        scale = 1

    # Step 3: Divide every coordinate by the scale
    normalized = []

    for point in centered:
        x = point[0] / scale
        y = point[1] / scale
        z = point[2] / scale

        normalized.append([x, y, z])

    return normalized


# --------------------------------
# MediaPipe setup
# --------------------------------

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
RunningMode = mp.tasks.vision.RunningMode

model_path = Path(__file__).parent / "models" / "hand_landmarker.task"

options = HandLandmarkerOptions(
    base_options=BaseOptions(
        model_asset_path=str(model_path)
    ),
    running_mode=RunningMode.IMAGE,
    num_hands=1
)


# --------------------------------
# Webcam
# --------------------------------

camera = cv2.VideoCapture(0)

with HandLandmarker.create_from_options(options) as landmarker:

    while True:

        success, frame = camera.read()

        if not success:
            print("Could not access camera")
            break

        rgb_frame = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )

        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb_frame
        )

        result = landmarker.detect(mp_image)

        # --------------------------------
        # Process hand
        # --------------------------------

        if result.hand_landmarks:

            hand = result.hand_landmarks[0]

            normalized = normalize_landmarks(hand)

            print("\n--- Normalized landmarks ---")

            for i, point in enumerate(normalized):

                print(
                    f"{i}: "
                    f"x={point[0]:.3f}, "
                    f"y={point[1]:.3f}, "
                    f"z={point[2]:.3f}"
                )

            # --------------------------------
            # Draw landmarks
            # --------------------------------

            for landmark in hand:

                x = int(
                    landmark.x * frame.shape[1]
                )

                y = int(
                    landmark.y * frame.shape[0]
                )

                cv2.circle(
                    frame,
                    (x, y),
                    5,
                    (0, 255, 0),
                    -1
                )

        cv2.imshow(
            "Normalized Hand",
            frame
        )

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break


camera.release()
cv2.destroyAllWindows()