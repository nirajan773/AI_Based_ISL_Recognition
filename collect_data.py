import cv2
import csv
import math
import time
import mediapipe as mp
from pathlib import Path


# ============================================================
# SETTINGS
# ============================================================

SAMPLES_PER_LETTER = 300
CAPTURE_INTERVAL = 0.15  # Seconds between samples

CSV_FILE = Path(__file__).parent / "hand_landmarks_isl.csv"
MODEL_PATH = Path(__file__).parent / "models" / "hand_landmarker.task"


# ============================================================
# NORMALIZE ONE HAND
# ============================================================

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

    normalized = []

    for point in centered:
        normalized.extend([
            point[0] / scale,
            point[1] / scale,
            point[2] / scale
        ])

    return normalized


def zero_hand():
    return [0.0] * 63


# ============================================================
# CSV HEADERS
# ============================================================

headers = []

for hand_name in ["left", "right"]:
    for i in range(21):
        headers.extend([
            f"{hand_name}_x{i}",
            f"{hand_name}_y{i}",
            f"{hand_name}_z{i}"
        ])

headers.append("label")


# ============================================================
# MEDIAPIPE SETUP
# ============================================================

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
RunningMode = mp.tasks.vision.RunningMode

options = HandLandmarkerOptions(
    base_options=BaseOptions(
        model_asset_path=str(MODEL_PATH)
    ),
    running_mode=RunningMode.IMAGE,
    num_hands=2
)

landmarker = HandLandmarker.create_from_options(options)


# ============================================================
# COLLECTION STATE
# ============================================================

current_label = None
collecting = False
sample_count = 0
last_capture_time = 0

camera = cv2.VideoCapture(0)

if not camera.isOpened():
    landmarker.close()
    raise RuntimeError("Could not open webcam.")


print("\n======================================")
print("       ISL AUTOMATIC DATA COLLECTOR")
print("======================================")
print("A-E : Select a letter")
print("C   : Start collecting 300 samples")
print("Q   : Quit")
print("======================================\n")


# ============================================================
# MAIN PROGRAM
# ============================================================

try:
    with open(CSV_FILE, "a", newline="") as file:

        writer = csv.writer(file)

        # Write the new header to a fresh file.
        if CSV_FILE.stat().st_size == 0:
            writer.writerow(headers)
            file.flush()

        while True:
            success, frame = camera.read()

            if not success:
                print("Could not read webcam frame.")
                break

            # Mirror the webcam image.
            frame = cv2.flip(frame, 1)

            rgb_frame = cv2.cvtColor(
                frame, cv2.COLOR_BGR2RGB
            )

            mp_image = mp.Image(
                image_format=mp.ImageFormat.SRGB,
                data=rgb_frame
            )

            result = landmarker.detect(mp_image)
            hands_detected = len(result.hand_landmarks)

            # Draw all detected landmarks.
            for hand in result.hand_landmarks:
                for landmark in hand:
                    x = int(landmark.x * frame.shape[1])
                    y = int(landmark.y * frame.shape[0])

                    cv2.circle(
                        frame, (x, y), 5,
                        (0, 255, 0), -1
                    )

            # Assign detected hands to left/right slots.
            left_hand = None
            right_hand = None

            for i, hand in enumerate(result.hand_landmarks):

                side = ""

                if i < len(result.handedness):
                    categories = result.handedness[i]

                    if categories:
                        side = categories[0].category_name.lower()

                if side == "left" and left_hand is None:
                    left_hand = hand

                elif side == "right" and right_hand is None:
                    right_hand = hand

                else:
                    # Fallback for an unexpected handedness result.
                    if left_hand is None:
                        left_hand = hand
                    elif right_hand is None:
                        right_hand = hand

            # Prepare the feature row.
            left_features = (
                normalize_landmarks(left_hand)
                if left_hand is not None
                else zero_hand()
            )

            right_features = (
                normalize_landmarks(right_hand)
                if right_hand is not None
                else zero_hand()
            )

            # Automatic sample capture.
            now = time.time()

            if (
                collecting
                and current_label is not None
                and hands_detected > 0
                and now - last_capture_time >= CAPTURE_INTERVAL
            ):
                row = (
                    left_features
                    + right_features
                    + [current_label]
                )

                writer.writerow(row)
                file.flush()

                sample_count += 1
                last_capture_time = now

                print(
                    f"\r{current_label}: "
                    f"{sample_count}/{SAMPLES_PER_LETTER}",
                    end="",
                    flush=True
                )

                if sample_count >= SAMPLES_PER_LETTER:
                    collecting = False
                    print(
                        f"\nFinished collecting letter "
                        f"{current_label}!"
                    )
                    print("Choose the next letter and press C.")

            # Display information.
            label_text = current_label or "NONE"
            mode_text = "COLLECTING" if collecting else "READY"

            cv2.putText(
                frame, f"Letter: {label_text}",
                (20, 40), cv2.FONT_HERSHEY_SIMPLEX,
                0.8, (0, 255, 0), 2
            )

            cv2.putText(
                frame, f"Hands detected: {hands_detected}",
                (20, 75), cv2.FONT_HERSHEY_SIMPLEX,
                0.7, (0, 255, 0), 2
            )

            cv2.putText(
                frame, f"Status: {mode_text}",
                (20, 110), cv2.FONT_HERSHEY_SIMPLEX,
                0.7, (0, 0, 255) if collecting else (255, 255, 0), 2
            )

            cv2.putText(
                frame, f"Samples: {sample_count}/{SAMPLES_PER_LETTER}",
                (20, 145), cv2.FONT_HERSHEY_SIMPLEX,
                0.7, (0, 255, 0), 2
            )

            cv2.imshow("ISL Dataset Collector", frame)

            key = cv2.waitKey(1) & 0xFF

            # Select a letter.
            if key in [ord("a"), ord("b"), ord("c"),
                       ord("d"), ord("e")]:

                current_label = chr(key).upper()
                collecting = False
                sample_count = 0

                print(f"\nSelected letter: {current_label}")
                print("Make the correct ISL sign, then press C.")

            # Start automatic collection.
            elif key == ord("c"):

                if current_label is None:
                    print("\nSelect a letter first using A-E.")

                elif collecting:
                    print("\nCollection is already running.")

                else:
                    sample_count = 0
                    last_capture_time = 0
                    collecting = True

                    print(
                        f"\nCollecting {SAMPLES_PER_LETTER} "
                        f"samples for {current_label}..."
                    )
                    print("Hold the sign steady and keep your hands visible.")

            elif key == ord("q"):
                break

finally:
    camera.release()
    cv2.destroyAllWindows()
    landmarker.close()

print(f"\nDataset saved at:\n{CSV_FILE}")
print("Collector closed.")