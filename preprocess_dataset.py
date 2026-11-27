
import cv2
import csv
import math
import mediapipe as mp

from pathlib import Path


# ==================================================
# PATHS
# ==================================================

BASE_DIR = Path(__file__).parent

DATASET_DIR = BASE_DIR

MODEL_PATH = BASE_DIR / "models" / "hand_landmarker.task"

OUTPUT_DIR = BASE_DIR / "processed"


# ==================================================
# NORMALIZE LANDMARKS
# ==================================================

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


# ==================================================
# CSV HEADERS
# ==================================================

headers = []

for hand_number in [1, 2]:

    for i in range(21):

        headers.extend([
            f"hand{hand_number}_x{i}",
            f"hand{hand_number}_y{i}",
            f"hand{hand_number}_z{i}"
        ])

headers.append("label")


# ==================================================
# MEDIAPIPE SETUP
# ==================================================

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


# ==================================================
# PROCESS DATASET
# ==================================================

splits = {
    "Training": "train.csv",
    "Validation": "validation.csv",
    "Testing": "test.csv"
}

OUTPUT_DIR.mkdir(exist_ok=True)

supported_extensions = {
    ".jpg", ".jpeg", ".png", ".bmp"
}


try:

    for split_name, output_name in splits.items():

        split_path = DATASET_DIR / split_name

        if not split_path.exists():

            print(f"Folder not found: {split_path}")
            continue

        output_path = OUTPUT_DIR / output_name

        processed = 0
        failed = 0

        print("\n================================")
        print(f"Processing {split_name}")
        print("================================")

        with open(output_path, "w", newline="") as file:

            writer = csv.writer(file)
            writer.writerow(headers)

            class_folders = sorted(
                folder for folder in split_path.iterdir()
                if folder.is_dir()
                and folder.name.upper() in "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
                and len(folder.name) == 1
            )

            for class_folder in class_folders:

                label = class_folder.name.upper()

                image_files = [
                    p for p in class_folder.iterdir()
                    if p.suffix.lower() in supported_extensions
                ]

                print(
                    f"\nLetter {label}: "
                    f"{len(image_files)} images"
                )

                for image_path in image_files:

                    image = cv2.imread(str(image_path))

                    if image is None:
                        failed += 1
                        continue

                    rgb_image = cv2.cvtColor(
                        image,
                        cv2.COLOR_BGR2RGB
                    )

                    mp_image = mp.Image(
                        image_format=mp.ImageFormat.SRGB,
                        data=rgb_image
                    )

                    result = landmarker.detect(mp_image)

                    if not result.hand_landmarks:

                        failed += 1
                        continue

                    hands = result.hand_landmarks

                    # First hand
                    hand1 = normalize_landmarks(hands[0])

                    # Second hand, if detected
                    if len(hands) >= 2:

                        hand2 = normalize_landmarks(hands[1])

                    else:

                        hand2 = [0.0] * 63

                    row = hand1 + hand2 + [label]

                    writer.writerow(row)

                    processed += 1

                print(
                    f"Total processed: {processed}"
                )

        print(f"\nSaved: {output_path}")
        print(f"Successful images: {processed}")
        print(f"Skipped images: {failed}")


finally:

    landmarker.close()


print("\n================================")
print("DATASET PROCESSING COMPLETED")
print("================================")
