
# AI-Based Indian Sign Language (ISL) Recognition

An AI-based project that recognizes Indian Sign Language (ISL) alphabet signs using computer vision and machine learning.

## About the Project

This project uses MediaPipe to detect hand landmarks and a Random Forest classifier to recognize ISL alphabet signs.

The goal is to make communication more accessible by exploring how computer vision and machine learning can help recognize sign language.

## Features

- Real-time hand detection using a webcam.
- Hand landmark detection using MediaPipe.
- Recognition of ISL alphabet signs.
- Machine learning classification using Random Forest.
- Model evaluation using a separate test dataset.

## Technologies Used

- Python
- OpenCV
- MediaPipe
- NumPy
- Pandas
- Scikit-learn
- Joblib

## Project Structure

```text
AI-Based-ISL-Recognition/
│
├── models/
│   ├── hand_landmarker.task
│   └── isl_random_forest.joblib
│
├── collect_data.py
├── hand_detection.py
├── normalize_test.py
├── preprocess_dataset.py
├── train_model.py
├── evaluate_model.py
├── live_recognition.py
│
├── requirements.txt
├── .gitignore
└── README.md
```

## Model Performance

The trained Random Forest model achieved approximately **92% accuracy** on the held-out test dataset.

Note: This is dataset test accuracy. Real-world webcam performance may differ.

## Installation

1. Clone this repository:

```bash
git clone https://github.com/nirajan773/AI_Based_ISL_Recognition.git
```

2. Open the project folder:

```bash
cd AI-Based-ISL-Recognition
```

3. Install the required libraries:

```bash
pip install -r requirements.txt
```

## Running the Project

Run the webcam recognition script:

```bash
python live_recognition.py
```

Make sure the required model files are present inside the `models` folder.

## Dataset

**Original Dataset:** RealSign – Indian Sign Language Dataset

**Dataset Repository:** https://github.com/RealSign62/RealSign-Indian-Sign-Language-Dataset

**Dataset Creators:** Aaptha Boggaram, Ankith Boggaram, Ashwin Srinivasa Ramanujan, and Aryan Rajesh Sharma.

## Future Improvements

- Improve real-time recognition accuracy.
- Improve recognition of two-hand signs.
- Add more sign language vocabulary.
- Explore dynamic gesture recognition.
- Develop a user-friendly interface.

## Acknowledgements

Thanks to the creators of the RealSign Indian Sign Language dataset and the developers of MediaPipe and Scikit-learn.

## Author

Created as a personal learning project in Artificial Intelligence, Computer Vision, and Machine Learning.
