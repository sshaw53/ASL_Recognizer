# ASL Recognizer

A real-time American Sign Language (ASL) fingerspelling recognizer that uses webcam hand-tracking and a machine learning classifier to identify letters from hand shape.

## How it works

1. **Hand tracking** — [MediaPipe HandLandmarker](https://developers.google.com/mediapipe) detects 21 landmark points on a single hand from the live webcam feed.
2. **Feature extraction** — the (x, y) coordinates of all 21 landmarks are used as input features.
3. **Classification** — a `KNeighborsClassifier` (scikit-learn), trained on a labeled dataset of landmark positions per letter, predicts the most likely ASL letter in real time.
4. **Display** — OpenCV renders the live video feed with the hand skeleton overlay, the predicted letter, and the word being built.

## Features

- Real-time letter prediction from webcam video
- Live hand-landmark visualization
- Built-in data collection mode for capturing new labeled training examples
- Word-building mode (append predicted letters into a word)

## Controls

| Key | Action |
|-----|--------|
| `p` | Capture the current hand landmarks as a training example for the current letter |
| `n` | Advance the data-collection label to the next letter |
| `a` | Append the currently predicted letter to the word |
| `x` | Clear the current word |
| `q` | Quit |

## Tech stack

Python · MediaPipe · OpenCV · pandas · scikit-learn

## Setup

```bash
pip install mediapipe opencv-python pandas scikit-learn
python asl_recognizer.py
```

Requires a webcam and the MediaPipe hand landmarker model file in `data/`.

## Future improvements

- Expand from single-letter recognition to full word/phrase recognition
- Swap the KNN classifier for a neural network trained on a larger landmark dataset
- Add support for two-handed signs
