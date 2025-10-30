# Sign Language Recognition

![Thumbnail](assets/thumbnail.jpg)

A real-time sign language recognition system built with Python, OpenCV, MediaPipe, TensorFlow, and cvzone. The project enables gesture-based communication by detecting and classifying hand signs captured through a webcam.

## Overview
This repository contains all core scripts and model files required to capture, preprocess, and classify sign language gestures. It uses MediaPipe for hand detection and TensorFlow for classification, running efficiently on Apple Silicon through TensorFlow Metal. The model recognizes common gestures such as “Hello,” “Yes,” “No,” “Please,” and “Thank you.”

## Repository Structure
```
Sign_Language/
│
├── data_collection.py          # Captures and saves hand gesture images
├── test.py                     # Runs real-time prediction and visualization
├── converted_keras/
│   ├── keras_model.h5          # Trained Keras model
│   └── labels.txt              # Class labels for inference
├── assets/
│   └── thumbnail.jpg           # Project thumbnail for README
├── README.md                   # Project documentation
└── .gitignore
```

## Installation and Setup
Clone the repository and create a Python 3.11 virtual environment:
```bash
git clone https://github.com/austin-stanley-hinson/Sign_Language.git
cd Sign_Language
python3.11 -m venv .venv
source .venv/bin/activate
```

Install dependencies:
```bash
pip install numpy==1.26.4 opencv-contrib-python==4.11.0.86 mediapipe==0.10.21 cvzone==1.6.1 tensorflow-macos==2.15.0 tensorflow-metal==1.1.0
```

## Usage
### Data Collection
Use the data collection script to record gesture samples:
```bash
python data_collection.py
```
Press **S** to save frames as you perform gestures. Images are automatically stored locally (not included in the repo for privacy and size reasons).

### Model Testing
Run real-time predictions using your trained model:
```bash
python test.py
```
The script will open a webcam window and display predicted gestures along with the confidence scores.

Ensure your model and labels are located in:
```
converted_keras/keras_model.h5
converted_keras/labels.txt
```

## Dataset
The dataset used for training was created locally using `data_collection.py`. Each gesture category (e.g., “Hello,” “Yes,” etc.) contains multiple hand image samples captured under different lighting and angles. These images are excluded from this repository to reduce size and preserve privacy. To recreate the dataset, simply run the collection script and organize samples in separate folders before training.

## Requirements
- Python 3.11  
- OpenCV 4.11.0  
- MediaPipe 0.10.21  
- cvzone 1.6.1  
- TensorFlow macOS 2.15.0  
- TensorFlow Metal 1.1.0  
- NumPy 1.26.4  

## Author
**Austin Stanley Hinson**  
Colby College  
Computer Science, Economics
