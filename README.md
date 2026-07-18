<div align="center">

# 🌟 Sports Person Classifier

**A machine learning web app that identifies sports celebrities from a photo — wavelet-based feature extraction, OpenCV face detection, and a Flask API**

[![Python](https://img.shields.io/badge/Python-3.x-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![OpenCV](https://img.shields.io/badge/OpenCV-Face%20Detection-5C3EE8?style=for-the-badge&logo=opencv&logoColor=white)](https://opencv.org/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-SVM%20Classifier-F7931E?style=for-the-badge&logo=scikitlearn&logoColor=white)](https://scikit-learn.org/)
[![Flask](https://img.shields.io/badge/Flask-REST%20API-000000?style=for-the-badge&logo=flask&logoColor=white)](https://flask.palletsprojects.com/)

</div>

---

## 📖 Overview

This project classifies images of well-known sports celebrities using classical computer vision and machine learning. Faces are detected and cropped with OpenCV's Haar Cascades, cleaned to keep only images where both eyes are clearly visible, and turned into feature vectors using a wavelet transform. A trained classifier then predicts which celebrity is in the photo — all served through a Flask API and a simple browser UI.

---

## ✨ Pipeline

| Stage | What Happens |
|---|---|
| 🔍 **Face & Eye Detection** | OpenCV Haar Cascades locate faces; images without two clearly visible eyes are discarded |
| 🌊 **Feature Extraction** | A wavelet transform captures the key structural features of each cropped face |
| 🧠 **Model Training** | Classifiers (SVM, Logistic Regression, Random Forest) are trained and tuned with `GridSearchCV` |
| 🔌 **Prediction API** | A Flask server loads the saved model and serves predictions over REST |
| 🖥️ **Web UI** | A browser-based interface for uploading a photo and viewing the predicted celebrity |

---

## 🛠️ Tech Stack

**Computer Vision** — OpenCV (Haar Cascade face & eye detection)
**Machine Learning** — Scikit-Learn (SVM · Logistic Regression · Random Forest) · PyWavelets · NumPy
**Backend** — Python · Flask
**Frontend** — HTML · CSS · JavaScript

---

## 📂 Directory Structure

```
Sports-Person-Classifier/
│
├── model/                          # Notebook for cleaning, training, and exporting the classifier
│   └── sports_celebrity_classification.ipynb
│
└── server/                         # Flask app: REST API + web UI
    ├── artifacts/                  # Saved model + class dictionary used at inference time
    ├── opencv/haarcascades/        # Haar Cascade files for face & eye detection
    ├── static/                     # Frontend JS/CSS + sample images
    ├── templates/index.html        # Upload UI
    ├── util.py                     # Preprocessing + prediction logic
    └── server.py                   # Flask entry point
```

---

## ⚙️ Setup and Installation

### Prerequisites

- Python 3.x
- pip

### 1. Clone the repository

```bash
git clone https://github.com/Shashank17singh/Sports-Person-Classifier.git
cd Sports-Person-Classifier
```

### 2. Install dependencies

```bash
pip install opencv-python numpy scikit-learn pywavelets flask joblib
```

### 3. Train (or retrain) the model

Open the notebook inside `model/` to walk through face cleaning, wavelet feature extraction, and classifier training. This step produces the saved model artifact used by the server.

### 4. Start the Flask server

```bash
cd server
python server.py
```

The API loads the trained model and starts listening for prediction requests.

### 5. Open the UI

Once the server is running, open `http://localhost:5000` in your browser and upload an image to get a prediction. The UI is served directly by Flask from `server/templates/index.html`.

---
