# 🤟 Real-Time Sign Language Recognition System

This project uses **deep learning** and **computer vision** to recognize American Sign Language (ASL) alphabets in real time through a webcam. It combines a **CNN model trained on ASL images** with **MediaPipe hand tracking** and **OpenCV visualization** to create an interactive and accurate gesture recognition system.

---

## 🚀 Features

- Real-time hand gesture detection using webcam
- ASL alphabet classification using trained CNN model
- Top-3 predictions with confidence scores on screen
- Hand landmark visualization using MediaPipe
- Dataset augmentation for improved model robustness

---
## 🧰 Technologies Used

### 🧠 Machine Learning / Deep Learning
- **TensorFlow** + **Keras** – Designing and training the CNN model
- **ImageDataGenerator** – For on-the-fly image augmentation during training
- **NumPy** – Efficient numerical operations and image array manipulation

### 📸 Computer Vision
- **OpenCV** – Capturing webcam feed and rendering real-time hand visualizations
- **MediaPipe** – Real-time hand landmark detection from video frames

### 📊 Evaluation & Visualization
- **scikit-learn** – Evaluation metrics like confusion matrix, classification report, ROC AUC
- **matplotlib** – Plotting confusion matrix, ROC curve, and saving sample predictions

## ⚙️ Installation & Usage Guide

Follow these steps to set up the project locally using a Python virtual environment:

---

### Step 1: Clone the Repository

```bash
git clone https://github.com/arindampattanayak/american-sign-language-recognition.git
cd american-sign-language-recognition
```
### Step 2: Create a Virtual Environment

🪟 For Windows:
   ```sh
python -m venv venv
venv\Scripts\activate
```

🐧 For macOS/Linux:
   ```sh
python3 -m venv venv
source venv/bin/activate
```
 ### Step 3: Install Dependencies
 
 Make sure you have requirements.txt in your project folder. Then run:
 ```sh
 pip install -r requirements.txt
```
 If you don’t have one yet, manually install the core packages:
```sh
 pip install tensorflow opencv-python mediapipe scikit-learn matplotlib numpy
```





