import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import img_to_array
import os

# Load trained model
model_path = r"E:\INTERN\FINAL MODEL\sign_language_model17.h5"
model = tf.keras.models.load_model(model_path)

# Load labels
LABELS = sorted(os.listdir("E:\INTERN\DATASET\processed_combine_asl_dataset"))

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.7)

# Define colors
finger_colors = [
    (200, 50, 50),   # Pinky
    (0, 200, 0),     # Ring
    (0, 200, 200),   # Middle
    (150, 80, 150),  # Index
    (216, 234, 247)  # Thumb
]
palm_color = (150, 150, 150)
palm_dot_color = (0, 0, 255)

# Finger joint indices
finger_joints = [
    [17, 18, 19, 20], # Pinky
    [13, 14, 15, 16], # Ring
    [9, 10, 11, 12],  # Middle
    [5, 6, 7, 8],     # Index
    [1, 2, 3, 4]      # Thumb
]
palm_connections = [(0, 1), (0, 5), (5, 9), (9, 13), (13, 17), (17, 0)]
palm_points = [0, 1, 5, 9, 13, 17]

cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("[ERROR] Failed to capture frame.")
        break

    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)
    black_bg = np.zeros_like(frame)

    if results.multi_hand_landmarks:
        print("[INFO] Hand detected.")
        for hand_landmarks in results.multi_hand_landmarks:
            landmark_points = []
            h, w, _ = frame.shape

            for landmark in hand_landmarks.landmark:
                cx, cy = int(landmark.x * w), int(landmark.y * h)
                landmark_points.append((cx, cy))

            # Draw finger lines and points on black background
            for finger_idx, joint_indices in enumerate(finger_joints):
                color = finger_colors[finger_idx]
                for j in range(len(joint_indices) - 1):
                    start = landmark_points[joint_indices[j]]
                    end = landmark_points[joint_indices[j + 1]]
                    cv2.line(black_bg, start, end, color, 2)
                    cv2.circle(black_bg, start, 4, color, -1)
                    cv2.circle(black_bg, start, 5, (255, 255, 255), 1)
                tip = landmark_points[joint_indices[-1]]
                cv2.circle(black_bg, tip, 4, color, -1)
                cv2.circle(black_bg, tip, 5, (255, 255, 255), 1)

            # Palm on black background
            for pt in palm_points:
                cv2.circle(black_bg, landmark_points[pt], 4, palm_dot_color, -1)
                cv2.circle(black_bg, landmark_points[pt], 5, (255, 255, 255), 1)
            for a, b in palm_connections:
                cv2.line(black_bg, landmark_points[a], landmark_points[b], palm_color, 2)

            # ➤ Draw same on actual camera feed
            for finger_idx, joint_indices in enumerate(finger_joints):
                color = finger_colors[finger_idx]
                for j in range(len(joint_indices) - 1):
                    start = landmark_points[joint_indices[j]]
                    end = landmark_points[joint_indices[j + 1]]
                    cv2.line(frame, start, end, color, 2)
                for idx in joint_indices:
                    cv2.circle(frame, landmark_points[idx], 3, color, -1)
            for a, b in palm_connections:
                cv2.line(frame, landmark_points[a], landmark_points[b], palm_color, 2)
            for pt in palm_points:
                cv2.circle(frame, landmark_points[pt], 3, palm_dot_color, -1)

            # Prediction
            roi = cv2.resize(black_bg, (64, 64))
            roi = img_to_array(roi) / 255.0
            roi = np.expand_dims(roi, axis=0)

            prediction = model.predict(roi)[0]
            predicted_class = np.argmax(prediction)
            confidence = np.max(prediction)

            if confidence > 0.80:
                predicted_label = LABELS[predicted_class].upper()
            else:
                predicted_label = "..."

            print(f"[PREDICTION] Label: {predicted_label}")
            print(f"[PREDICTION] Probabilities: {prediction}")

            # Display prediction
            font = cv2.FONT_HERSHEY_SIMPLEX
            text = f"Predicted: {predicted_label}"
            cv2.putText(black_bg, text, (10, 50), font, 1.0, (0, 255, 0), 2, cv2.LINE_AA)
            cv2.putText(frame, text, (10, 40), font, 0.8, (0, 255, 0), 2, cv2.LINE_AA)

    else:
        print("[INFO] No hand detected.")

    cv2.imshow("Processed for Prediction (black background)", black_bg)
    cv2.imshow("Actual Camera Feed with Hand Lines", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("[INFO] Quitting.")
        break

cap.release()
cv2.destroyAllWindows()
