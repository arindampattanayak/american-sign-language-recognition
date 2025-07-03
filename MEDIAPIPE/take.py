import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
import os
import random
from tensorflow.keras.preprocessing.image import img_to_array, load_img

# Load trained model
model_path = r"E:\INTERN\FINAL MODEL\sign_language_model17.h5"
model = tf.keras.models.load_model(model_path)

# Load labels
DATASET_PATH = r"E:\INTERN\DATASET\processed_combine_asl_dataset"
LABELS = sorted(os.listdir(DATASET_PATH))

# Function to randomly select an image from dataset
def get_random_image():
    class_name = random.choice(LABELS)  # Select a random class
    class_path = os.path.join(DATASET_PATH, class_name)
    image_name = random.choice(os.listdir(class_path))  # Select a random image
    image_path = os.path.join(class_path, image_name)
    return image_path, class_name

while True:
    # Get a random image and its label
    image_path, actual_label = get_random_image()
    
    # Load and preprocess the image
    img = load_img(image_path, target_size=(64, 64))
    img_array = img_to_array(img) / 255.0  # Normalize
    img_array = np.expand_dims(img_array, axis=0)
    
    # Predict using the model
    prediction = model.predict(img_array)[0]
    predicted_class = np.argmax(prediction)
    predicted_label = LABELS[predicted_class]
    
    # Get sorted predictions
    sorted_indices = np.argsort(prediction)[::-1]  # Sort in descending order
    prob_texts = [f"{LABELS[i]}: {prediction[i] * 100:.2f}%" for i in sorted_indices]
    
    # Compute accuracy (if the top prediction matches the actual label)
    accuracy = prediction[predicted_class] * 100
    is_correct = predicted_label == actual_label
    
    # Load image for display
    display_img = cv2.imread(image_path)
    display_img = cv2.resize(display_img, (300, 300))
    
    # Display actual and predicted label
    cv2.putText(display_img, f"Actual: {actual_label}", (10, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
    cv2.putText(display_img, f"Predicted: {predicted_label}", (10, 80),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0) if is_correct else (0, 0, 255), 2)
    cv2.putText(display_img, f"Accuracy: {accuracy:.2f}%", (10, 120),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    
    # Display top 3 predictions
    for idx, prob in enumerate(prob_texts[:3]):
        cv2.putText(display_img, prob, (10, 160 + idx * 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    cv2.imshow("Sign Language Recognition", display_img)
    
    # Press 'q' to exit
    if cv2.waitKey(2000) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
