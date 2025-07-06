import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from sklearn.metrics import confusion_matrix, classification_report, ConfusionMatrixDisplay, roc_auc_score, roc_curve
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import random
from sklearn.preprocessing import label_binarize


model_path = r"E:\INTERN\FINAL MODEL\sign_language_model17.h5"
data_path = r"E:\INTERN\DATASET\processed_combine_asl_dataset"
result_path = r"E:\INTERN\RESULT"
os.makedirs(result_path, exist_ok=True)


model = tf.keras.models.load_model(model_path)


LABELS = sorted(os.listdir(data_path))
num_classes = len(LABELS)


X, y = [], []
print("Loading dataset...")
image_limit_per_class = 300

for label_index, label in enumerate(LABELS):
    folder = os.path.join(data_path, label)
    images = [f for f in os.listdir(folder) if f.endswith(".jpg") or f.endswith(".png")]
    selected_images = random.sample(images, min(image_limit_per_class, len(images)))

    for file in selected_images:
        img_path = os.path.join(folder, file)
        try:
            img = load_img(img_path, target_size=(64, 64))
            img_array = img_to_array(img) / 255.0
            X.append(img_array)
            y.append(label_index)
        except Exception as e:
            print(f"Error loading image {file}: {e}")

X, y = np.array(X), np.array(y)
_, X_test, _, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

print("Running predictions...")
y_pred_probs = model.predict(X_test)
y_pred = np.argmax(y_pred_probs, axis=1)


report = classification_report(y_test, y_pred, target_names=LABELS, output_dict=False)
print("\nClassification Report:")
print(report)
with open(os.path.join(result_path, "classification_report.txt"), "w") as f:
    f.write(report)


cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=LABELS)
fig, ax = plt.subplots(figsize=(12, 10))
disp.plot(ax=ax, cmap=plt.cm.Blues, xticks_rotation='vertical')
plt.title("Confusion Matrix")
plt.tight_layout()
confusion_path = os.path.join(result_path, "confusion_matrix.png")
plt.savefig(confusion_path)
plt.show()


print("Calculating ROC curves...")
y_test_bin = label_binarize(y_test, classes=list(range(num_classes)))
fpr, tpr, _ = roc_curve(y_test_bin.ravel(), y_pred_probs.ravel())
roc_auc = roc_auc_score(y_test_bin, y_pred_probs, average="macro", multi_class="ovo")
plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f"ROC curve (area = {roc_auc:.2f})")
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Receiver Operating Characteristic (ROC)")
plt.legend(loc="lower right")
roc_path = os.path.join(result_path, "roc_curve.png")
plt.savefig(roc_path)
plt.show()


print("Saving sample predictions...")
os.makedirs(os.path.join(result_path, "samples"), exist_ok=True)
for i in range(25):
    img = (X_test[i] * 255).astype(np.uint8)
    actual = LABELS[y_test[i]]
    probs = y_pred_probs[i]
    top3 = np.argsort(probs)[-3:][::-1]
    pred_str = " | ".join([f"{LABELS[idx]}: {probs[idx]*100:.2f}%" for idx in top3])
    pred_label = LABELS[top3[0]]
    title_color = ("green" if actual == pred_label else "red")

    plt.imshow(img)
    plt.title(f"Actual: {actual}\n{pred_str}", fontsize=10, color=title_color)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(os.path.join(result_path, "samples", f"sample_{i}.png"))
    plt.close()


with open(os.path.join(result_path, "summary.txt"), "w") as f:
    accuracy = np.mean(y_pred == y_test)
    error_rate = 1 - accuracy
    f.write(f"Accuracy: {accuracy*100:.2f}%\n")
    f.write(f"Error Rate: {error_rate*100:.2f}%\n")
    f.write(f"ROC AUC: {roc_auc:.4f}\n")

print("\nAll evaluation results saved to E:\\INTERN\\RESULT")
