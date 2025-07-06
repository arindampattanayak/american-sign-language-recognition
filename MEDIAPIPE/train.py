import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os


DATASET_DIR = "E:/INTERN/DATASET/processed_combine_asl_dataset"


IMG_SIZE = (64, 64)
BATCH_SIZE = 32


datagen = ImageDataGenerator(
    rescale=1.0/255,
    validation_split=0.2,
    rotation_range=15, 
    width_shift_range=0.1,  
    height_shift_range=0.1,  
    horizontal_flip=True 
)

train_generator = datagen.flow_from_directory(
    DATASET_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='training'
)

val_generator = datagen.flow_from_directory(
    DATASET_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='validation'
)


model = keras.Sequential([
    keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3)),
    keras.layers.MaxPooling2D(2, 2),
    
    keras.layers.Conv2D(64, (3, 3), activation='relu'),
    keras.layers.MaxPooling2D(2, 2),
    
    keras.layers.Conv2D(128, (3, 3), activation='relu'),  
    keras.layers.MaxPooling2D(2, 2),
    
    keras.layers.Flatten(),
    keras.layers.Dense(256, activation='relu'),  
    keras.layers.Dropout(0.5), 
    keras.layers.Dense(len(train_generator.class_indices), activation='softmax')
])


optimizer = keras.optimizers.Adam(learning_rate=0.0005)  
model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])


history = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=15,  
    verbose=1
)


MODEL_DIR = "E:/INTERN/FINAL MODEL"
os.makedirs(MODEL_DIR, exist_ok=True)
model_path = os.path.join(MODEL_DIR, "sign_language_model17.h5")
model.save(model_path)

print(f"Model saved successfully at: {model_path}")
