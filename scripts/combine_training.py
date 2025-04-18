# Import necessary libraries
import os
import cv2
import numpy as np
import tensorflow as tf
import mediapipe as mp
from imgaug import augmenters as iaa
import random
from PIL import Image
from tensorflow.keras.layers import Input
import matplotlib.pyplot as plt

# Enable mixed precision (new API)
tf.keras.mixed_precision.set_global_policy('mixed_float16')

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, max_num_hands=1, min_detection_confidence=0.5)

# Set up paths
data_dirs = ['../data/asl', '../data/dgs', '../data/lse']
preprocessed_dir = "../data/preprocessed"
os.makedirs(preprocessed_dir, exist_ok=True)

IMG_SIZE = 128

# Maps which letters are shared across which languages
letter_relationships = {
    "A": ["asl-dgs-lse"],
    "B": ["asl-dgs", "lse"],
    "C": ["asl-dgs-lse"],
    "D": ["asl", "dgs-lse"],
    "E": ["asl", "dgs", "lse"],
    "F": ["asl", "dgs", "lse"],
    "G": ["asl", "dgs-lse"],
    "H": ["asl", "dgs"],
    "I": ["asl-dgs-lse"],
    "J": ["asl", "dgs"],
    "K": ["asl", "dgs", "lse"],
    "L": ["asl-dgs-lse"],
    "M": ["asl-lse", "dgs"],
    "N": ["asl-lse", "dgs"],
    "O": ["asl-dgs-lse"],
    "P": ["asl", "dgs", "lse"],
    "Q": ["asl-dgs", "lse"],
    "R": ["asl-dgs-lse"],
    "S": ["asl-dgs", "lse"],
    "Sch": ["dgs"],
    "T": ["asl", "dgs", "lse"],
    "U": ["asl-dgs"],
    "V": ["asl-dgs"],
    "W": ["asl-dgs"],
    "X": ["asl-dgs"],
    "Y": ["asl-dgs"],
    "Z": ["asl", "dgs"],
}

# Helper to read and preprocess images
def get_preprocessed_images(folder_path, sample_count, padding):
    all_files = [f for f in os.listdir(folder_path) if f != ".DS_Store"]
    processed_images = []
    for file in all_files:
        file_path = os.path.join(folder_path, file)
        img = cv2.imread(file_path)
        if img is None:
            continue
        processed_img = preprocess_image(img, padding=padding)
        if processed_img is not None:
            processed_images.append((file, processed_img))
            flipped_img = cv2.flip(processed_img, 1)
            processed_images.append((f"flipped_{file}", flipped_img))
    return processed_images

# Main function to preprocess and save images
def preprocess_and_save_images():
    lang_dirs = {os.path.basename(d): d for d in data_dirs}
    for letter, lang_groups in letter_relationships.items():
        for group in lang_groups:
            langs = group.split("-")
            output_class_dir = os.path.join(preprocessed_dir, f"{letter}_{group}")
            os.makedirs(output_class_dir, exist_ok=True)
            for lang in langs:
                lang_dir = lang_dirs[lang]
                class_path = os.path.join(lang_dir, letter)
                padding = 60 if lang == "lse" else 30
                images = get_preprocessed_images(class_path, None, padding)
                for file_name, img in images:
                    output_path = os.path.join(output_class_dir, f"{lang}_{file_name}")
                    cv2.imwrite(output_path, img)

def balance_class_distribution():
    # Placeholder if additional balancing logic is needed later
    pass

# Function to crop and resize images
def preprocess_image(image, padding=30):
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_image)
    if results.multi_hand_landmarks:
        h, w, _ = image.shape
        x_min, x_max, y_min, y_max = w, 0, h, 0
        for hand_landmarks in results.multi_hand_landmarks:
            for lm in hand_landmarks.landmark:
                x, y = int(lm.x * w), int(lm.y * h)
                x_min, x_max = min(x_min, x), max(x_max, x)
                y_min, y_max = min(y_min, y), max(y_max, y)
        x_min, x_max = max(0, x_min - padding), min(w, x_max + padding)
        y_min, y_max = max(0, y_min - padding), min(h, y_max + padding)
        crop_w, crop_h = x_max - x_min, y_max - y_min
        if crop_w > crop_h:
            cy = (y_min + y_max) // 2
            half = crop_w // 2
            y_min, y_max = max(0, cy - half), min(h, cy + half)
        else:
            cx = (x_min + x_max) // 2
            half = crop_h // 2
            x_min, x_max = max(0, cx - half), min(w, cx + half)
        cropped = image[y_min:y_max, x_min:x_max]
        ch, cw, _ = cropped.shape
        scale = IMG_SIZE / max(ch, cw)
        nw, nh = int(cw * scale), int(ch * scale)
        resized = cv2.resize(cropped, (nw, nh), interpolation=cv2.INTER_LINEAR)
        if nw > nh:
            start = (nw - nh) // 2
            square = resized[:, start:start + nh]
        else:
            start = (nh - nw) // 2
            square = resized[start:start + nw, :]
        return cv2.resize(square, (IMG_SIZE, IMG_SIZE))
    return None

# Prepare dataset
def prepare_combined_dataset():
    combined_data = []
    class_weights = {}
    class_counts = {}
    max_samples = 0

    for root, dirs, _ in os.walk(preprocessed_dir):
        for cls in dirs:
            path = os.path.join(root, cls)
            count = len([f for f in os.listdir(path) if f != ".DS_Store"])
            if count > 0:
                combined_data.append((path, cls))
                class_counts[cls] = count
                max_samples = max(max_samples, count)

    combined_data.sort(key=lambda x: x[1])
    for idx, (_, cls) in enumerate(combined_data):
        class_weights[idx] = min(2.0, max_samples / class_counts[cls])

    print("\nClass Distribution:")
    for _, cls in combined_data:
        print(f"{cls}: {class_counts[cls]} images")
    print("\nClass Weights:", class_weights)

    return combined_data, class_weights

# Custom Data Generator
class CombinedDataGenerator(tf.keras.utils.Sequence):
    def __init__(self, data, batch_size, is_training=True):
        self.data = data
        self.batch_size = batch_size
        self.is_training = is_training
        self.indices = []
        max_imgs = max(len(os.listdir(p)) for p, _ in data)
        for p, _ in data:
            files = [f for f in os.listdir(p) if f != '.DS_Store']
            ratio = min(2, max_imgs / len(files))
            oversampled = random.choices(files, k=int(len(files) * ratio))
            self.indices += [(p, f) for f in oversampled]
        if self.is_training:
            random.shuffle(self.indices)

    def __len__(self):
        return min(350, len(self.indices) // self.batch_size)

    def __getitem__(self, idx):
        batch = self.indices[idx * self.batch_size:(idx + 1) * self.batch_size]
        X, y = [], np.zeros((len(batch), len(self.data)))
        seq = iaa.Sequential([iaa.Multiply((0.8,1.2)), iaa.Add((-10,10))])
        for i, (p, fn) in enumerate(batch):
            img = cv2.imread(os.path.join(p, fn))
            img = seq(image=img)
            X.append(img)
            cls_idx = next(j for j,(pp,_) in enumerate(self.data) if pp==p)
            y[i, cls_idx] = 1
        return np.array(X)/255.0, y

    def on_epoch_end(self):
        if self.is_training:
            random.shuffle(self.indices)

# Build & compile model (with MobileNetV2 backbone)
def build_model(num_classes):
    base = tf.keras.applications.MobileNetV2(
        input_shape=(IMG_SIZE,IMG_SIZE,3),
        include_top=False,
        weights='imagenet'
    )
    x = tf.keras.layers.GlobalAveragePooling2D()(base.output)
    x = tf.keras.layers.Dropout(0.5)(x)
    outputs = tf.keras.layers.Dense(
        num_classes, activation='softmax', dtype='float32', name='classifier'
    )(x)
    return tf.keras.Model(inputs=base.input, outputs=outputs)

if __name__ == "__main__":
    # Preprocess and prepare data
    preprocess_and_save_images()
    balance_class_distribution()
    combined_data, class_weights = prepare_combined_dataset()
    num_classes = len(combined_data)

    # Generators
    batch_size = 64
    train_gen = CombinedDataGenerator(combined_data, batch_size, is_training=True)
    val_gen   = CombinedDataGenerator(combined_data, batch_size, is_training=False)

    # Model setup
    model = build_model(num_classes)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss', patience=3, restore_best_weights=True
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss', factor=0.2, patience=2
        ),
        tf.keras.callbacks.TensorBoard(log_dir='logs')
    ]

    # Training
    history = model.fit(
        train_gen,
        epochs=30,
        validation_data=val_gen,
        class_weight=class_weights,
        callbacks=callbacks
    )

    # Save model
    model.save('../models/combined_classes.h5')

    # Final accuracy reporting
    val_loss, val_acc = model.evaluate(val_gen)
    print(f"\nFinal validation accuracy: {val_acc * 100:.2f}%")

    # Plot accuracy curves
    plt.figure()
    plt.plot(history.history['accuracy'], label='train-acc')
    plt.plot(history.history['val_accuracy'], label='val-acc')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.title('Training & Validation Accuracy')
    plt.show()