# Import necessary libraries
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import EfficientNetB2
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout, BatchNormalization, Input, concatenate
from tensorflow.keras.models import Model
import cv2
import matplotlib.pyplot as plt

# Set up paths - using your exact directory structure
base_data_dir = '../data'
language_dirs = {
    'asl': 'asl',
    'dgs': 'dgs',
    'lsl': 'LSL'
}

# Set hyperparameters
img_size = (240, 240)  # Increased size for better feature detection
batch_size = 32
initial_learning_rate = 5e-4
epochs = 50
validation_split = 0.2

# Enhanced preprocessing for sign language images
def enhance_sign_image(img):
    """Simplified preprocessing for sign language images"""
    if img is None or len(img.shape) < 2:
        return None
        
    # Apply preprocessing only for color images
    if len(img.shape) == 3 and img.shape[2] == 3:
        try:
            # Apply slight blur to reduce noise
            blurred = cv2.GaussianBlur(img, (3, 3), 0)
            
            # Simple contrast adjustment without CLAHE
            # Convert to uint8 if it's not already
            if blurred.dtype != np.uint8:
                # If image is float 0-1, convert to 0-255
                if blurred.max() <= 1.0:
                    blurred = (blurred * 255).astype(np.uint8)
                else:
                    blurred = blurred.astype(np.uint8)
            
            # Simple contrast enhancement using standard CV2 function
            alpha = 1.2  # Contrast control (1.0 means no change)
            beta = 10    # Brightness control (0 means no change)
            adjusted = cv2.convertScaleAbs(blurred, alpha=alpha, beta=beta)
            
            return adjusted
        except Exception as e:
            print(f"Warning: Image enhancement failed: {e}. Using original image.")
            return img
    return img

# Prepare combined dataset with language prefixes
def prepare_combined_dataset():
    """Prepare a combined dataset with language prefixes for class names"""
    # We'll combine all data but prefix class names with language identifier
    combined_data_dir = '../data/combined_sign_data'
    os.makedirs(combined_data_dir, exist_ok=True)
    
    # Track class names and counts
    class_mapping = {}
    sample_counts = {}
    
    # Process each language directory
    for lang, lang_dir in language_dirs.items():
        lang_path = os.path.join(base_data_dir, lang_dir)
        
        # Process each class directory (A, B, C, etc.)
        for class_name in os.listdir(lang_path):
            class_dir = os.path.join(lang_path, class_name)
            if not os.path.isdir(class_dir):
                continue
                
            # Create prefixed class name (e.g., asl_A, dgs_B)
            prefixed_class = f"{lang}_{class_name}"
            prefixed_dir = os.path.join(combined_data_dir, prefixed_class)
            os.makedirs(prefixed_dir, exist_ok=True)
            
            # Add to class mapping
            class_mapping[prefixed_class] = (lang, class_name)
            
            # Copy or link files (using symbolic links to save space)
            sample_count = 0
            for img_file in os.listdir(class_dir):
                if img_file.lower().endswith(('.png', '.jpg', '.jpeg')):
                    # Create symbolic link instead of copying
                    src_path = os.path.join(class_dir, img_file)
                    dst_path = os.path.join(prefixed_dir, f"{lang}_{img_file}")
                    
                    if not os.path.exists(dst_path):
                        try:
                            # Try symbolic link first (saves space)
                            os.symlink(os.path.abspath(src_path), dst_path)
                        except:
                            # Fall back to copying if symbolic links not supported
                            import shutil
                            shutil.copy2(src_path, dst_path)
                    
                    sample_count += 1
            
            sample_counts[prefixed_class] = sample_count
            print(f"Processed {prefixed_class}: {sample_count} samples")
    
    # Print dataset summary
    print(f"Combined dataset created at {combined_data_dir}")
    print(f"Total classes: {len(class_mapping)}")
    
    return combined_data_dir, class_mapping, sample_counts

# Create optimized data generators
def create_data_generators(data_dir):
    """Create training and validation generators with enhanced augmentation"""
    # Training generator with augmentation
    train_datagen = ImageDataGenerator(
        # We'll do preprocessing manually in a separate step
        rescale=1./255,
        rotation_range=15,          # Moderate rotation
        width_shift_range=0.1,      # Slight shifts
        height_shift_range=0.1,
        zoom_range=0.1,             # Slight zoom
        brightness_range=[0.9, 1.1], # Slight brightness adjustment
        horizontal_flip=False,      # Don't flip (could change sign meaning)
        fill_mode='nearest',
        validation_split=validation_split
    )
    
    # Validation generator with only preprocessing
    valid_datagen = ImageDataGenerator(
        rescale=1./255,
        validation_split=validation_split
    )
    
    # Create generators
    train_generator = train_datagen.flow_from_directory(
        data_dir,
        target_size=img_size,
        batch_size=batch_size,
        class_mode='categorical',
        subset='training',
        shuffle=True
    )
    
    valid_generator = valid_datagen.flow_from_directory(
        data_dir,
        target_size=img_size,
        batch_size=batch_size,
        class_mode='categorical',
        subset='validation',
        shuffle=False
    )
    
    return train_generator, valid_generator

# Create an optimized single model for all sign languages
def create_unified_sign_model(num_classes):
    """Create a unified model optimized for multi-language sign recognition"""
    # Use EfficientNetB2 for better performance
    base_model = EfficientNetB2(
        weights='imagenet', 
        include_top=False, 
        input_shape=(*img_size, 3)
    )
    
    # Initially freeze the base model
    for layer in base_model.layers:
        layer.trainable = False
    
    # Create model architecture
    inputs = Input(shape=(*img_size, 3))
    x = base_model(inputs)
    x = GlobalAveragePooling2D()(x)
    x = BatchNormalization()(x)
    
    # Create multiple branches for feature extraction
    # Branch 1: Deep for complex features
    branch1 = Dense(512, activation='relu')(x)
    branch1 = BatchNormalization()(branch1)
    branch1 = Dropout(0.4)(branch1)
    branch1 = Dense(256, activation='relu')(branch1)
    branch1 = BatchNormalization()(branch1)
    branch1 = Dropout(0.4)(branch1)
    
    # Branch 2: Wider for broader feature capture
    branch2 = Dense(768, activation='relu')(x)
    branch2 = BatchNormalization()(branch2)
    branch2 = Dropout(0.4)(branch2)
    
    # Merge branches
    merged = concatenate([branch1, branch2])
    merged = Dense(384, activation='relu')(merged)
    merged = BatchNormalization()(merged)
    merged = Dropout(0.3)(merged)
    
    # Final classification layer
    outputs = Dense(num_classes, activation='softmax')(merged)
    
    # Create model
    model = Model(inputs=inputs, outputs=outputs)
    
    # Compile model
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=initial_learning_rate),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model, base_model

# Train with a two-phase approach
def train_model(model, base_model, train_generator, valid_generator):
    """Train the model with a two-phase approach for better accuracy"""
    # Phase 1: Train only the top layers
    print("\n===== Phase 1: Training top layers =====")
    
    # Set up callbacks
    callbacks_phase1 = [
        EarlyStopping(
            monitor='val_accuracy',
            patience=7,
            restore_best_weights=True
        ),
        ReduceLROnPlateau(
            monitor='val_accuracy',
            factor=0.5,
            patience=3,
            min_lr=1e-6
        ),
        ModelCheckpoint(
            '../models/unified_sign_model_phase1.h5',
            save_best_only=True,
            monitor='val_accuracy'
        )
    ]
    
    # Train top layers
    history_phase1 = model.fit(
        train_generator,
        validation_data=valid_generator,
        epochs=15,
        callbacks=callbacks_phase1
    )
    
    # Phase 2: Fine-tune upper layers of the base model
    print("\n===== Phase 2: Fine-tuning top layers of base model =====")
    
    # Unfreeze top layers of the base model
    for layer in base_model.layers[-30:]:
        layer.trainable = True
    
    # Recompile with lower learning rate
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # Set up callbacks for phase 2
    callbacks_phase2 = [
        EarlyStopping(
            monitor='val_accuracy',
            patience=10,
            restore_best_weights=True
        ),
        ReduceLROnPlateau(
            monitor='val_accuracy',
            factor=0.2,
            patience=5,
            min_lr=1e-7
        ),
        ModelCheckpoint(
            '../models/unified_sign_model_phase2.h5',
            save_best_only=True,
            monitor='val_accuracy'
        )
    ]
    
    # Train with fine-tuning
    history_phase2 = model.fit(
        train_generator,
        validation_data=valid_generator,
        epochs=35,
        callbacks=callbacks_phase2,
        initial_epoch=len(history_phase1.history['loss'])
    )
    
    # Combine histories
    combined_history = {
        'accuracy': history_phase1.history['accuracy'] + history_phase2.history['accuracy'],
        'val_accuracy': history_phase1.history['val_accuracy'] + history_phase2.history['val_accuracy'],
        'loss': history_phase1.history['loss'] + history_phase2.history['loss'],
        'val_loss': history_phase1.history['val_loss'] + history_phase2.history['val_loss']
    }
    
    return model, combined_history

# Evaluate and save the model
def evaluate_and_save_model(model, valid_generator, class_mapping):
    """Evaluate model performance and save it"""
    # Final evaluation
    evaluation = model.evaluate(valid_generator)
    print(f"\nFinal model - Loss: {evaluation[0]:.4f}, Accuracy: {evaluation[1]:.4f}")
    
    # Save the final model
    model.save('../models/unified_sign_model.h5')
    print("Final model saved to ../models/unified_sign_model.h5")
    
    # Save the class mapping
    import json
    with open('../models/class_mapping.json', 'w') as f:
        json.dump(class_mapping, f, indent=2)
    
    # Create prediction function for easy use
    with open('../models/sign_predictor.py', 'w') as f:
        f.write('''
import os
import cv2
import numpy as np
import tensorflow as tf
import json

# Load class mapping
with open('class_mapping.json', 'r') as f:
    CLASS_MAPPING = json.load(f)

def enhance_sign_image(img):
    """Enhanced preprocessing for sign language images"""
    if img is None or len(img.shape) < 2:
        return None
        
    # Apply preprocessing only for color images
    if len(img.shape) == 3 and img.shape[2] == 3:
        # Convert to HSV for better hand segmentation
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        
        # Apply slight blur to reduce noise
        blurred = cv2.GaussianBlur(img, (3, 3), 0)
        
        # Enhance contrast using CLAHE
        lab = cv2.cvtColor(blurred, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        cl = clahe.apply(l)
        enhanced_lab = cv2.merge((cl, a, b))
        enhanced = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)
        
        return enhanced
    return img

def preprocess_image(image_path, target_size=(240, 240)):
    """Preprocess an image for prediction"""
    # Load image
    if isinstance(image_path, str):
        img = cv2.imread(image_path)
    else:
        img = image_path
    
    if img is None:
        raise ValueError("Could not load image")
    
    # Resize
    img = cv2.resize(img, target_size)
    
    # Apply enhancement
    img = enhance_sign_image(img)
    
    # Normalize
    img = img.astype(np.float32) / 255.0
    
    # Add batch dimension
    img = np.expand_dims(img, axis=0)
    
    return img

def predict_sign(model, image_path):
    """Predict sign from image"""
    # Preprocess image
    img = preprocess_image(image_path)
    
    # Make prediction
    prediction = model.predict(img)[0]
    class_idx = np.argmax(prediction)
    
    # Get class name from generator
    from tensorflow.keras.preprocessing.image import ImageDataGenerator
    datagen = ImageDataGenerator(rescale=1./255)
    generator = datagen.flow_from_directory(
        '../data/combined_sign_data',
        target_size=(240, 240),
        batch_size=1,
        class_mode='categorical'
    )
    
    # Get class name
    class_indices = generator.class_indices
    class_names = {v: k for k, v in class_indices.items()}
    predicted_class = class_names[class_idx]
    
    # Parse language and sign
    language, sign = predicted_class.split('_', 1)
    
    # Get confidence
    confidence = float(prediction[class_idx])
    
    return {
        "language": language,
        "sign": sign,
        "confidence": confidence,
        "all_predictions": {class_names[i]: float(prediction[i]) for i in np.argsort(-prediction)[:5]}
    }

# Example usage:
# model = tf.keras.models.load_model('unified_sign_model.h5')
# result = predict_sign(model, 'path_to_image.jpg')
# print(result)
        ''')
    
    return evaluation

# Plot and save training history
def plot_training_history(history):
    """Plot and save training history"""
    plt.figure(figsize=(12, 4))
    
    # Plot accuracy
    plt.subplot(1, 2, 1)
    plt.plot(history['accuracy'], label='Train')
    plt.plot(history['val_accuracy'], label='Validation')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    # Plot loss
    plt.subplot(1, 2, 2)
    plt.plot(history['loss'], label='Train')
    plt.plot(history['val_loss'], label='Validation')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('../models/training_history.png')
    print("Training history plot saved to ../models/training_history.png")

# Main function
def main():
    """Main function to train the unified sign language model"""
    print("=== Starting Unified Sign Language Model Training ===")
    
    # Prepare combined dataset
    data_dir, class_mapping, sample_counts = prepare_combined_dataset()
    
    # Create data generators
    train_generator, valid_generator = create_data_generators(data_dir)
    
    # Print class information
    num_classes = len(train_generator.class_indices)
    print(f"Number of classes: {num_classes}")
    print(f"Class mapping: {train_generator.class_indices}")
    
    # Create model
    model, base_model = create_unified_sign_model(num_classes)
    print(model.summary())
    
    # Train model
    model, history = train_model(model, base_model, train_generator, valid_generator)
    
    # Evaluate and save model
    evaluation = evaluate_and_save_model(model, valid_generator, class_mapping)
    
    # Plot training history
    plot_training_history(history)
    
    print("=== Training Complete ===")
    print(f"Final Accuracy: {evaluation[1]:.4f}")

if __name__ == "__main__":
    # Configure GPU memory growth to avoid OOM errors
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    
    # Run main function
    main()