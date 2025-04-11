import tensorflow as tf
from tensorflow.keras.layers import Layer, Conv2D, TimeDistributed, Dense, GlobalAveragePooling2D, Multiply, Lambda
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import LearningRateScheduler
import numpy as np
import os
import pandas as pd
from sklearn.model_selection import train_test_split
import glob
import warnings

# Suppress TensorFlow warnings (optional)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Suppress PyDataset warning
warnings.filterwarnings('ignore', category=UserWarning, message='.*PyDataset.*super.*')

# Simplified preprocessing function to ensure float32 format
def preprocess_image(image):
    """
    Ensure the image is in float32 format with values in [0, 1].
    This function is applied after ImageDataGenerator's rescale.
    """
    # Image should already be in float32 and in [0, 1] due to rescale=1./255
    # Just ensure the dtype is float32 (in case it's not)
    image = image.astype(np.float32)
    return image

# Custom Spatial Attention Layer
class SpatialAttentionModule(Layer):
    def __init__(self, **kwargs):
        super(SpatialAttentionModule, self).__init__(**kwargs)

    def build(self, input_shape):
        # Initialize the Conv2D and Multiply layers based on the input shape
        self.conv = Conv2D(1, (1, 1), padding='same', activation='sigmoid')
        self.multiply = Multiply()
        # Call the build method on the sub-layers
        self.conv.build(input_shape)
        self.multiply.build([input_shape, input_shape])  # Multiply takes two inputs of the same shape
        self.built = True

    def call(self, inputs):
        # Apply the Conv2D layer to the input to generate the attention map
        attention = self.conv(inputs)
        # Use the Multiply layer to apply attention to the inputs
        output = self.multiply([inputs, attention])
        return output

    def compute_output_shape(self, input_shape):
        # The output shape is the same as the input shape because the attention mechanism
        # doesn't change the spatial dimensions or the number of channels
        return input_shape

# Function to build the model for sign language recognition
def build_model(base_model, num_classes):
    """
    Build the model with a spatial attention module.
    base_model: Pre-trained base model (e.g., EfficientNetB0)
    num_classes: Number of sign language classes
    """
    # Freeze the base model
    base_model.trainable = False

    # Define the input (raw image input, e.g., 224x224x3)
    inputs = tf.keras.Input(shape=(224, 224, 3))

    # Add a time dimension (1 time step) for TimeDistributed using Lambda
    x = Lambda(lambda z: tf.expand_dims(z, axis=1))(inputs)  # Shape: (batch_size, 1, 224, 224, 3)

    # Apply the base model to each time step
    x = TimeDistributed(base_model, name='base_model')(x)  # Shape: (batch_size, 1, 7, 7, 1280)

    # Apply the spatial attention module
    x = TimeDistributed(SpatialAttentionModule(), name='spatial_attention')(x)  # Shape: (batch_size, 1, 7, 7, 1280)

    # Remove the time dimension using Lambda
    x = Lambda(lambda z: tf.squeeze(z, axis=1))(x)  # Shape: (batch_size, 7, 7, 1280)

    # Global average pooling to reduce spatial dimensions
    x = GlobalAveragePooling2D()(x)  # Shape: (batch_size, 1280)

    # Add a dense layer for classification
    x = Dense(128, activation='relu')(x)
    outputs = Dense(num_classes, activation='softmax')(x)

    # Build the model
    model = Model(inputs, outputs)
    return model

# Learning rate scheduler
def lr_scheduler(epoch):
    initial_lr = 0.00025
    return initial_lr  # You can modify this to decay the learning rate if needed

# Train the model
def train_model(model, base_model, train_generator, valid_generator):
    """
    Train the model in phases.
    model: The full model with the spatial attention module
    base_model: The pre-trained base model (e.g., EfficientNetB0)
    train_generator: Training data generator
    valid_generator: Validation data generator
    """
    # Phase 1: Train the top layers
    print("===== Phase 1: Training top layers =====")
    
    # Ensure base model is frozen
    base_model.trainable = False

    # Compile the model
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.00025),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    # Define callbacks
    callbacks = [
        LearningRateScheduler(lr_scheduler)
    ]

    # Train the model
    history_phase1 = model.fit(
        train_generator,
        validation_data=valid_generator,
        epochs=20,
        callbacks=callbacks,
        verbose=1
    )

    return model, history_phase1

# Load and preprocess the dataset
def load_data(data_dir, target_size=(224, 224), batch_size=32, validation_split=0.2):
    """
    Load and preprocess the dataset from a single directory containing asl, dgs, and lsl folders.
    data_dir: Path to the dataset directory (contains asl, dgs, lsl folders)
    target_size: Target image size (e.g., (224, 224))
    batch_size: Batch size for training
    validation_split: Fraction of data to use for validation (e.g., 0.2 for 20%)
    """
    # List of sign languages
    sign_languages = ['asl', 'dgs', 'lsl']

    # Collect all image paths and labels
    image_paths = []
    labels = []

    # Iterate over each sign language folder
    for sign_lang in sign_languages:
        sign_lang_dir = os.path.join(data_dir, sign_lang)
        if not os.path.exists(sign_lang_dir):
            print(f"Warning: Directory {sign_lang_dir} not found. Skipping.")
            continue

        # Iterate over each alphabet folder (A, B, ..., Z)
        for label in os.listdir(sign_lang_dir):
            label_dir = os.path.join(sign_lang_dir, label)
            if not os.path.isdir(label_dir):
                continue

            # Collect all image paths in this alphabet folder
            for img_file in glob.glob(os.path.join(label_dir, '*.[jp][pn][gf]')):  # Match .jpg, .png, .jpeg
                image_paths.append(img_file)
                labels.append(label)

    # Create a DataFrame with image paths and labels
    df = pd.DataFrame({'filename': image_paths, 'class': labels})

    # Split the data into training and validation sets
    train_df, valid_df = train_test_split(df, test_size=validation_split, stratify=df['class'], random_state=42)

    # Print the number of images in each set
    print(f"Total images: {len(df)}")
    print(f"Training images: {len(train_df)}")
    print(f"Validation images: {len(valid_df)}")
    print(f"Number of classes: {len(df['class'].unique())}")

    # Data augmentation for training
    train_datagen = ImageDataGenerator(
        rescale=1./255,  # Normalize pixel values to [0, 1]
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest',
        preprocessing_function=preprocess_image
    )

    # Only rescaling for validation
    valid_datagen = ImageDataGenerator(
        rescale=1./255,
        preprocessing_function=preprocess_image
    )

    # Create training data generator
    train_generator = train_datagen.flow_from_dataframe(
        dataframe=train_df,
        x_col='filename',
        y_col='class',
        target_size=target_size,
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=True
    )

    # Create validation data generator
    valid_generator = valid_datagen.flow_from_dataframe(
        dataframe=valid_df,
        x_col='filename',
        y_col='class',
        target_size=target_size,
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=False
    )

    return train_generator, valid_generator

# Main function
def main():
    # Path to your dataset (adjust this path to match your directory structure)
    data_dir = "C:/Users/nived/OneDrive/Desktop/Spring 2025/Foundations of AI/Project/AI-SignLanguage/data"

    # Ensure directory exists
    if not os.path.exists(data_dir):
        raise FileNotFoundError(f"Dataset directory not found: {data_dir}. Please check the path.")

    # Load the data
    target_size = (224, 224)  # Input size for EfficientNetB0
    batch_size = 32
    validation_split = 0.2  # 20% of the data for validation
    train_generator, valid_generator = load_data(data_dir, target_size, batch_size, validation_split)

    # Number of classes (based on the unique labels in the dataset)
    num_classes = len(train_generator.class_indices)
    print(f"Number of classes: {num_classes}")

    # Load the base model (EfficientNetB0)
    base_model = tf.keras.applications.EfficientNetB0(
        include_top=False,
        weights='imagenet',
        input_shape=(224, 224, 3)
    )

    # Build the model
    model = build_model(base_model, num_classes)

    # Print model summary
    model.summary()

    # Train the model
    model, history = train_model(model, base_model, train_generator, valid_generator)

    # Save the model
    model.save('C:/Users/nived/OneDrive/Desktop/Spring 2025/Foundations of AI/Project/AI-SignLanguage/models/sign_language_model.h5')

    return model, history

if __name__ == "__main__":
    try:
        model, history = main()
        print("Training completed successfully!")
    except Exception as e:
        print(f"An error occurred: {str(e)}")