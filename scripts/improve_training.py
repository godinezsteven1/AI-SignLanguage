# Import necessary libraries
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout, BatchNormalization
from tensorflow.keras.models import Model
 
# Set up paths
data_dir = '../data/asl'  # Update this to your dataset path
img_size = (224, 224)  # Standard input size for many pre-trained models
 
# More advanced data augmentation for improved generalization
datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)
 
train_generator = datagen.flow_from_directory(
    data_dir,
    target_size=img_size,
    batch_size=32,
    class_mode='categorical',
    subset='training'
)
 
validation_generator = datagen.flow_from_directory(
    data_dir,
    target_size=img_size,
    batch_size=32,
    class_mode='categorical',
    subset='validation'
)
 
num_classes = len(train_generator.class_indices)
print(f"Number of classes: {num_classes}")
 
# Use transfer learning with a pre-trained model (MobileNetV2)
# MobileNetV2 is efficient and performs well on mobile/edge devices
base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
 
# Freeze the base model layers to keep the pre-trained weights
for layer in base_model.layers:
    layer.trainable = False
 
# Add custom classification layers on top of the base model
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = BatchNormalization()(x)
x = Dense(256, activation='relu')(x)
x = Dropout(0.5)(x)  # Add dropout to prevent overfitting
x = BatchNormalization()(x)
x = Dense(128, activation='relu')(x)
x = Dropout(0.3)(x)
predictions = Dense(num_classes, activation='softmax')(x)
 
# Create the full model
model = Model(inputs=base_model.input, outputs=predictions)
 
# Compile the model with a lower learning rate
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)
 
# Set up callbacks for better training
callbacks = [
    EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
    ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, min_lr=1e-6),
    ModelCheckpoint('../models/asl_model_checkpoint.h5', save_best_only=True, monitor='val_accuracy')
]
 
# Train the model
history = model.fit(
    train_generator,
    validation_data=validation_generator,
    epochs=30,  # Set a higher number of epochs, early stopping will prevent overfitting
    callbacks=callbacks
)
 
# Fine-tune the model: Unfreeze some of the top layers of the base model
for layer in base_model.layers[-20:]:
    layer.trainable = True
 
# Recompile the model with a lower learning rate for fine-tuning
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),  # Lower learning rate for fine-tuning
    loss='categorical_crossentropy',
    metrics=['accuracy']
)
 
# Continue training with unfrozen layers
history_fine = model.fit(
    train_generator,
    validation_data=validation_generator,
    epochs=15,
    callbacks=callbacks
)
 
# Save the final model
model.save('../models/improved_asl_model.h5')
 
# Convert to TensorFlow Lite for mobile deployment
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()
 
# Save the TF Lite model
with open('../models/asl_model.tflite', 'wb') as f:
    f.write(tflite_model)
 
# Optional: Quantize the model for even more efficiency
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
quantized_tflite_model = converter.convert()
 
# Save the quantized model
with open('../models/asl_model_quantized.tflite', 'wb') as f:
    f.write(quantized_tflite_model)
 
print("Model training and optimization complete!")