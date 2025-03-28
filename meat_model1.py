import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt

# Paths to your dataset (update these!)
train_dir = "/home/daniah/Desktop/ExternalDrive/meat_data/train"
val_dir = "/home/daniah/Desktop/ExternalDrive/meat_data/validation"

# Load datasets
train_ds = tf.keras.utils.image_dataset_from_directory(
    train_dir,
    image_size=(150, 150),
    batch_size=32,
    label_mode='binary'  # Binary labels (0=spoiled, 1=fresh)
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    val_dir,
    image_size=(150, 150),
    batch_size=32,
    label_mode='binary'
)

# Build the model
model = models.Sequential([
    layers.Rescaling(1./255),  # Normalize pixel values
    layers.Conv2D(32, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(1, activation='sigmoid')  # Binary output
])

# Compile the model
model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# Train the model
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=10
)

# Save the model and plot results
model.save('ExternalDrive/meat_data/Meat_Model.h5')  # Save to external drive

plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.savefig('ExternalDrive/meat_data/training_results.png')
print("Training complete! Model saved to 'Meat_Model.h5'.")
model.save('Meat_Model.h5')  # Should be in the same directory as predict.py
