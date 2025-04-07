import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input  # Adjust if using another model

# Load the model
model = tf.keras.models.load_model('/home/daniah/Desktop/ExternalDrive/meat_data/Meat_Model.keras')

# Load and preprocess the image
img_path = '/home/daniah/Desktop/ExternalDrive/meat_data/test_images/download.jpeg'  # Your image path
img = image.load_img(img_path, target_size=(150, 150))  # Resize image to (224, 224)
img_array = image.img_to_array(img)  # Convert image to numpy array
img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
img_array = preprocess_input(img_array)  # Normalize image if needed (based on your model)

# Make a prediction
prediction = model.predict(img_array)

# Print raw prediction value
print(f"Raw prediction: {prediction}")

# Threshold to classify the result
if prediction[0] > 0.2:
    print("The item is fresh.")
else:
    print("The item is spoiled.")

import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

# Load the model
model_path = '/home/daniah/Desktop/ExternalDrive/meat_data/Meat_Model.keras'
model = tf.keras.models.load_model(model_path)

# Folder with test images
folder_path = '/home/daniah/Desktop/ExternalDrive/meat_data/test_images'

# Expected labels for each image manually
expected_labels = {
    'f1.jpeg': 'fresh',
    'f2.jpeg': 'fresh',
    's1.jpeg': 'spoiled',
    's2.jpeg': 'spoiled',
    's3.jpeg': 'spoiled'
}

# Threshold
threshold = 0.7703

# Loop through all test images
for img_name in expected_labels:
    img_path = os.path.join(folder_path, img_name)

    # Load and preprocess image
    img = image.load_img(img_path, target_size=(150, 150))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)

    # Predict
    prediction = model.predict(img_array)[0][0]

    # Decide class
    predicted_class = 'fresh' if prediction > threshold else 'spoiled'
    actual_class = expected_labels[img_name]

    # Print result
    print(f"\n🖼️ Image: {img_name}")
    print(f"🔮 Prediction value: {prediction:.4f}")
    print(f"✅ Model says: {predicted_class}")
    print(f"🎯 Ground truth: {actual_class}")
    print("✔️ Correct!" if predicted_class == actual_class else "❌ Wrong!")
