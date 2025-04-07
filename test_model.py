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

