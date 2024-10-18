import os
import numpy as np
import tensorflow as tf
from flask import Flask, request, jsonify
import cv2
from pathlib import Path

app = Flask(__name__)

# Load the trained model
model = tf.keras.models.load_model('nail_disease_model.keras')

# Define the class names
class_names = ['healthy', 'onychomycosis', 'psoriasis']

img_height, img_width = 224, 224  # Image size for MobileNetV3

def load_and_preprocess_image(image):
    image_resized = cv2.resize(image, (img_height, img_width))
    image_array = np.expand_dims(image_resized, axis=0)
    return image_array

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400
    
    file = request.files['file']
    
    try:
        # Read and decode the image
        image = np.frombuffer(file.read(), np.uint8)
        image = cv2.imdecode(image, cv2.IMREAD_COLOR)

        # Preprocess the image
        processed_image = load_and_preprocess_image(image)

        # Predict the class
        pred = model.predict(processed_image)
        predicted_class = np.argmax(pred)

        # Return the predicted class as a response
        return jsonify({
            'predicted_class': class_names[predicted_class]
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))  # Default to port 5000 if PORT is not set
    app.run(debug=True, host='0.0.0.0', port=port)
