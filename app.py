from flask import Flask, request, render_template, redirect, url_for
from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image
import io
import os

app = Flask(__name__)

# Load the trained model
MODEL_PATH = 'model/nail_disease_model.keras'
model = load_model(MODEL_PATH)

# Define class names
class_names = ['healthy', 'onychomycosis', 'psoriasis']

# Define image size
IMG_SIZE = 224

def preprocess_image(image):
    """
    Preprocess the input image for model prediction.
    """
    img = image.resize((IMG_SIZE, IMG_SIZE))
    img_array = np.array(img) / 255.0  # Normalize to [0,1]
    if img_array.shape[-1] == 4:
        img_array = img_array[..., :3]  # Remove alpha channel if present
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    return img_array

@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            return redirect(request.url)
        if file:
            try:
                img = Image.open(file.stream)
                img_preprocessed = preprocess_image(img)
                predictions = model.predict(img_preprocessed)
                predicted_class = class_names[np.argmax(predictions)]
                confidence = np.max(predictions) * 100
                # Save the uploaded image to display
                img_path = os.path.join('static', 'uploaded_image.png')
                img.save(img_path)
                return render_template('index.html', prediction=predicted_class, confidence=confidence, img_path=img_path)
            except Exception as e:
                print(e)
                return redirect(request.url)
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
