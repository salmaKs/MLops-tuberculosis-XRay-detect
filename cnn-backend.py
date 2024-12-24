from flask import Flask, request, jsonify
from flask_cors import CORS
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np
import os
import keras

app = Flask(__name__)
CORS(app, resources={r"/upload": {"origins": "http://localhost:4200"}})
@app.after_request
def after_request(response):
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
    response.headers.add('Access-Control-Allow-Methods', 'GET,POST,OPTIONS')
    response.headers.add('Access-Control-Allow-Origin', 'http://localhost:4200')
    return response

@app.route("/")
def home():
    return "Welcome to the CNN Backend!"

# Load the model
MODEL_PATH = "C:\\Users\\HP\\BackendMLops\\cnn_model.keras"
print(f"Attempting to load model from: {MODEL_PATH}")
model = load_model(MODEL_PATH)

# Preprocess the image to match the model input
def preprocess_image(image_path):
    image = load_img(image_path, target_size=(200, 200))  # Resize to match your model
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    image = image / 255.0  # Normalize
    return image

# API endpoint to handle image upload and prediction
@app.route('/upload', methods=['POST'])
def upload():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['file']
    file_path = os.path.join('uploads', file.filename)
    file.save(file_path)

    # Preprocess and predict
    processed_image = preprocess_image(file_path)
    prediction = model.predict(processed_image)
    os.remove(file_path)  # Clean up the saved file

    # Return the prediction result
    result = 'TUBERCULOSIS' if prediction[0][0] > 0.5 else 'HEALTHY'
    return jsonify({'prediction': result})

if __name__ == '__main__':
    os.makedirs('uploads', exist_ok=True)  # Create an uploads folder
    app.run(debug=True)
