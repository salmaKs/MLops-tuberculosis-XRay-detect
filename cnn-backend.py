from flask import Flask, request, jsonify
from flask_cors import CORS
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np
import os
import keras
import cv2

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
    image = load_img(image_path, target_size=(200, 200))
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)
    image = image / 255.0 
    return image

# API endpoint to handle image upload and prediction
@app.route('/upload', methods=['POST'])
def upload():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['file']
    file_path = os.path.join('uploads', file.filename)
    file.save(file_path)
    try: 
    # Preprocess and predict
        processed_image = preprocess_image(file_path)
        prediction = model.predict(processed_image)
        predicted_class = np.argmax(prediction)
        os.remove(file_path) 

        if predicted_class == 1:
            result = 'TUBERCULOSIS'
        elif predicted_class == 0:
            result = 'HEALTHY'
    # Ensure the logic matches your model's encoding
    #result = 'HEALTHY' if prediction[0][0] > 0.7 else 'TUBERCULOSIS'
    
    # Return the result as a JSON response
        return jsonify({'prediction': result})
    except Exception as e:
        os.remove(file_path)
        return jsonify({'error': str (e)}), 500



if __name__ == '__main__':
    os.makedirs('uploads', exist_ok=True)  # Create an uploads folder
    app.run(debug=True)
