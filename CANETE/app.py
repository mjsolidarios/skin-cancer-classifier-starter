import os
from flask import Flask, request, jsonify, render_template
import tensorflow as tf
import numpy as np
from PIL import Image
import json
from werkzeug.utils import secure_filename
import google.generativeai as genai
from dotenv import load_dotenv, dotenv_values

# Suppress TensorFlow logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress all logs except errors

app = Flask(__name__)

# Load API key from .env file
load_dotenv()
config = dotenv_values('.env')

if 'GEMINI_API_KEY' not in config:
    raise ValueError("GEMINI_API_KEY not found in .env file. Please ensure the .env file is correctly configured.")

genai.configure(api_key=config['GEMINI_API_KEY'])

# Define the base directory for the model and label files
base_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'skin_cancer_model')

# Debug: Print the base directory
print(f"Base directory: {base_dir}")

# Load the TFLite model
model_path = os.path.join(base_dir, 'model.tflite')
print(f"Looking for model at: {model_path}")  # Debug statement

if not os.path.exists(model_path):
    raise FileNotFoundError(
        f'Model file not found at {model_path}. '
        'Please ensure the "skin_cancer_model" directory and its contents are included in your deployment.'
    )

interpreter = tf.lite.Interpreter(model_path=model_path)
interpreter.allocate_tensors()

# Get input and output details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Load labels from labels.txt
labels_path = os.path.join(base_dir, 'labels.txt')
if not os.path.exists(labels_path):
    raise FileNotFoundError(f'Labels file not found at {labels_path}. Please ensure the labels file is in the correct directory.')

with open(labels_path, 'r') as f:
    labels = f.read().strip().split('\n')

# Load label names from labels.json
labels_json_path = os.path.join(base_dir, 'labels.json')
if not os.path.exists(labels_json_path):
    raise FileNotFoundError(f'Labels JSON file not found at {labels_json_path}. Please ensure the labels JSON file is in the correct directory.')

with open(labels_json_path, 'r') as f:
    label_names = {item['label']: item['name'] for item in json.load(f)}

# Allowed file extensions
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def preprocess_image(image_path):
    try:
        image = Image.open(image_path).convert('RGB')
        image = image.resize((224, 224))  # Resize to match model's expected input size
        image = np.array(image) / 255.0   # Normalize pixel values
        image = np.expand_dims(image, axis=0).astype(np.float32)
        return image
    except Exception as e:
        raise ValueError(f"Error preprocessing image: {e}")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        if 'example' in request.form:
            example_image = request.form['example']
            file_path = os.path.join('static', 'images', example_image)
        else:
            if 'file' not in request.files:
                return jsonify({'error': 'No file uploaded.'}), 400

            file = request.files['file']
            if file.filename == '':
                return jsonify({'error': 'No selected file.'}), 400

            if not allowed_file(file.filename):
                return jsonify({'error': 'Invalid file type. Only PNG, JPG, and JPEG are allowed.'}), 400

            file_path = os.path.join('static/uploads', secure_filename(file.filename))
            file.save(file_path)

        # Preprocess the image
        input_data = preprocess_image(file_path)

        # Perform inference
        interpreter.set_tensor(input_details[0]['index'], input_data)
        interpreter.invoke()
        predictions = interpreter.get_tensor(output_details[0]['index'])[0]

        # Get the predicted label and confidence
        predicted_index = np.argmax(predictions)
        predicted_label = labels[predicted_index]
        predicted_name = label_names[predicted_label]
        confidence = float(predictions[predicted_index])

        # Clean up uploaded file
        if 'file' in request.files and os.path.exists(file_path):
            os.remove(file_path)

        # Generate treatment and description using Google Generative AI
        model = genai.GenerativeModel('gemini-1.5-flash-latest')
        response = model.generate_content(
            f'Provide a detailed description and recommended treatment for {predicted_name} in 8 sentences.'
        )
        description_and_treatment = response.text

        return jsonify({
            'label': predicted_label,
            'name': predicted_name,
            'confidence': confidence,
            'description_and_treatment': description_and_treatment
        })
    except FileNotFoundError as e:
        return jsonify({'error': str(e)}), 404
    except Exception as e:
        print('Error during prediction:', str(e))
        return jsonify({'error': 'An error occurred during prediction.'}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)