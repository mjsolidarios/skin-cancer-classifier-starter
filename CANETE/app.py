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
app.config['UPLOAD_FOLDER'] = 'static/uploads'

# Ensure the upload folder exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Load API key from .env file
load_dotenv()
config = dotenv_values('.env')

if 'GEMINI_API_KEY' not in config:
    raise ValueError("GEMINI_API_KEY not found in .env file. Please ensure the .env file is correctly configured.")

# Configure Generative AI
try:
    genai.configure(api_key=config['GEMINI_API_KEY'])
except Exception as e:
    raise ValueError(f"Error configuring Generative AI: {e}")

# Define the base directory for the model and label files
base_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'skin_cancer_model')

# Load the TFLite model
model_path = os.path.join(base_dir, 'model.tflite')

interpreter = tf.lite.Interpreter(model_path=model_path)
interpreter.allocate_tensors()

# Get input and output details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Load labels from labels.txt
with open(os.path.join(base_dir, 'labels.txt'), 'r') as f:
    labels = f.read().strip().split('\n')

# Load label names from labels.json
with open(os.path.join(base_dir, 'labels.json'), 'r') as f:
    label_names = {item['label']: item['name'] for item in json.load(f)}

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def preprocess_image(image_path):
    image = Image.open(image_path).convert('RGB')
    image = image.resize((224, 224))
    image = np.array(image) / 255.0
    image = np.expand_dims(image, axis=0).astype(np.float32)
    return image

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file uploaded.'}), 400

        file = request.files['file']

        if file.filename == '':
            return jsonify({'error': 'No selected file.'}), 400

        if not allowed_file(file.filename):
            return jsonify({'error': 'Invalid file type. Only PNG, JPG, and JPEG are allowed.'}), 400

        file_path = os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(file.filename))
        file.save(file_path)

        input_data = preprocess_image(file_path)

        interpreter.set_tensor(input_details[0]['index'], input_data)
        interpreter.invoke()
        predictions = interpreter.get_tensor(output_details[0]['index'])[0]

        predicted_index = np.argmax(predictions)
        predicted_label = labels[predicted_index]
        predicted_name = label_names[predicted_label]
        confidence = float(predictions[predicted_index])

        os.remove(file_path)

        model = genai.GenerativeModel('gemini-1.5-flash-latest')
        response = model.generate_content(f'Provide a detailed description and recommended treatment for {predicted_name} in 8 sentences.')

        return jsonify({
            'label': predicted_label,
            'name': predicted_name,
            'confidence': confidence,
            'description_and_treatment': response.text if response else 'No response from AI model.'
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
