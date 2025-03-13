from flask import Flask, request, jsonify, render_template
import os
import numpy as np
from PIL import Image, UnidentifiedImageError
import tensorflow as tf
import json
import logging
from google import genai
from dotenv import load_dotenv, dotenv_values


load_dotenv()
config = dotenv_values(".env")


client = genai.Client(api_key=config.get('GEMINI_API_KEY', ''))


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


BASE_DIR = os.path.abspath(os.path.dirname(__file__))
TEMPLATE_DIR = os.path.join(BASE_DIR, "frontend")

app = Flask(__name__, template_folder=TEMPLATE_DIR)


UPLOAD_FOLDER = os.path.join(BASE_DIR, "uploads")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


LABELS_FILE = os.path.join(BASE_DIR, "models/labels.json")

def load_labels():
    if os.path.exists(LABELS_FILE):
        with open(LABELS_FILE, "r") as f:
            label_data = json.load(f)
        class_labels = [entry["name"] for entry in label_data]  
      
        return class_labels
    else:
        logger.error("labels.json file not found")
        return []

class_labels = load_labels()

# Load TensorFlow Lite model
def load_model():
    MODEL_PATH = os.path.join(BASE_DIR, "models", "model.tflite")
    

    try:
        interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
        interpreter.allocate_tensors()
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        return interpreter, input_details, output_details
        
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        raise RuntimeError("Failed to load the model")

interpreter, input_details, output_details = load_model()

def process_image(image_path):
    try:
        img = Image.open(image_path).convert("RGB")
        img = img.resize((224, 224)) 
        img = np.array(img, dtype=np.float32) / 255.0  
        img = np.expand_dims(img, axis=0) 
        return img
    except UnidentifiedImageError:
        logger.error("Invalid image format")
        raise ValueError("Invalid image format")
    except Exception as e:
        logger.error(f"Error processing image: {e}")
        raise

def generate_cancer_details(disease_name):
    """Generate a three-sentence summary about the detected skin cancer."""
    try:
        response = client.models.generate_content(
            model="gemini-2.0-flash",
            contents=[
                f"Provide a concise 3-sentence description of {disease_name}, including causes, risk factors, and possible treatments."
            ]
        )
        return response.text.strip() if response else "No AI-generated details available."
    except Exception as e:
        logger.error(f"Error generating AI details: {e}")
        return "Error generating AI details."

@app.route("/", methods=["GET"])
def home():
    return render_template("index.html")

@app.route("/upload", methods=["POST"])
def upload_image():
    logger.info("Received request to upload image")

    if "image" not in request.files:
        logger.error("No file part in request")
        return jsonify({"error": "No file part"}), 400

    file = request.files["image"]
    if file.filename == "":
        logger.error("No selected file")
        return jsonify({"error": "No selected file"}), 400

    try:
        file_path = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
        file.save(file_path)
        logger.info(f"Image saved to {file_path}")

        img = process_image(file_path).astype(np.float32)  

        # Perform inference
        interpreter.set_tensor(input_details[0]["index"], img)
        interpreter.invoke()
        predictions = interpreter.get_tensor(output_details[0]["index"])
        logger.info(f"Raw Predictions: {predictions}")
        logger.info(f"Labels: {class_labels}")
       


   
        predicted_class_index = np.argmax(predictions)
        logger.info(f"Predicted class index: {predicted_class_index}")
        confidence_score = float(predictions[0][predicted_class_index])

        
        if predicted_class_index < len(class_labels):
            predicted_class = class_labels[predicted_class_index]
        else:
            predicted_class = "Unknown"

        ai_details = generate_cancer_details(predicted_class)

        response = {
            "prediction": predicted_class,
            "confidence": confidence_score,
            "ai_generated_info": ai_details
        }

        logger.info(f"Prediction: {predicted_class}, Confidence: {confidence_score}")
        return jsonify(response)

    except ValueError as e:
        logger.error(f"Invalid input: {e}")
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)