from flask import Flask, render_template, request, send_file, jsonify, url_for
from werkzeug.utils import secure_filename
import os
import shutil
import numpy as np
from PIL import Image
import tensorflow as tf

app = Flask(__name__)

# Configuration
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['PROCESSED_FOLDER'] = 'processed'
MODEL_PATH = r"D:/AI LEARN/RENAMEAIBDMS/flask/keras_model.h5"
LABELS_PATH = r"D:/AI LEARN/RENAMEAIBDMS/flask/labels.txt"
CONFIDENCE_THRESHOLD = 0.60

# Ensure directories exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['PROCESSED_FOLDER'], exist_ok=True)

# Load the ML model and labels
model = tf.keras.models.load_model(MODEL_PATH)
with open(LABELS_PATH, 'r', encoding='utf-8') as f:
    labels = f.read().splitlines()

def classify_image(image_path):
    """Classify image using the pre-trained model."""
    img = Image.open(image_path).convert('RGB')
    img = img.resize((224, 224))  # Assuming the model expects 224x224 input
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    prediction = model.predict(img_array)
    confidence = float(np.max(prediction))
    class_idx = int(np.argmax(prediction))
    label = labels[class_idx] if 0 <= class_idx < len(labels) else "Uncertain"
    return label, confidence

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_images():
    """Handle image uploads."""
    files = request.files.getlist('images')
    if not files:
        return jsonify({'error': 'No files uploaded'}), 400

    results = []
    for file in files:
        if file and file.filename:
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            results.append({
                'url': url_for('serve_image', filename=filename),
                'display_name': filename
            })

    return jsonify({'files': results})

@app.route('/process', methods=['POST'])
def process_images():
    """Process uploaded images using the ML model."""
    files = os.listdir(app.config['UPLOAD_FOLDER'])
    results = []
    for filename in files:
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        label, confidence = classify_image(filepath)
        if confidence >= CONFIDENCE_THRESHOLD:
            new_name = f"{label}.png"
        else:
            new_name = f"SIDE_{filename}"

        new_path = os.path.join(app.config['PROCESSED_FOLDER'], new_name)
        shutil.move(filepath, new_path)
        results.append({
            'original': filename,
            'new_name': new_name,
            'confidence': confidence
        })

    return jsonify({'results': results})

@app.route('/serve_image/<filename>')
def serve_image(filename):
    """Serve images from the upload or processed folder."""
    file_path = os.path.join(app.config['PROCESSED_FOLDER'], filename)
    if not os.path.exists(file_path):
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    if not os.path.exists(file_path):
        return "File not found", 404
    return send_file(file_path)

@app.route('/download', methods=['GET'])
def download_images():
    """Download all processed images in a zip file."""
    zip_filename = 'processed_images'
    shutil.make_archive(zip_filename, 'zip', app.config['PROCESSED_FOLDER'])
    return send_file(f"{zip_filename}.zip", as_attachment=True)

if __name__ == '__main__':
    app.run(debug=True)
