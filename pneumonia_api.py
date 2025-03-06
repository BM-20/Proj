from flask import Flask, request, jsonify, render_template, send_from_directory, url_for
import torch
import torchvision.transforms as transforms
from PIL import Image, UnidentifiedImageError
import io
import os
import torch.nn as nn
import time
import pydicom
from torchvision.models import resnet18
import torch.nn.functional as F
import signal
import sys
import json

# Initialize Flask app
app = Flask(__name__, static_url_path='/static')

# Ensure static storage for uploaded images
UPLOAD_FOLDER = "static/uploads"
BATCHES_FOLDER = "static/batches"
RESULTS_FILE = "static/results.json"
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)
if not os.path.exists(BATCHES_FOLDER):
    os.makedirs(BATCHES_FOLDER)

# Load previous results if available
if os.path.exists(RESULTS_FILE):
    with open(RESULTS_FILE, "r") as f:
        stored_results = json.load(f)
else:
    stored_results = {}

# Function to save results
def save_results():
    with open(RESULTS_FILE, "w") as f:
        json.dump(stored_results, f, indent=4)
# Allowed file extensions
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "dcm"}

# Load Pretrained ResNet18 for Feature Extraction
xray_detector = resnet18(pretrained=True)
xray_detector = torch.nn.Sequential(*(list(xray_detector.children())[:-1]))  # Remove classification layer
xray_detector.eval()

# Load trained model for pneumonia detection
model = resnet18(weights=None)  # Load architecture
model.fc = nn.Sequential(
    nn.Linear(model.fc.in_features, 128),
    nn.BatchNorm1d(128),
    nn.ReLU(),
    nn.Linear(128, 32),
    nn.BatchNorm1d(32),
    nn.ReLU(),
    nn.Linear(32, 2),  # Binary classification (NORMAL vs. PNEUMONIA)
    nn.Softmax(dim=1)
)

# Load trained weights (update 'model.pth' with actual path)
model_path = os.path.join(os.path.dirname(__file__), "model.pth")
model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
model.eval()  # Set to evaluation mode

# Define image transformation
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Function to check file extension
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Function to load DICOM images
def load_dicom(file):
    dicom = pydicom.dcmread(file)
    image = dicom.pixel_array  # Extract pixel data
    return Image.fromarray(image)

# X-ray detection function
def is_xray(image: Image.Image):
    """Use pre-trained CNN to check if the image looks like an X-ray."""
    image = transform(image).unsqueeze(0)  # Convert to tensor
    with torch.no_grad():
        features = xray_detector(image)  # Extract ResNet features
        avg_feature_response = features.mean().item()  # Compute mean activation

    return avg_feature_response > 0.72  # Adjust threshold based on testing

# Pneumonia prediction function
def predict(image: Image.Image):
    image = transform(image).unsqueeze(0)  # Add batch dimension
    with torch.no_grad():
        output = model(image)  # Get raw model outputs (logits)
        probabilities = torch.nn.functional.softmax(output, dim=1).squeeze()  # Apply softmax
        prediction = torch.argmax(probabilities).item()
        confidence = probabilities[prediction].item() * 100  # Convert to percentage

        if confidence < 70:
            return "Uncertain Image - Not a valid X-ray", "invalid", confidence

        label = "PNEUMONIA" if prediction == 1 else "NORMAL"
    return label, "valid", confidence

@app.route('/')
def upload_form():
    return render_template('index.html')
@app.route('/predict', methods=['POST'])
def predict_pneumonia():
    if 'file' not in request.files:
        return render_template('index.html', error="No file uploaded.")

    files = request.files.getlist('file')  # Allow multiple uploads
    if not files or all(file.filename == "" for file in files):
        return render_template('index.html', error="No selected files.")

    # Load previous results (if any)
    if os.path.exists(RESULTS_FILE):
        with open(RESULTS_FILE, "r") as f:
            stored_results = json.load(f)
    else:
        stored_results = {}

    predictions = []
    stored_files = []
    for file in files:
        if not allowed_file(file.filename):
            predictions.append((file.filename, "Invalid file type"))
            continue

        try:
            image = Image.open(io.BytesIO(file.read())).convert("RGB")
        except UnidentifiedImageError:
            predictions.append((file.filename, "Invalid image file"))
            continue

        save_path = os.path.join(UPLOAD_FOLDER, file.filename)
        image.save(save_path)
        stored_files.append(file.filename)

        result, _, confidence = predict(image)
        prediction_text = f"{result} ({confidence:.2f}% confidence)"
        predictions.append((file.filename, prediction_text))

        # Save result in global storage
        stored_results[file.filename] = prediction_text

    # Save predictions persistently
    with open(RESULTS_FILE, "w") as f:
        json.dump(stored_results, f, indent=4)

    return render_template('index.html', predictions=predictions, stored_files=stored_files)


@app.route('/store_tests', methods=['POST'])
def store_tests():
    folder_name = request.args.get('folder', 'default_batch')
    batch_folder = os.path.join(BATCHES_FOLDER, folder_name)
    
    if not os.path.exists(batch_folder):
        os.makedirs(batch_folder)

    # Load global stored predictions
    if os.path.exists(RESULTS_FILE):
        with open(RESULTS_FILE, "r") as f:
            stored_results = json.load(f)
    else:
        stored_results = {}

    batch_results = []  # List to store image details

    for image in os.listdir(UPLOAD_FOLDER):
        source = os.path.join(UPLOAD_FOLDER, image)
        destination = os.path.join(batch_folder, image)
        os.rename(source, destination)  # Move image to batch folder

        # Copy stored predictions for this image
        prediction_text = stored_results.get(image, "Prediction Unavailable")
        batch_results.append({"filename": image, "prediction": prediction_text})

    # Save batch-specific predictions in results.json inside the batch folder
    batch_results_path = os.path.join(batch_folder, "results.json")
    with open(batch_results_path, "w") as f:
        json.dump(batch_results, f, indent=4)

    return jsonify({"success": True, "message": f"Batch '{folder_name}' stored successfully!"})


@app.route('/view_batches')
def view_batches():
    batches = os.listdir(BATCHES_FOLDER)
    return render_template('view_batches.html', batches=batches)

@app.route('/delete_batch/<batch_name>', methods=['POST'])
def delete_batch(batch_name):
    batch_folder = os.path.join(BATCHES_FOLDER, batch_name)
    if os.path.exists(batch_folder):
        for file in os.listdir(batch_folder):
            os.remove(os.path.join(batch_folder, file))
        os.rmdir(batch_folder)
    return jsonify({"success": True})

@app.route('/view_tests/<batch_name>')
def view_tests(batch_name):
    batch_folder = os.path.join(BATCHES_FOLDER, batch_name)
    images = os.listdir(batch_folder) if os.path.exists(batch_folder) else []

    # Load predictions from the batch's results.json
    batch_results_path = os.path.join(batch_folder, "results.json")
    if os.path.exists(batch_results_path):
        with open(batch_results_path, "r") as f:
            batch_results = json.load(f)
    else:
        batch_results = []

    return render_template('view_tests.html', images=batch_results, batch_name=batch_name)



@app.route('/quit', methods=['POST'])
def quit_app():
    """Properly terminate Flask application."""
    if sys.platform == "win32":
        os._exit(0)  # Force exit on Windows
    else:
        os.kill(os.getpid(), signal.SIGTERM)  # Unix-based systems
    return jsonify({"success": True})  # Won't reach this

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)