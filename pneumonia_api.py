# from flask import Flask, request, jsonify, render_template
# import torch
# import torchvision.transforms as transforms
# from PIL import Image
# import io
# import os
# import torch.nn as nn
# from torchvision.models import resnet18

# # Initialize Flask app
# app = Flask(__name__)

# # Load trained model
# model = resnet18(weights=None)  # Load architecture
# model.fc = nn.Sequential(
#     nn.Linear(model.fc.in_features, 128),
#     nn.BatchNorm1d(128),
#     nn.ReLU(),
#     nn.Linear(128, 32),
#     nn.BatchNorm1d(32),
#     nn.ReLU(),
#     nn.Linear(32, 2),  # Binary classification (NORMAL vs. PNEUMONIA)
#     nn.Softmax(dim=1)
# )

# # Load trained weights (update 'model.pth' with actual path)
# model_path = os.path.join(os.path.dirname(__file__), "model.pth")
# model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
# model.eval()  # Set to evaluation mode

# # Define image transformation
# transform = transforms.Compose([
#     transforms.Resize((224, 224)),
#     transforms.Grayscale(num_output_channels=3),
#     transforms.ToTensor(),
#     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
# ])

# # Prediction function
# def predict(image: Image.Image):
#     image = transform(image).unsqueeze(0)  # Add batch dimension
#     with torch.no_grad():
#         output = model(image)
#         prediction = torch.argmax(output, dim=1).item()
#         label = "PNEUMONIA" if prediction == 1 else "NORMAL"
#     return label

# @app.route('/')
# def upload_form():
#     return render_template('index.html')

# @app.route('/predict', methods=['POST'])
# def predict_pneumonia():
#     if 'file' not in request.files:
#         return jsonify({"error": "No file uploaded"}), 400
    
#     file = request.files['file']
#     image = Image.open(io.BytesIO(file.read())).convert("RGB")
#     result = predict(image)
    
#     return render_template('index.html', prediction=result)

# if __name__ == "__main__":
#     from pneumonia_api import app
#     app.run(host="0.0.0.0", port=5000)

from flask import Flask, request, jsonify, render_template
import torch
import torchvision.transforms as transforms
from PIL import Image, UnidentifiedImageError
import io
import os
import torch.nn as nn
import pyttsx3  # For text-to-speech
from torchvision.models import resnet18

# Initialize Flask app
app = Flask(__name__, static_url_path='/static')

# Allowed file extensions
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg"}
MAX_FILE_SIZE = 5 * 1024 * 1024  # 5MB limit
EXPECTED_IMAGE_SIZE = (224, 224)  # Expected image size for model

# Load trained model
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

# Initialize text-to-speech engine
engine = pyttsx3.init()
engine.setProperty('rate', 150)  # Set speed of speech

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

# Prediction function
def predict(image: Image.Image):
    image = transform(image).unsqueeze(0)  # Add batch dimension
    with torch.no_grad():
        output = model(image)
        prediction = torch.argmax(output, dim=1).item()
        confidence = torch.max(output).item()
        if confidence < 0.7:
            return "Uncertain Image - Not a valid X-ray", "invalid"
        label = "PNEUMONIA" if prediction == 1 else "NORMAL"
    return label, "valid"

@app.route('/')
def upload_form():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict_pneumonia():
    if 'file' not in request.files:
        return render_template('index.html', error="No file uploaded.")
    
    file = request.files['file']
    
    # Check if file is valid
    if file.filename == "":
        return render_template('index.html', error="No selected file.")
    
    if not allowed_file(file.filename):
        return render_template('index.html', error="Invalid file type. Please upload a PNG, JPG, or JPEG.")
    
    # Check file size
    file.seek(0, os.SEEK_END)
    file_length = file.tell()
    file.seek(0)
    if file_length > MAX_FILE_SIZE:
        return render_template('index.html', error="File too large. Maximum size allowed is 5MB.")
    
    # Try to open the image
    try:
        image = Image.open(io.BytesIO(file.read())).convert("RGB")
        if image.size != EXPECTED_IMAGE_SIZE:
            return render_template('index.html', error="Invalid image size. Please upload an X-ray with 224x224 resolution.")
    except UnidentifiedImageError:
        return render_template('index.html', error="Invalid image file. Please upload a valid PNG, JPG, or JPEG.")
    
    result, status = predict(image)
    
    # Convert text result to speech
    audio_file = "static/result.mp3"
    engine.save_to_file(result, audio_file)
    engine.runAndWait()
    
    return render_template('index.html', prediction=result, audio_file=audio_file)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
