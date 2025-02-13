# from flask import Flask, request, jsonify, render_template
# import torch
# import torchvision.transforms as transforms
# from PIL import Image, UnidentifiedImageError
# import io
# import os
# import torch.nn as nn
# from torchvision.models import resnet18

# # Initialize Flask app
# app = Flask(__name__, static_url_path='/static')

# # Allowed file extensions
# ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg"}

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

# # Function to check file extension
# def allowed_file(filename):
#     return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# # Prediction function
# def predict(image: Image.Image):
#     image = transform(image).unsqueeze(0)  # Add batch dimension
#     with torch.no_grad():
#         output = model(image)
#         prediction = torch.argmax(output, dim=1).item()
#         confidence = torch.max(output).item()
#         if confidence < 0.7:
#             return "Uncertain Image - Not a valid X-ray", "invalid"
#         label = "PNEUMONIA" if prediction == 1 else "NORMAL"
#     return label, "valid"

# @app.route('/')
# def upload_form():
#     return render_template('index.html')

# @app.route('/predict', methods=['POST'])
# def predict_pneumonia():
#     if 'file' not in request.files:
#         return render_template('index.html', error="No file uploaded.")

#     file = request.files['file']

#     if file.filename == "":
#         return render_template('index.html', error="No selected file.")

#     if not allowed_file(file.filename):
#         return render_template('index.html', error="Invalid file type. Please upload a PNG, JPG, or JPEG.")

#     try:
#         image = Image.open(io.BytesIO(file.read())).convert("RGB")
#     except UnidentifiedImageError:
#         return render_template('index.html', error="Invalid image file. Please upload a valid PNG, JPG, or JPEG.")

#     result, status = predict(image)
    
#     return render_template('index.html', prediction=result)

# if __name__ == "__main__":
#     app.run(host="0.0.0.0", port=5000)
from flask import Flask, request, jsonify, render_template, send_from_directory, url_for
import torch
import torchvision.transforms as transforms
from PIL import Image, UnidentifiedImageError
import io
import os
import torch.nn as nn
from gtts import gTTS  # For text-to-speech
import time
from torchvision.models import resnet18

# Initialize Flask app
app = Flask(__name__, static_url_path='/static')

# Allowed file extensions
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg"}

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
def generate_audio(result_text, audio_path):
    static_dir = "static"
    if not os.path.exists(static_dir):
        os.makedirs(static_dir)
    if os.path.exists(audio_path):
        os.remove(audio_path)

    # Generate speech and save as mp3
    tts = gTTS(text=result_text, lang="en")
    tts.save(audio_path)

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

@app.route('/static/<path:filename>')
def serve_static(filename):
    return send_from_directory('static', filename)

@app.route('/predict', methods=['POST'])
def predict_pneumonia():
    if 'file' not in request.files:
        return render_template('index.html', error="No file uploaded.")

    file = request.files['file']

    if file.filename == "":
        return render_template('index.html', error="No selected file.")

    if not allowed_file(file.filename):
        return render_template('index.html', error="Invalid file type. Please upload a PNG, JPG, or JPEG.")

    try:
        image = Image.open(io.BytesIO(file.read())).convert("RGB")
    except UnidentifiedImageError:
        return render_template('index.html', error="Invalid image file. Please upload a valid PNG, JPG, or JPEG.")

    result, status = predict(image)
    
    # Save audio result
    audio_file = "static/result.mp3"
    generate_audio(result, audio_file)
    
    return render_template('index.html', prediction=result, audio_file=url_for('serve_static', filename='result.mp3', _t=int(time.time())))

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
