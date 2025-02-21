# from flask import Flask, request, jsonify, render_template, send_from_directory, url_for
# import torch
# import torchvision.transforms as transforms
# from PIL import Image, UnidentifiedImageError
# import io
# import os
# import torch.nn as nn
# from gtts import gTTS  # For text-to-speech
# import time
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

# # Initialize text-to-speech engine
# def generate_audio(result_text, audio_path):
#     static_dir = "static"
#     if not os.path.exists(static_dir):
#         os.makedirs(static_dir)
#     if os.path.exists(audio_path):
#         os.remove(audio_path)

#     # Generate speech and save as mp3
#     tts = gTTS(text=result_text, lang="en")
#     tts.save(audio_path)

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

# @app.route('/static/<path:filename>')
# def serve_static(filename):
#     return send_from_directory('static', filename)

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
    
#     # Save audio result
#     audio_file = "static/result.mp3"
#     generate_audio(result, audio_file)
    
#     return render_template('index.html', prediction=result, audio_file=url_for('serve_static', filename='result.mp3', _t=int(time.time())))

# if __name__ == "__main__":
#     app.run(host="0.0.0.0", port=5000)



# from flask import Flask, request, jsonify, render_template, send_from_directory, url_for
# import torch
# import torchvision.transforms as transforms
# from PIL import Image, UnidentifiedImageError
# import io
# import os
# import torch.nn as nn
# import time
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
#         output = model(image)  # Get raw model outputs (logits)
#         probabilities = output.squeeze()  # Use raw logits

#         prediction = torch.argmax(probabilities).item()
#         confidence = torch.nn.functional.softmax(probabilities, dim=0)[prediction].item() * 100  # Apply softmax only once

#         if confidence < 70:
#             return "Uncertain Image - Not a valid X-ray", "invalid", confidence

#         label = "PNEUMONIA" if prediction == 1 else "NORMAL"
#     return label, "valid", confidence




# @app.route('/')
# def upload_form():
#     return render_template('index.html')

# @app.route('/static/<path:filename>')
# def serve_static(filename):
#     return send_from_directory('static', filename)

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

#     result, status, confidence = predict(image)
#     prediction_text = f"{result} ({confidence:.2f}% confidence)"
    
#     return render_template('index.html', prediction=prediction_text)


# from flask import Flask, request, jsonify, render_template, send_from_directory, url_for
# import torch
# import torchvision.transforms as transforms
# from PIL import Image, UnidentifiedImageError
# import io
# import os
# import torch.nn as nn
# import time
# from torchvision.models import resnet18
# import torch.nn.functional as F

# # Initialize Flask app
# app = Flask(__name__, static_url_path='/static')

# # Allowed file extensions
# ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg"}

# # Load Pretrained ResNet18 for X-ray detection
# xray_detector = resnet18(pretrained=True)
# xray_detector = torch.nn.Sequential(*(list(xray_detector.children())[:-1]))  # Remove the classification layer
# xray_detector.eval()  # Put model in evaluation mode

# # Load trained model for pneumonia detection
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

# # X-ray detection function using pre-trained CNN
# def is_xray(image: Image.Image):
#     """Use pre-trained CNN to check if the image looks like an X-ray."""
#     image = transform(image).unsqueeze(0)  # Convert to tensor
#     with torch.no_grad():
#         features = xray_detector(image)  # Extract ResNet features
#         avg_feature_response = features.mean().item()  # Get the mean activation

#     return avg_feature_response > 0.72  # Adjust threshold based on performance
    

# # Pneumonia prediction function
# def predict(image: Image.Image):
#     image = transform(image).unsqueeze(0)  # Add batch dimension
#     with torch.no_grad():
#         output = model(image)  # Get raw model outputs (logits)
#         probabilities = torch.nn.functional.softmax(output, dim=1).squeeze()  # Apply softmax
#         prediction = torch.argmax(probabilities).item()
#         confidence = probabilities[prediction].item() * 100  # Convert to percentage

#         if confidence < 70:
#             return "Uncertain Image - Not a valid X-ray", "invalid", confidence

#         label = "PNEUMONIA" if prediction == 1 else "NORMAL"
#     return label, "valid", confidence

# @app.route('/')
# def upload_form():
#     return render_template('index.html')

# @app.route('/static/<path:filename>')
# def serve_static(filename):
#     return send_from_directory('static', filename)

# @app.route('/predict', methods=['POST'])
# def predict_pneumonia():
#     if 'file' not in request.files:
#         return render_template('index.html', error="No file uploaded.")

#     files = request.files.getlist('file')  # Allow multiple file uploads
#     if not files or all(file.filename == "" for file in files):
#         return render_template('index.html', error="No selected files.")

#     predictions = []
#     for file in files:
#         if not allowed_file(file.filename):
#             predictions.append((file.filename, "Invalid file type"))
#             continue

#         try:
#             image = Image.open(io.BytesIO(file.read())).convert("RGB")
#         except UnidentifiedImageError:
#             predictions.append((file.filename, "Invalid image file"))
#             continue

#         # **Step 1: Check if it's an X-ray**
#         if not is_xray(image):
#             predictions.append((file.filename, "Rejected: This is not an X-ray"))
#             continue

#         # **Step 2: Proceed with Pneumonia Classification**
#         result, status, confidence = predict(image)
#         prediction_text = f"{result} ({confidence:.2f}% confidence)"
#         predictions.append((file.filename, prediction_text))

#     return render_template('index.html', predictions=predictions)

# if __name__ == "__main__":
#     app.run(host="0.0.0.0", port=5000)


# from flask import Flask, request, jsonify, render_template, send_from_directory, url_for
# import torch
# import torchvision.transforms as transforms
# from PIL import Image, UnidentifiedImageError
# import io
# import os
# import torch.nn as nn
# import time
# import pydicom
# from torchvision.models import resnet18
# import torch.nn.functional as F

# # Initialize Flask app
# app = Flask(__name__, static_url_path='/static')

# # Ensure static storage for uploaded images
# UPLOAD_FOLDER = "static/uploads"
# if not os.path.exists(UPLOAD_FOLDER):
#     os.makedirs(UPLOAD_FOLDER)

# STORED_TESTS_FOLDER = "static/stored_tests"
# if not os.path.exists(STORED_TESTS_FOLDER):
#     os.makedirs(STORED_TESTS_FOLDER)

# # Allowed file extensions
# ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "dcm"}

# # Load Pretrained ResNet18 for Feature Extraction
# xray_detector = resnet18(pretrained=True)
# xray_detector = torch.nn.Sequential(*(list(xray_detector.children())[:-1]))  # Remove classification layer
# xray_detector.eval()

# # Load trained model for pneumonia detection
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

# # Function to load DICOM images
# def load_dicom(file):
#     dicom = pydicom.dcmread(file)
#     image = dicom.pixel_array  # Extract pixel data
#     return Image.fromarray(image)

# # X-ray detection function
# def is_xray(image: Image.Image):
#     """Use pre-trained CNN to check if the image looks like an X-ray."""
#     image = transform(image).unsqueeze(0)  # Convert to tensor
#     with torch.no_grad():
#         features = xray_detector(image)  # Extract ResNet features
#         avg_feature_response = features.mean().item()  # Compute mean activation

#     return avg_feature_response > 0.72  # Adjust threshold based on testing

# # Pneumonia prediction function
# def predict(image: Image.Image):
#     image = transform(image).unsqueeze(0)  # Add batch dimension
#     with torch.no_grad():
#         output = model(image)  # Get raw model outputs (logits)
#         probabilities = torch.nn.functional.softmax(output, dim=1).squeeze()  # Apply softmax
#         prediction = torch.argmax(probabilities).item()
#         confidence = probabilities[prediction].item() * 100  # Convert to percentage

#         if confidence < 70:
#             return "Uncertain Image - Not a valid X-ray", "invalid", confidence

#         label = "PNEUMONIA" if prediction == 1 else "NORMAL"
#     return label, "valid", confidence

# @app.route('/')
# def upload_form():
#     return render_template('index.html')

# @app.route('/static/<path:filename>')
# def serve_static(filename):
#     return send_from_directory('static', filename)

# @app.route('/predict', methods=['POST'])
# def predict_pneumonia():
#     if 'file' not in request.files:
#         return render_template('index.html', error="No file uploaded.")

#     files = request.files.getlist('file')  # Allow multiple file uploads
#     if not files or all(file.filename == "" for file in files):
#         return render_template('index.html', error="No selected files.")

#     predictions = []
#     stored_files = []
#     for file in files:
#         if not allowed_file(file.filename):
#             predictions.append((file.filename, "Invalid file type"))
#             continue

#         try:
#             if file.filename.lower().endswith(".dcm"):
#                 image = load_dicom(file)
#             else:
#                 image = Image.open(io.BytesIO(file.read())).convert("RGB")
#         except UnidentifiedImageError:
#             predictions.append((file.filename, "Invalid image file"))
#             continue

#         # Save image for later viewing
#         save_path = os.path.join(UPLOAD_FOLDER, file.filename)
#         image.save(save_path)
#         stored_files.append(file.filename)

#         # **Step 1: Check if it's an X-ray**
#         if not is_xray(image):
#             predictions.append((file.filename, "Rejected: This is not an X-ray"))
#             continue

#         # **Step 2: Proceed with Pneumonia Classification**
#         result, status, confidence = predict(image)
#         prediction_text = f"{result} ({confidence:.2f}% confidence)"
#         predictions.append((file.filename, prediction_text))

#     return render_template('index.html', predictions=predictions, stored_files=stored_files)

# @app.route('/store_tests', methods=['POST'])
# def store_tests():
#     for image in os.listdir(UPLOAD_FOLDER):
#         source = os.path.join(UPLOAD_FOLDER, image)
#         destination = os.path.join(STORED_TESTS_FOLDER, image)
#         os.rename(source, destination)
#     return jsonify({"success": True})

# @app.route('/view_tests')
# def view_tests():
#     images = os.listdir(STORED_TESTS_FOLDER)
#     return render_template('view_tests.html', images=images)

# if __name__ == "__main__":
#     app.run(host="0.0.0.0", port=5000)
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

# Initialize Flask app
app = Flask(__name__, static_url_path='/static')

# Ensure static storage for uploaded images
UPLOAD_FOLDER = "static/uploads"
BATCHES_FOLDER = "static/batches"
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)
if not os.path.exists(BATCHES_FOLDER):
    os.makedirs(BATCHES_FOLDER)

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

    files = request.files.getlist('file')  # Allow multiple file uploads
    if not files or all(file.filename == "" for file in files):
        return render_template('index.html', error="No selected files.")

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

        save_path = os.path.join("static/uploads", file.filename)
        image.save(save_path)
        stored_files.append(file.filename)

        result, _, confidence = predict(image)
        prediction_text = f"{result} ({confidence:.2f}% confidence)"
        predictions.append((file.filename, prediction_text))

    return render_template('index.html', predictions=predictions, stored_files=stored_files)


@app.route('/store_tests', methods=['POST'])
def store_tests():
    folder_name = request.args.get('folder', 'default_batch')
    batch_folder = os.path.join(BATCHES_FOLDER, folder_name)
    if not os.path.exists(batch_folder):
        os.makedirs(batch_folder)
    
    for image in os.listdir(UPLOAD_FOLDER):
        source = os.path.join(UPLOAD_FOLDER, image)
        destination = os.path.join(batch_folder, image)
        os.rename(source, destination)
    return jsonify({"success": True})

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
    return render_template('view_tests.html', images=images, batch_name=batch_name)

@app.route('/quit', methods=['POST'])
def quit_app():
    """Terminate Flask application."""
    os.kill(os.getpid(), signal.SIGTERM)  # Kill the Flask process
    return jsonify({"success": True})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
