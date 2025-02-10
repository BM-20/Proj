from flask import Flask, request, jsonify
import torch
import torchvision.transforms as transforms
from PIL import Image
import io
import torch.nn as nn
from torchvision.models import resnet18

# Initialize Flask app
app = Flask(__name__)

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
model.load_state_dict(torch.load("model.pth", map_location=torch.device('cpu')))
model.eval()  # Set to evaluation mode

# Define image transformation
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Prediction function
def predict(image: Image.Image):
    image = transform(image).unsqueeze(0)  # Add batch dimension
    with torch.no_grad():
        output = model(image)
        prediction = torch.argmax(output, dim=1).item()
        label = "PNEUMONIA" if prediction == 1 else "NORMAL"
    return label

@app.route("/predict", methods=["POST"])
def predict_pneumonia():
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400
    
    file = request.files['file']
    image = Image.open(io.BytesIO(file.read())).convert("RGB")
    result = predict(image)
    
    return jsonify({"prediction": result})

if __name__ == "__main__":
    from pneumonia_api import app  # Explicitly import the app
    app.run(host="0.0.0.0", port=5000)

