import os
import torch
import torch.nn as nn
from torchvision import transforms, models
from torchvision.models import GoogLeNet_Weights
from PIL import Image
from flask import Flask, render_template, request

# ===============================
# CONFIG
# ===============================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
UPLOAD_FOLDER = "static/uploads"

CLASS_NAMES = [
    'AnnualCrop', 'Forest', 'HerbaceousVegetation',
    'Highway', 'Industrial', 'Pasture',
    'PermanentCrop', 'Residential',
    'River', 'SeaLake'
]

# ===============================
# FLASK APP
# ===============================
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# ===============================
# IMAGE TRANSFORM (SAME AS TRAINING)
# ===============================
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# ===============================
# LOAD MODEL
# ===============================
model = models.googlenet(weights=GoogLeNet_Weights.DEFAULT)
model.fc = nn.Linear(model.fc.in_features, len(CLASS_NAMES))
model.load_state_dict(torch.load("googlenet_cloud_model.pth", map_location=DEVICE))
model = model.to(DEVICE)
model.eval()

# ===============================
# ROUTES
# ===============================
@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    confidence = None
    image_path = None

    if request.method == "POST":
        file = request.files["image"]
        if file:
            image_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(image_path)

            image = Image.open(image_path).convert("RGB")
            image = transform(image).unsqueeze(0).to(DEVICE)

            with torch.no_grad():
                outputs = model(image)
                probs = torch.softmax(outputs, dim=1)
                confidence, predicted = torch.max(probs, 1)

            prediction = CLASS_NAMES[predicted.item()]
            confidence = round(confidence.item() * 100, 2)

    return render_template(
        "index.html",
        prediction=prediction,
        confidence=confidence,
        image_path=image_path
    )

# ===============================
# RUN SERVER
# ===============================
if __name__ == "__main__":
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    app.run(debug=False,port="0.0.0.0")

