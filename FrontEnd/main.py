from __future__ import annotations

import sys
from pathlib import Path

from flask import Flask, jsonify, request
from PIL import Image
import torch
from torchvision import transforms

BASE_DIR = Path(__file__).resolve().parents[1]
FRONTEND_DIR = Path(__file__).resolve().parent
MODEL_FILENAME = "wildfire_model_epoch7_acc0.720_20251116_194209.pth"

if str(BASE_DIR) not in sys.path:
    sys.path.append(str(BASE_DIR))

from improved_model import Config, ImprovedWildfireCNN, convert_to_rgb  # noqa: E402


app = Flask(__name__, static_folder=str(FRONTEND_DIR), static_url_path="")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

inference_transforms = transforms.Compose([
    transforms.Lambda(convert_to_rgb),
    transforms.Resize((Config.IMG_SIZE, Config.IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


def load_model() -> tuple[torch.nn.Module, list[str]]:
    model_path = BASE_DIR / MODEL_FILENAME
    if not model_path.exists():
        raise FileNotFoundError(
            f"Model checkpoint '{MODEL_FILENAME}' was not found in {BASE_DIR}."
        )

    checkpoint = torch.load(model_path, map_location=device)
    model = ImprovedWildfireCNN(num_classes=Config.NUM_CLASSES)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    class_names = checkpoint.get("class_names") or ["No Fire", "Fire"]
    return model, class_names


model, class_names = load_model()


@app.route("/")
def index():
    return app.send_static_file("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "Empty filename"}), 400

    try:
        image = Image.open(file.stream).convert("RGB")
        tensor = inference_transforms(image).unsqueeze(0).to(device)
    except Exception as exc:
        return jsonify({"error": f"Unable to process image: {exc}"}), 400

    with torch.inference_mode():
        outputs = model(tensor)
        probabilities = torch.softmax(outputs, dim=1)
        confidence, predicted_idx = probabilities.max(dim=1)

    prediction = class_names[predicted_idx.item()] if predicted_idx.item() < len(class_names) else "Unknown"

    return jsonify({
        "label": prediction,
        "confidence": round(confidence.item() * 100, 2),
        "probabilities": {
            class_names[i] if i < len(class_names) else f"class_{i}": round(prob.item() * 100, 2)
            for i, prob in enumerate(probabilities.squeeze())
        }
    })


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)