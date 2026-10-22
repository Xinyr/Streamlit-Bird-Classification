
import json
from pathlib import Path

import streamlit as st
import torch
from PIL import Image
from torch import nn
from torchvision import models, transforms

BASE = Path(__file__).parent
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the species names.
with open(BASE / "class_names.json", encoding="utf-8") as f:
    CLASS_NAMES = {
        int(k): v for k, v in json.load(f).items()
    }

# Rebuild the same architecture used during training.
model = models.resnet50(weights=None)
num_features = model.fc.in_features

model.fc = nn.Sequential(
    nn.Dropout(p=0.4),
    nn.Linear(num_features, 200)
)

checkpoint = torch.load(
    BASE / "best_resnet50.pth",
    map_location="cpu",
    weights_only=True
)

# Support either a training checkpoint or a raw state dictionary.
state_dict = checkpoint.get("model_state_dict", checkpoint) \
    if isinstance(checkpoint, dict) else checkpoint

model.load_state_dict(state_dict)
model.to(DEVICE)
model.eval()

# Use the same normalization as during validation.
preprocess = transforms.Compose([
    transforms.Resize(255),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
        [0.485, 0.456, 0.406],
        [0.229, 0.224, 0.225]
    ),
])

def predict(image: Image.Image):
    if image is None:
        return {}

    x = preprocess(image.convert("RGB")).unsqueeze(0).to(DEVICE)

    with torch.inference_mode():
        probabilities = torch.softmax(model(x)[0], dim=0)

    scores, indices = probabilities.topk(5)

    return {
        CLASS_NAMES[int(i)]: float(score)
        for score, i in zip(scores.cpu(), indices.cpu())
    }

demo = st.Interface(
    fn=predict,
    inputs=st.Image(type="pil", label="Upload a bird photo"),
    outputs=st.Label(num_top_classes=5, label="Top predictions"),
    title="Bird Species Classifier",
    description="Upload a bird image to see the five most likely species."
)

if __name__ == "__main__":
    demo.launch()
