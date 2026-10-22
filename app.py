
import json
from pathlib import Path

import streamlit as st
import torch
from PIL import Image
from torch import nn
from torchvision import models, transforms

BASE = Path(__file__).parent
DEVICE = torch.device("cpu")

st.set_page_config(
    page_title="Bird Species Classifier",
    page_icon="🐦",
    layout="centered"
)

st.title("Bird Species Classifier")
st.write(
    "Upload a bird photo to see the five most likely species."
)

# Load species names.
with open(BASE / "class_names.json", encoding="utf-8") as f:
    CLASS_NAMES = {
        int(k): v for k, v in json.load(f).items()
    }


# Load the trained model once and cache it.
@st.cache_resource
def load_model():
    model = models.resnet50(weights=None)
    num_features = model.fc.in_features

    # Must match the architecture used during training.
    model.fc = nn.Sequential(
        nn.Dropout(p=0.4),
        nn.Linear(num_features, 200)
    )

    checkpoint = torch.load(
        BASE / "best_resnet50.pth",
        map_location="cpu",
        weights_only=True
    )

    state_dict = (
        checkpoint["model_state_dict"]
        if isinstance(checkpoint, dict)
        and "model_state_dict" in checkpoint
        else checkpoint
    )

    model.load_state_dict(state_dict)
    model.to(DEVICE)
    model.eval()

    return model


# Match the validation preprocessing.
preprocess = transforms.Compose([
    transforms.Resize(255),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
        [0.485, 0.456, 0.406],
        [0.229, 0.224, 0.225]
    ),
])


uploaded_file = st.file_uploader(
    "Choose a bird photo",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    try:
        image = Image.open(uploaded_file).convert("RGB")

        st.image(
            image,
            caption="Uploaded bird photo",
            use_container_width=True
        )

        with st.spinner("Classifying bird species..."):
            model = load_model()
            x = preprocess(image).unsqueeze(0).to(DEVICE)

            with torch.inference_mode():
                probabilities = torch.softmax(
                    model(x)[0], dim=0
                )

            scores, indices = probabilities.topk(5)

        st.subheader("Top 5 predictions")

        for score, index in zip(
            scores.tolist(), indices.tolist()
        ):
            species = CLASS_NAMES[index]

            st.write(f"**{species}** — {score:.2%}")
            st.progress(float(score))

    except Exception as e:
        st.error(f"Prediction failed: {e}")
