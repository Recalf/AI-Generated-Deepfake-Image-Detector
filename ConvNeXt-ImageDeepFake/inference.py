import streamlit as st
import torch
import torch.nn.functional as F
from PIL import Image

from model.convnext import build_model
from transforms import test_transforms
from streamlit_paste_button import paste_image_button as pbutton


# ------------------
# Setup
# ------------------
st.set_page_config(page_title="DeepFake Detector", layout="centered")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

@st.cache_resource
def load_model():
    model = build_model()
    ckpt = torch.load(r"D:\Pytorch\Projects\ConvNeXt-ImageDeepFake\train\checkpoint1_phase3.pth", map_location=device)
    model.load_state_dict(ckpt["model"])
    model.to(device)
    model.eval()
    return model

model = load_model()
transform = test_transforms()

# ------------------
# UI
# ------------------
st.title("DeepFake Image Detector")
st.write("Upload an image and the model will predict **Real vs Fake**.")



uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
paste_result = pbutton("📋 Paste Image")

image = None
if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
elif paste_result.image_data is not None:
    image = paste_result.image_data.convert("RGB") 

# ------------------
# Inference
# ------------------
if image is not None:
    st.image(image, caption="Input Image", use_container_width=True)

    img_tensor = transform(image).unsqueeze(0).to(device)

    with torch.inference_mode():
        logits = model(img_tensor)
        probs = F.softmax(logits, dim=1)[0]

    pred_idx = probs.argmax().item()
    confidence = probs[pred_idx].item()

    classes = ["Real", "Fake"]  # change if your label order differs

    st.markdown("---")
    st.subheader("Prediction")
    st.write(f"**Class:** {classes[pred_idx]}")
    st.write(f"**Confidence:** {confidence * 100:.2f}%")
    st.progress(confidence)