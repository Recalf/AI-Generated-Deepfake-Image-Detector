import streamlit as st
import torch
import torch.nn.functional as F
from PIL import Image

from model.convnext import build_model
from transforms import test_transforms
from streamlit_paste_button import paste_image_button as pbutton

st.set_page_config(page_title="Toufik AI Gen & DeepFake Images Detector", layout="centered")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

@st.cache_resource # for persistance when rerun
def load_model():
    model = build_model()
    ckpt = torch.load("checkpoints/checkpoint_phase2.pth", map_location=device)
    model.load_state_dict(ckpt["model"])
    model.to(device)
    model.eval()
    return model

model = load_model()
transform = test_transforms()

st.title("Toufik's AI Generated & DeepFake Images Detector")
st.write("Upload or paste an image in the left sidebar and the model will predict **Real vs Fake**.")


# sidebar
st.sidebar.header("Input")
uploaded_file = st.sidebar.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
with st.sidebar:
    paste_result = pbutton("Paste Image")

st.sidebar.header("Settings")
threshold = st.sidebar.slider("Fake threshold", 0.0, 1.0, 0.50, 0.01)
st.sidebar.caption("Higher = stricter (more fake confidence to label Fake)")

image = None
if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
elif paste_result.image_data is not None:
    image = paste_result.image_data.convert("RGB") 

# inference
if image is not None:
    preview = image.copy()
    preview.thumbnail((720, 420)) 
    st.image(preview, caption="Input Image")


    img_tensor = transform(image).unsqueeze(0).to(device)

    with torch.inference_mode():
        logits = model(img_tensor)
        probs = F.softmax(logits, dim=1)[0]

    fake_prob = probs[1].item()
    pred_index = 1 if fake_prob > threshold else 0
    confidence = fake_prob if pred_index == 1 else (1 - fake_prob)

    classes = ["Real", "Fake"] # 0/1

    real_prob = 1 - fake_prob

    label = classes[pred_index]
    if label == "Fake":
        st.error(f"**{label}**")
    else:
        st.success(f"**{label}**")

    st.metric("Confidence", f"{confidence*100:.2f}%")
    st.progress(confidence)

    st.caption("Model probabilities")
    st.write(f"Real: **{real_prob*100:.2f}%**")
    st.write(f"Fake: **{fake_prob*100:.2f}%**")