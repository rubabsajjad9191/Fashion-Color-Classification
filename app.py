
import streamlit as st
import numpy as np
import joblib
from PIL import Image

st.title("Fashion Colour Classification")
st.write("Upload a clothing image to predict its colour.")

@st.cache_resource
def load_models():
    rf = joblib.load("random_forest_model (1).pkl")
    lr = joblib.load("logistic_regression_model.pkl")
    return rf, lr

rf_model, lr_model = load_models()

def extract_features(image):
    image = image.convert("RGB")
    features = []

    for channel in range(3):
        histogram = np.array(
            image.getchannel(channel).histogram()
        )
        histogram = histogram.reshape(8, 32).sum(axis=1)
        histogram = histogram / histogram.sum()
        features.extend(histogram)

    return np.array(features).reshape(1, -1)

uploaded_file = st.file_uploader(
    "Upload a clothing image",
    type=["jpg", "jpeg", "png", "webp"]
)

model_choice = st.selectbox(
    "Select AI model",
    ["Random Forest", "Logistic Regression"]
)

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="Your clothing image",
             use_container_width=True)

    if st.button("Predict Colour"):
        features = extract_features(image)

        if model_choice == "Random Forest":
            prediction = rf_model.predict(features)[0]
        else:
            prediction = lr_model.predict(features)[0]

        st.success(f"Predicted colour: {prediction.title()}")
