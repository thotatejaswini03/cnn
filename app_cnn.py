import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np

# Page Configuration
st.set_page_config(
    page_title="Cat vs Dog Classifier",
    page_icon="🐾",
    layout="centered"
)

# Load Model
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("cat_dog_cnn.h5")

model = load_model()

# Title
st.title("🐾 Cat vs Dog Image Classifier")
st.write("Upload an image and the CNN model will predict whether it is a Cat or a Dog.")

# File Upload
uploaded_file = st.file_uploader(
    "Choose an image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:

    # Display Image
    image = Image.open(uploaded_file).convert("RGB")

    st.image(
        image,
        caption="Uploaded Image",
        use_container_width=True
    )

    # Preprocessing
    img = image.resize((128, 128))
    img = np.array(img) / 255.0
    img = np.expand_dims(img, axis=0)

    # Prediction
    with st.spinner("Analyzing image..."):
        prediction = model.predict(img, verbose=0)

    probability = float(prediction[0][0])

    st.subheader("Prediction Result")

    if probability > 0.5:
        confidence = probability * 100

        st.success("🐶 Dog")
        st.metric(
            label="Confidence",
            value=f"{confidence:.2f}%"
        )

    else:
        confidence = (1 - probability) * 100

        st.success("🐱 Cat")
        st.metric(
            label="Confidence",
            value=f"{confidence:.2f}%"
        )

    # Probability Bar
    st.subheader("Prediction Score")

    st.progress(
        min(max(probability, 0.0), 1.0)
    )

    st.write(f"Dog Probability: **{probability:.4f}**")
    st.write(f"Cat Probability: **{1 - probability:.4f}**")

# Sidebar
st.sidebar.header("About")
st.sidebar.write(
    """
    This application uses a Convolutional Neural Network (CNN)
    trained on Cat and Dog images.

    Steps:
    1. Upload an image
    2. Image is resized to 128×128
    3. CNN extracts features
    4. Model predicts Cat or Dog
    """
)
