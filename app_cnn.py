import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np

# -----------------------------
# Page Configuration
# -----------------------------
st.set_page_config(
    page_title="Cat vs Dog Classifier",
    page_icon="🐾",
    layout="centered"
)

# -----------------------------
# Custom CSS
# -----------------------------
st.markdown("""
<style>

/* Background */
.stApp {
    background: linear-gradient(135deg, #1e3c72, #2a5298);
}

/* Main Title */
.main-title {
    text-align: center;
    font-size: 3rem;
    font-weight: 800;
    color: white;
    margin-top: 10px;
    margin-bottom: 10px;
}

/* Subtitle */
.subtitle {
    text-align: center;
    color: #e8e8e8;
    font-size: 1.1rem;
    margin-bottom: 30px;
}

/* Upload Box */
[data-testid="stFileUploader"] {
    background-color: rgba(255,255,255,0.08);
    padding: 20px;
    border-radius: 15px;
    border: 2px dashed white;
}

/* Prediction Card */
.prediction-card {
    background: white;
    padding: 20px;
    border-radius: 15px;
    text-align: center;
    font-size: 26px;
    font-weight: bold;
    color: #222;
    box-shadow: 0px 4px 20px rgba(0,0,0,0.25);
    margin-top: 15px;
    margin-bottom: 15px;
}

/* Section Heading */
.section-heading {
    color: white;
    font-size: 1.2rem;
    font-weight: bold;
    margin-top: 20px;
}

/* White text */
.white-text {
    color: white;
    font-size: 1rem;
}

/* Hide Streamlit Branding */
#MainMenu {
    visibility: hidden;
}

footer {
    visibility: hidden;
}

header {
    visibility: hidden;
}

</style>
""", unsafe_allow_html=True)

# -----------------------------
# Load Model
# -----------------------------
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("cat_dog_cnn.h5")

model = load_model()

# -----------------------------
# Header
# -----------------------------
st.markdown(
    '<div class="main-title">🐾 Cat vs Dog Classifier</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">Upload an image and let the CNN predict whether it is a Cat or Dog.</div>',
    unsafe_allow_html=True
)

# -----------------------------
# Upload Image
# -----------------------------
uploaded_file = st.file_uploader(
    "📤 Upload an Image",
    type=["jpg", "jpeg", "png"]
)

# -----------------------------
# Prediction
# -----------------------------
if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")

    st.image(
        image,
        caption="Uploaded Image",
        use_container_width=True
    )

    # Preprocess Image
    img = image.resize((128, 128))
    img = np.array(img) / 255.0
    img = np.expand_dims(img, axis=0)

    with st.spinner("🔍 Analyzing Image..."):
        prediction = model.predict(img, verbose=0)

    probability = float(prediction[0][0])

    st.markdown(
        '<div class="section-heading">Prediction Result</div>',
        unsafe_allow_html=True
    )

    if probability > 0.5:

        confidence = probability * 100

        st.markdown(
            f"""
            <div class="prediction-card">
                🐶 DOG<br>
                Confidence: {confidence:.2f}%
            </div>
            """,
            unsafe_allow_html=True
        )

    else:

        confidence = (1 - probability) * 100

        st.markdown(
            f"""
            <div class="prediction-card">
                🐱 CAT<br>
                Confidence: {confidence:.2f}%
            </div>
            """,
            unsafe_allow_html=True
        )

    st.markdown(
        '<div class="section-heading">Prediction Score</div>',
        unsafe_allow_html=True
    )

    st.progress(
        min(max(probability, 0.0), 1.0)
    )

    st.markdown(
        f'<p class="white-text">🐶 Dog Probability: <b>{probability:.4f}</b></p>',
        unsafe_allow_html=True
    )

    st.markdown(
        f'<p class="white-text">🐱 Cat Probability: <b>{1 - probability:.4f}</b></p>',
        unsafe_allow_html=True
    )
