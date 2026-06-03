import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np

# ---------------------------------------------------
# PAGE CONFIG
# ---------------------------------------------------
st.set_page_config(
    page_title="PetVision AI",
    page_icon="🐾",
    layout="centered"
)

# ---------------------------------------------------
# CUSTOM CSS
# ---------------------------------------------------
st.markdown("""
<style>

/* Hide Streamlit Branding */
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}

/* Main Background */
.stApp {
    background: linear-gradient(
        135deg,
        #667eea 0%,
        #764ba2 50%,
        #6B73FF 100%
    );
}

/* Title */
.main-title {
    text-align: center;
    color: white;
    font-size: 4rem;
    font-weight: 800;
    margin-bottom: 0;
}

/* Subtitle */
.subtitle {
    text-align: center;
    color: #f3f3f3;
    font-size: 1.1rem;
    margin-bottom: 30px;
}

/* Glass Container */
.glass {
    background: rgba(255,255,255,0.15);
    backdrop-filter: blur(12px);
    border-radius: 25px;
    padding: 25px;
    border: 1px solid rgba(255,255,255,0.2);
    box-shadow: 0px 8px 32px rgba(0,0,0,0.25);
}

/* Prediction Card */
.prediction-card {
    background: white;
    border-radius: 20px;
    padding: 25px;
    text-align: center;
    box-shadow: 0px 10px 25px rgba(0,0,0,0.15);
    margin-top: 20px;
}

/* Upload Area */
[data-testid="stFileUploader"] {
    background: rgba(255,255,255,0.12);
    border: 2px dashed rgba(255,255,255,0.4);
    border-radius: 20px;
    padding: 20px;
}

/* White Text */
.white-text {
    color: white;
    text-align: center;
}

/* Confidence Text */
.confidence-text {
    color: white;
    text-align: center;
    font-size: 1.1rem;
    font-weight: bold;
}

</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------
# LOAD MODEL
# ---------------------------------------------------
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("cat_dog_cnn.h5")

model = load_model()

# ---------------------------------------------------
# HEADER
# ---------------------------------------------------
st.markdown(
    """
    <div class="main-title">
        🐾 PetVision AI
    </div>
    """,
    unsafe_allow_html=True
)

st.markdown(
    """
    <div class="subtitle">
        Deep Learning Powered Cat & Dog Recognition
    </div>
    """,
    unsafe_allow_html=True
)

# ---------------------------------------------------
# UPLOAD IMAGE
# ---------------------------------------------------
uploaded_file = st.file_uploader(
    "📤 Upload a Cat or Dog Image",
    type=["jpg", "jpeg", "png"]
)

# ---------------------------------------------------
# PREDICTION
# ---------------------------------------------------
if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")

    st.markdown('<div class="glass">', unsafe_allow_html=True)

    st.image(
        image,
        use_container_width=True
    )

    st.markdown('</div>', unsafe_allow_html=True)

    # Preprocess
    img = image.resize((128, 128))
    img = np.array(img) / 255.0
    img = np.expand_dims(img, axis=0)

    # Predict
    with st.spinner("🔍 AI is analyzing the image..."):
        prediction = model.predict(img, verbose=0)

    probability = float(prediction[0][0])

    # Dog
    if probability > 0.5:

        confidence = probability * 100

        st.markdown(
            f"""
            <div class="prediction-card">
                <h1>🐶 DOG</h1>
                <h3>Confidence: {confidence:.2f}%</h3>
            </div>
            """,
            unsafe_allow_html=True
        )

    # Cat
    else:

        confidence = (1 - probability) * 100

        st.markdown(
            f"""
            <div class="prediction-card">
                <h1>🐱 CAT</h1>
                <h3>Confidence: {confidence:.2f}%</h3>
            </div>
            """,
            unsafe_allow_html=True
        )

    # Confidence Bar
    st.markdown(
        """
        <div class="confidence-text">
            Prediction Confidence
        </div>
        """,
        unsafe_allow_html=True
    )

    st.progress(confidence / 100)

    st.markdown(
        f"""
        <div class="white-text">
            Confidence Score: {confidence:.2f}%
        </div>
        """,
        unsafe_allow_html=True
    )

# ---------------------------------------------------
# FOOTER
# ---------------------------------------------------
st.markdown("<br><br>", unsafe_allow_html=True)

st.markdown(
    """
    <div style="text-align:center;color:white;opacity:0.8;">
        Powered by Convolutional Neural Networks (CNN)
    </div>
    """,
    unsafe_allow_html=True
)
