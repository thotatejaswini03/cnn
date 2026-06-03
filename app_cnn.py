import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np

# -------------------------------------------------
# PAGE CONFIG
# -------------------------------------------------
st.set_page_config(
    page_title="CNN Classification App",
    page_icon="🐾",
    layout="wide"
)

# -------------------------------------------------
# CUSTOM CSS
# -------------------------------------------------
st.markdown("""
<style>

/* Background */
.stApp{
    background: linear-gradient(
        135deg,
        #F8F7FF 0%,
        #EEF4FF 50%,
        #FFF7F8 100%
    );
}

/* Hide Streamlit */
#MainMenu {visibility:hidden;}
footer {visibility:hidden;}
header {visibility:hidden;}

/* Main Title */
.main-title{
    font-size:48px;
    font-weight:700;
    color:#111827;
    margin-bottom:10px;
}

/* Top Line */
.hr-line{
    height:4px;
    background:#7C3AED;
    border:none;
    border-radius:10px;
    margin-bottom:20px;
}

/* Section Headings */
.section-title{
    font-size:32px;
    font-weight:700;
    color:#111827;
    margin-bottom:15px;
}

/* Cards */
.card{
    background:white;
    padding:20px;
    border-radius:16px;
    box-shadow:0 4px 15px rgba(0,0,0,0.08);
}

/* Prediction Card */
.prediction-card{
    background:white;
    padding:25px;
    border-radius:16px;
    text-align:center;
    box-shadow:0 4px 15px rgba(0,0,0,0.08);
}

/* Prediction Result */
.big-result{
    font-size:50px;
    font-weight:700;
    color:#111827;
}

.confidence{
    font-size:28px;
    color:#059669;
    font-weight:700;
}

/* Labels */
.label-text{
    color:#374151;
    font-weight:600;
    font-size:18px;
}

/* Probability Heading */
.breakdown-title{
    color:#111827;
    font-size:36px;
    font-weight:700;
    margin-top:20px;
}

/* Probability Labels */
.prob-text{
    color:#111827;
    font-size:20px;
    font-weight:600;
    margin-bottom:5px;
}

/* Sidebar */
section[data-testid="stSidebar"]{
    background-color:#F9FAFB;
}

</style>
""", unsafe_allow_html=True)

# -------------------------------------------------
# LOAD MODEL
# -------------------------------------------------
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("cat_dog_cnn.h5")

model = load_model()

# -------------------------------------------------
# TITLE
# -------------------------------------------------
st.markdown(
    '<div class="main-title">CNN Classification App</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="hr-line"></div>',
    unsafe_allow_html=True
)

# -------------------------------------------------
# SIDEBAR
# -------------------------------------------------
st.sidebar.title("App Navigation")

st.sidebar.markdown("### Supported Classes")

st.sidebar.markdown("🐱 Cat")
st.sidebar.markdown("🐶 Dog")

st.sidebar.markdown("---")

st.sidebar.info(
    "Upload an image and the CNN model will classify it as Cat or Dog."
)

# -------------------------------------------------
# MAIN LAYOUT
# -------------------------------------------------
left_col, right_col = st.columns([1.2,1])

# -------------------------------------------------
# LEFT COLUMN
# -------------------------------------------------
with left_col:

    st.markdown(
        '<div class="section-title">📥 Input Source</div>',
        unsafe_allow_html=True
    )

    uploaded_file = st.file_uploader(
        "Upload an image",
        type=["jpg","jpeg","png"]
    )

    if uploaded_file is not None:

        image = Image.open(uploaded_file).convert("RGB")

        st.image(
            image,
            use_container_width=True
        )

# -------------------------------------------------
# RIGHT COLUMN
# -------------------------------------------------
with right_col:

    st.markdown(
        '<div class="section-title">📊 Classification Output</div>',
        unsafe_allow_html=True
    )

    if uploaded_file is not None:

        img = image.resize((128,128))
        img = np.array(img)/255.0
        img = np.expand_dims(img, axis=0)

        prediction = model.predict(
            img,
            verbose=0
        )

        dog_prob = float(prediction[0][0])
        cat_prob = 1 - dog_prob

        if dog_prob > 0.5:
            predicted_class = "DOG 🐶"
            confidence = dog_prob * 100
        else:
            predicted_class = "CAT 🐱"
            confidence = cat_prob * 100

        st.markdown(
            f"""
            <div class="prediction-card">

                <p style="
                color:#6B7280;
                font-size:14px;
                letter-spacing:1px;
                ">
                TOP PREDICTED CATEGORY
                </p>

                <div class="big-result">
                {predicted_class}
                </div>

                <br>

                <div class="confidence">
                Confidence Match: {confidence:.2f}%
                </div>

            </div>
            """,
            unsafe_allow_html=True
        )

        st.markdown(
            '<div class="breakdown-title">Probability Breakdown</div>',
            unsafe_allow_html=True
        )

        st.markdown(
            '<div class="prob-text">🐱 Cat</div>',
            unsafe_allow_html=True
        )
        st.progress(cat_prob)
        st.markdown(
            f'<div class="label-text">{cat_prob*100:.2f}%</div>',
            unsafe_allow_html=True
        )

        st.markdown("<br>", unsafe_allow_html=True)

        st.markdown(
            '<div class="prob-text">🐶 Dog</div>',
            unsafe_allow_html=True
        )
        st.progress(dog_prob)
        st.markdown(
            f'<div class="label-text">{dog_prob*100:.2f}%</div>',
            unsafe_allow_html=True
        )

    else:

        st.info("Upload an image to see prediction results.")

# -------------------------------------------------
# FOOTER
# -------------------------------------------------
st.markdown("<br><br>", unsafe_allow_html=True)

st.markdown(
    """
    <div style="
    text-align:center;
    color:#6B7280;
    font-size:15px;
    ">
    Powered by Convolutional Neural Networks (CNN)
    </div>
    """,
    unsafe_allow_html=True
)
