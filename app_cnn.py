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

.stApp{
    background-color:#f5f7fb;
}

/* Hide Streamlit menu */
#MainMenu {visibility:hidden;}
footer {visibility:hidden;}
header {visibility:hidden;}

/* Main Title */
.main-title{
    font-size:48px;
    font-weight:700;
    color:#5B4BDB;
    margin-bottom:10px;
}

.hr-line{
    height:4px;
    background:#D946EF;
    border:none;
    border-radius:10px;
    margin-bottom:20px;
}

/* Section Headings */
.section-title{
    font-size:28px;
    font-weight:700;
    color:#5B4BDB;
    margin-bottom:10px;
}

/* Cards */
.card{
    background:white;
    padding:20px;
    border-radius:15px;
    box-shadow:0 2px 12px rgba(0,0,0,0.08);
}

/* Prediction Card */
.prediction-card{
    background:white;
    padding:25px;
    border-radius:15px;
    text-align:center;
    box-shadow:0 2px 12px rgba(0,0,0,0.08);
}

.big-result{
    font-size:40px;
    font-weight:bold;
    color:#5B4BDB;
}

.confidence{
    font-size:24px;
    color:#10B981;
    font-weight:600;
}

/* Sidebar */
.css-1d391kg{
    background:white;
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

st.sidebar.success("🐱 Cat")
st.sidebar.success("🐶 Dog")

st.sidebar.markdown("---")

st.sidebar.info(
    """
    Upload an image and the CNN model will
    classify it as Cat or Dog.
    """
)

# -------------------------------------------------
# TWO COLUMN LAYOUT
# -------------------------------------------------
left_col, right_col = st.columns([1.2, 1])

# =================================================
# LEFT SIDE
# =================================================
with left_col:

    st.markdown(
        '<div class="section-title">📥 Input Source</div>',
        unsafe_allow_html=True
    )

    uploaded_file = st.file_uploader(
        "Upload an image",
        type=["jpg", "jpeg", "png"]
    )

    if uploaded_file is not None:

        image = Image.open(uploaded_file).convert("RGB")

        st.image(
            image,
            caption="Uploaded Image",
            use_container_width=True
        )

# =================================================
# RIGHT SIDE
# =================================================
with right_col:

    st.markdown(
        '<div class="section-title">📊 Classification Output</div>',
        unsafe_allow_html=True
    )

    if uploaded_file is not None:

        # Preprocess image
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

            <p style="color:gray;
                      font-size:14px;
                      margin-bottom:10px;">
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

        st.markdown("<br>", unsafe_allow_html=True)

        st.subheader("Probability Breakdown")

        st.write("🐱 Cat")
        st.progress(cat_prob)
        st.write(f"{cat_prob*100:.2f}%")

        st.write("🐶 Dog")
        st.progress(dog_prob)
        st.write(f"{dog_prob*100:.2f}%")

    else:

        st.info(
            "Upload an image to see prediction results."
        )

# -------------------------------------------------
# FOOTER
# -------------------------------------------------
st.markdown("<br><br>", unsafe_allow_html=True)

