import streamlit as st
import numpy as np
import cv2
from PIL import Image
from tensorflow.keras.models import load_model

# ---------- Page config ----------
st.set_page_config(
    page_title="Handwritten Digit Recognition",
    page_icon="✍️",
    layout="centered",
)

# ---------- Sidebar ----------
st.sidebar.title("About")
st.sidebar.info(
    """
    **Handwritten Digit Recognition**

    - **Model**: Convolutional Neural Network (trained on MNIST)  
    - **Author**: PVB ADITHYA  
    - **Predicts**: digits 0‑9 from 28×28 grayscale images

    **How to use**
    1. Click *Browse files* and upload an image (JPG/PNG).  
    2. The app preprocesses the image and displays the predicted digit.
    """
)

st.sidebar.markdown("---")
st.sidebar.write("GitHub Repo: [PVBA - HDR](https://github.com/pvba-py/Handwritten-Digit-Recognition/)")

# ---------- Main title ----------
st.title("🧠 Handwritten Digit Recognition")
st.caption("Upload an image of a handwritten digit; the CNN model will predict the number.")

# ---------- File uploader ----------
uploaded_file = st.file_uploader(
    "Choose an image …",
    type=["jpg", "jpeg", "png"],
    accept_multiple_files=False,
    help="Upload a clear 28×28 grayscale (or any resolution) digit on a plain background",
)

# ---------- Helper: preprocess ----------
def preprocess_image(img_pil: Image.Image) -> np.ndarray:
    """Convert PIL image to normalized 28×28 NumPy array (shape = 1×28×28×1)."""
    img = img_pil.convert("L")            # to grayscale
    img = img.resize((28, 28))            # 28×28
    img = np.array(img)
    img = cv2.bitwise_not(img)            # make white digits black & vice‑versa (optional)
    img = img / 255.0                     # normalize to [0,1]
    img = img.reshape(1, 28, 28, 1)       # add batch & channel dims
    return img

# ---------- Prediction section ----------
if uploaded_file:
    col_left, col_mid, col_right = st.columns([1, 2, 1])

    with col_mid:
        # Display uploaded image
        img_display = Image.open(uploaded_file).convert("L")
        st.image(img_display, caption="Uploaded Image", width=200)

        # Preprocess & predict
        with st.spinner("Predicting…"):
            x = preprocess_image(img_display)
            model = load_model("cnn_trained_model.h5")
            pred = model.predict(x)
            digit = np.argmax(pred)

        st.markdown(
            f"<h2 style='text-align:center; color:#4CAF50;'>Predicted Digit: {digit}</h2>",
            unsafe_allow_html=True,
        )
else:
    st.info("👈 Upload a digit image from the sidebar or above to get started.")

# ---------- Footer ----------
st.markdown("---")
st.markdown(
    "<div style='text-align:center;'>Made with ❤️ using Streamlit & TensorFlow</div>",
    unsafe_allow_html=True,
)
