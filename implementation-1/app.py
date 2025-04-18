import streamlit as st
import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
from PIL import Image
from io import BytesIO

st.set_page_config(page_title="Neural Style Transfer", layout="wide")

@st.cache_resource
def load_model():
    return hub.load('https://tfhub.dev/google/magenta/arbitrary-image-stylization-v1-256/2')

stylize_model = load_model()

def load_image(image_file, image_size=(512, 512)):
    try:
        img = Image.open(image_file).convert('RGB')
        img = np.array(img, dtype=np.float32) / 255.0
        img = tf.image.resize(img, image_size, preserve_aspect_ratio=True)
        return img[tf.newaxis, :]
    except Exception as e:
        st.error(f"Error loading image: {e}")
        return None

def export_image(tf_img):
    img = np.squeeze(tf_img, axis=0) * 255
    pil_img = Image.fromarray(img.astype(np.uint8))
    buffer = BytesIO()
    pil_img.save(buffer, format="PNG")
    return buffer.getvalue()

def main():
    st.sidebar.title("Style Transfer")
    st.sidebar.write("Blend content with style!")

    content_file = st.sidebar.file_uploader("Content Image", ["jpg", "jpeg", "png"], help="Image to stylize")
    style_file = st.sidebar.file_uploader("Style Image", ["jpg", "jpeg", "png"], help="Style source")

    col1, col2, col3 = st.columns(3)

    content_img = load_image(content_file) if content_file else None
    if content_file:
        col1.header("Content")
        col1.image(content_file, use_container_width=True)

    style_img = load_image(style_file, (256, 256)) if style_file else None
    if style_file:
        col2.header("Style")
        col2.image(style_file, use_container_width=True)

    if st.sidebar.button("Stylize"):
        if content_img is not None and style_img is not None:
            with st.spinner("Generating..."):
                stylized_img = stylize_model(content_img, style_img)[0].numpy()
                col3.header("Result")
                col3.image(stylized_img, use_container_width=True)
                col3.download_button(
                    "Download",
                    export_image(stylized_img),
                    "stylized_image.png",
                    "image/png"
                )
        else:
            st.sidebar.error("Upload both images.")

if __name__ == "__main__":
    main()