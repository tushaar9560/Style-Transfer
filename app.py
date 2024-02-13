import streamlit as st
import sys
from PIL import Image
from src.logger import logging
from src.exception import CustomException
# from src.pipelines.style import StyleTransfer
import main
st.title("Style Transfer App")

img = st.sidebar.selectbox(
    'Select Image',
    ("amber.jpg", 'cat.png')
)

style_name = st.sidebar.selectbox(
    "Select Style",
    ('candy', 'mosaic', 'rain_princess', 'udnie')
)

model_format = st.sidebar.selectbox(
    'Select Model Format',
    ('.pth', '.onnx')
)

model_path = "artifacts/models/" + style_name + model_format
input_image_path = f"artifacts/inputs/{img}"
output_image_path = f"artifacts/outputs/{style_name}-{img}"

st.write("Source image:")
image = Image.open(input_image_path)
st.image(image, width = 400)

clicked = st.button('Stylize')

if clicked:
    try:
        model = main.load_model(model_path)
        main.stylize(model, input_image_path, output_image_path)
        st.write("Output image:")
        image = Image.open(output_image_path)
        st.image(image, width = 400)
    except Exception as e:
        logging.info(e)
        raise CustomException(e,sys)
