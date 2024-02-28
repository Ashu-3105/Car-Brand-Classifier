import streamlit as st
import cv2
from tensorflow.keras.models import load_model
import numpy as np
from tensorflow.keras.preprocessing import image
from PIL import Image
model=load_model('Model_resnet50_E42.h5')
st.sidebar.title("Dashboard")
app_mode = st.sidebar.selectbox("Select Page",["Webcam","Image File"])

def model_predict(img_path, model):
    img = Image.open(img_path)
    img = img.resize((224, 224), resample=Image.BILINEAR)


    # Preprocessing the image
    x = image.img_to_array(img)
    # x = np.true_divide(x, 255)
    ## Scaling
    x = x / 255
    x = np.expand_dims(x, axis=0)

    preds = model.predict(x)
    preds = np.argmax(preds, axis=1)
    if preds == 0:
        preds = "Audi"
    elif preds == 1:
        preds = "Lamborghini"
    else:
        preds = "Mercedes"

    return preds


if app_mode=='Webcam':
    img_file_buffer = st.camera_input("Take a picture")

    if img_file_buffer is not None:
        result = model_predict(img_file_buffer, model)
        st.text(f'car brand is {result}')
elif app_mode=='Image File':
    img_file_buffer = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

    if img_file_buffer is not None:
        result = model_predict(img_file_buffer, model)
        st.text(f'car brand is {result}')


