import streamlit as st
import tensorflow as tf
 
# set.set_option('deprecation.showfileUploaderEncoding', False)
# # @st.cache(allow_input_mutation=True)
# def load_model():
#     global model
#     model = load_model('VGG70-30_100epochs.h5')
#     return model
# model = load_model('VGG70-30_100epochs.h5')
from tensorflow import keras
model = keras.models.load_model(r'C:\Users\Umar\Desktop\Landslide Classification\Saved_Model\VGG70-30_100epochs.h5')

st.write("""
        # Landslide Classification
        """
    )

file = st.file_uploader("Please upload a landslide image", type=["jpg", "jpeg", "png"])
import cv2
from PIL import Image, ImageOps
import numpy as np

def import_and_predict(image_data, model):
    size = (224,224)
    image = ImageOps.fit(image_data, size, Image.ANTIALIAS)
    img = np.asarray(image)
    img_reshape = img[np.newaxis, ...]
    prediction = model.predict(img_reshape)
    return prediction

if file is None:
    st.text("Please upload an image file")
else:
    image = Image.open(file)
    st.image(image, use_column_width=True)
    predictions = import_and_predict(image, model)
    class_names = ['Debrisflow', 'Earthflow', 'Rockfall']
    string = "Prediction of this image is : "+class_names[np.argmax(predictions)]
    st.success(string)