import numpy as np
from io import BytesIO
from PIL import  Image
import pickle
import tensorflow as tf
import pandas as pd
import streamlit as st
def load_model():
    MODEl=tf.keras.models.load_model("../my_model")

    return MODEl

def load_image():
    uploaded_file = st.file_uploader(label='Pick an image to test')
    if uploaded_file is not None:
        image_data = uploaded_file.getvalue()

        st.image(image_data)
        y=np.array(Image.open(BytesIO(image_data)))
        image_batch = np.expand_dims(y, 0)/256
        image_batch=np.expand_dims(image_batch,3)
        image_batch=tf.convert_to_tensor(
    image_batch, dtype=tf.float32,
)
        return  image_batch
    else:
        return None




def predict(image,CLASS_NAMES,MODEl):
    prediction = MODEl.predict(image)
    predicted_class = CLASS_NAMES[np.argmax(prediction[0])]
    confidence = np.max(prediction[0])
    return {
        'class': predicted_class,
        'confidence': float(confidence)
    }

def main():
    st.title('NUMBER PREDICTION')
    model = load_model()
    categories = ["zero","one","two","three","four","five","six","seven","eight","nine"]
    image = load_image()
    result = st.button('Run on image')
    if result:
        st.write('Calculating results.....')
        type=predict(image, categories, model)
        st.write(type['class'])
        st.write(type['confidence'])

if __name__ == '__main__':
    main()