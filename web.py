import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import io
def model_prediction(test_image):
    model= tf.keras.models.load_model("trained_plant_disease_model.keras")
    image= tf.keras.preprocessing.image.load_img(test_image,target_size=(128,128))
    input_arr = tf.keras.preprocessing.image.img_to_array(image)
    input_arr=np.array([input_arr])
    predictions= model.predict(input_arr)
    return np.argmax(predictions)
st.sidebar.title("Plant Disease system for Sustainable Agriculture")
app_mode = st.sidebar.selectbox('select page',['Home','Disease Recognition'])

from PIL import Image
img= Image.open('Diseases.png')
st.image(img)

if(app_mode=='Home'):
    st.markdown("<h1 style='text-align: center;'>Plant Disease Detection System for Sustainable Agriculture", unsafe_allow_html=True)

elif(app_mode=='Disease Recognition'):
    st.header('Plant Disease Detection System For Sustainable Agriculture')


test_image= st.file_uploader('Choose an image:')
if st.button('Show Image'):
    if test_image is not None:
        image = Image.open(test_image)
        st.image(image, use_container_width=True)
    else:
        st.warning("Please upload an image first.")

if st.button('Predict'):
    if test_image is not None:
        st.snow()
        st.write('Our Prediction')

        result_index = model_prediction(test_image)

        class_name = [
            'Potato___Early_blight',
            'Potato___Late_blight',
            'Potato___healthy'
        ]

        st.success(
            f"Model predicts: {class_name[result_index]}"
        )
    else:
        st.warning("Please upload an image first.")
