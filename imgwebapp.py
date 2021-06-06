import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import tensorflow as tf
import numpy as np
import streamlit as st
from PIL import Image
import requests
from io import BytesIO

st.set_option('deprecation.showfileUploaderEncoding', False)
st.title("Cat Family Classifier")
st.text("Provide URL of bean Image for image classification")

@st.cache(allow_output_mutation=True)
def load_model():

  #model = tf.keras.models.load_model('/Users/s.sriaravind/Desktop/Dev/Data_Science/Projects/Cats_Family_Detection/models')
  model = tf.keras.models.load_model('/Users/s.sriaravind/Desktop/Dev/Data_Science/Projects/Cats_Family_Detection/models')
  return model

with st.spinner('Loading Model Into Memory....'):
  model = load_model()

classes=['Cheetah', 'Jaguar', 'Tiger']

def decode_img(image):
  img = tf.image.decode_jpeg(image, channels=3)
  img = tf.image.resize(img,[224,224])
  return np.expand_dims(img, axis=0)

path = st.text_input('Enter Image URL to Classify.. ','https://content.jwplatform.com/thumbs/h5Ko1nUT-720.jpg')
if path is not None:
    content = requests.get(path).content

    st.write("Predicted Class :")
    with st.spinner('classifying.....'):
      label =np.argmax(model.predict(decode_img(content)),axis=1)
      st.write(classes[label[0]])
    st.write("")
    image = Image.open(BytesIO(content))
    st.image(image, caption='Classifying Cat Family', use_column_width=True)