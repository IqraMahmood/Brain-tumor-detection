import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np

# Load pre-trained model
model = load_model('inceptionv3_best.h5')

# Define class labels
labels = ['No Tumor', 'Tumor']

# Set app title and description
st.title("Brain Tumor Detection App")
st.write("Upload an MRI image and the app will predict whether it contains a brain tumor.")

# Define function to preprocess image
def preprocess_image(img):
  img = img.resize((224,224))
  img_array = np.array(img)
  img_array = img_array.astype('float32') / 255.0
  img_array = np.expand_dims(img_array, axis=0)
  return img_array

# Define function to make prediction
def make_prediction(img):
  # Preprocess image
  img_array = preprocess_image(img)
  # Make prediction
  prediction = model.predict(img_array)
  # Get predicted class label
  predicted_label = labels[np.argmax(prediction)]
  # Get predicted class probability
  predicted_proba = np.max(prediction)
  return predicted_label, predicted_proba

# Allow user to upload image
uploaded_file = st.file_uploader("Choose an MRI image", type=['jpg','jpeg','png'])

# Make prediction if image is uploaded
if uploaded_file is not None:
  # Load and display image
  img = Image.open(uploaded_file)
  st.image(img, caption='Uploaded MRI image', use_column_width=True)
  # Make prediction
  predicted_label, predicted_proba = make_prediction(img)
  # Display predicted label and probability
  st.write('Prediction:')
  st.write(f'- {predicted_label} ({predicted_proba:.2f})')
