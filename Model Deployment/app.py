# Import packages
import streamlit as st
import numpy as np
from PIL import Image
import cv2
from tensorflow.keras.models import load_model

######################################### CLASS NAMES #########################################

# Define the class labels
class_names = [
    "Speed limit (20km/h)", "Speed limit (30km/h)", "Speed limit (50km/h)", 
    "Speed limit (60km/h)", "Speed limit (70km/h)", "Speed limit (80km/h)", 
    "End of speed limit (80km/h)", "Speed limit (100km/h)", "Speed limit (120km/h)", 
    "No passing", "No passing for vehicles over 3.5 metric tons", 
    "Right-of-way at the next intersection", "Priority road", "Yield", "Stop", 
    "No vehicles", "Vehicles over 3.5 metric tons prohibited", "No entry", 
    "General caution", "Dangerous curve to the left", "Dangerous curve to the right", 
    "Double curve", "Bumpy road", "Slippery road", "Road narrows on the right", 
    "Road work", "Traffic signals", "Pedestrians", "Children crossing", 
    "Bicycles crossing", "Beware of ice/snow", "Wild animals crossing", 
    "End of all speed and passing limits", "Turn right ahead", "Turn left ahead", 
    "Ahead only", "Go straight or right", "Go straight or left", "Keep right", 
    "Keep left", "Roundabout mandatory", "End of no passing", 
    "End of no passing by vehicles over 3.5 metric tons"
]

# Sidebar to display class labels
st.sidebar.title('What images can you upload?')
for idx, name in enumerate(class_names):
    st.sidebar.write(f'{idx} - {name}')

######################################### MODEL DEPLOYMENT #########################################

# Load the model
model = load_model('classifier.h5')

# Function for preprocessing the images
def preprocess(img):
  img = np.array(img)
  img = cv2.resize(img, (32, 32))
  img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
  img = cv2.equalizeHist(img)
  img = np.divide(img, 255)
  return img

# Function to predict on new images
def predict(input_data):
  predictions = model.predict(np.array([input_data]))
  return predictions

######################################### USER INTERFACE #########################################

# Streamlit Interface
st.title('Traffic Sign Classifier')
st.header('How this app work?')
st.write('This application is designed to classify different traffic signs using computer vision techniques, ' 
          'mainly through the implementation of convolutional neural networks. ' 
          'The main objective of the application is to improve road safety by ' 
          'providing real-time recognition of traffic signs, allowing for better ' 
          'decision-making in autonomous vehicles or driver-assistance systems. ')
st.write('Feel free to upload an image using the button below, and get your prediction in real-time!')

######################################### MAKING PREDICTION #########################################

uploaded_file = st.file_uploader('Upload an image', type=['jpg', 'jpeg', 'png'])
col1, col2 = st.columns(2)

if uploaded_file is not None:
  image = Image.open(uploaded_file)
  with col1:
    st.image(image, caption='Uploaded Image', width=150)

  # Preprocess the image
  input_data  = preprocess(image)
  input_data = np.expand_dims(input_data, axis=-1)

  # Get predictions
  prediction = predict(input_data)
  predicted_class = np.argmax(prediction, axis=1)[0]  # Get the class index as a scalar
  
  # Display prediction result
  with col2:
    st.subheader('Prediction Result')
    st.write(f'Predicted Class: {predicted_class} - {class_names[predicted_class]}')