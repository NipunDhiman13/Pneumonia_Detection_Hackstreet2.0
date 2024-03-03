import streamlit as st
import cv2
import numpy as np
import tensorflow as tf
from keras.models import load_model

# Load the trained model
loaded_model = load_model('Pneumonia3.h5')

# Function to preprocess the custom image
def preprocess_image(image):
    resized_img = cv2.resize(image, (224, 224))/255
    preprocessed_img = resized_img.reshape(1, 224, 224, 1) 
    return preprocessed_img

# Streamlit UI
background_image_url = "your_image_url.jpg"  # Replace with your image URL

# Define the custom CSS styles
custom_styles = f"""
    body {{
        background-image: url('https://cdn.wallpapersafari.com/85/53/guDl3n.png');
        background-size: cover;
        background-repeat: no-repeat;
        background-attachment: fixed;
    }}
"""

# Apply the custom styles to the Streamlit app
st.markdown(f'<style>{custom_styles}</style>', unsafe_allow_html=True)
st.title("Pneumonia Prediction App")

# Upload custom image through Streamlit UI
uploaded_file = st.file_uploader("Choose a custom image", type=["jpg", "png"])

if uploaded_file is not None:
    # Read the uploaded image
    file_bytes = uploaded_file.read()

    # Process the image
    custom_img = cv2.imdecode(np.fromstring(file_bytes, np.uint8), cv2.IMREAD_GRAYSCALE)
    preprocessed_custom_img = preprocess_image(custom_img)
    prediction = loaded_model.predict(preprocessed_custom_img)
    score = tf.nn.softmax(prediction[0])

    # Display results
    st.image(custom_img, caption="Uploaded Image", use_column_width=True)
    
    if uploaded_file.type == "image/png":
        st.warning("Invalid X-RAY - Need a Chest X-RAY to analyze.")
    else:
        st.write("Results:")
        st.write(f"Normal: {score[0]:.2%}")
        st.write(f"Pneumonia: {score[1]:.2%}")
