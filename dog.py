import streamlit as st
import cv2
import numpy as np
from keras.models import load_model

# Load the model 
new_model = load_model("model (1).h5", compile=False)
new_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Function to preprocess the uploaded image
def process_image(uploaded_file):
    # Convert BytesIO object to bytes
    image_bytes = uploaded_file.read()

    # Use np.frombuffer to convert bytes to numpy array
    nparr = np.frombuffer(image_bytes, np.uint8)

    # Decode the numpy array as an image using OpenCV
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    # Resize and preprocess the image
    resized_image = cv2.resize(image, (256, 256))
    processed_image = resized_image / 255.0  # Normalize pixel values to [0, 1]
    processed_image = np.expand_dims(processed_image, axis=0)  # Add batch dimension

    return processed_image

# Streamlit app
st.title("Puppy Sentiment Prediction")
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    st.image(uploaded_file, caption="Uploaded Image.", use_column_width=True)
    st.write("")
    st.write("Classifying...")

    # Process the uploaded image and make predictions
    processed_image = process_image(uploaded_file)
    prediction = new_model.predict(processed_image)

    # Determine sentiment based on the prediction
    if prediction < 0.5:
        st.write("Puppy is feeling happy!")
    else:
        st.write("Puppy is in a sad mood.")


