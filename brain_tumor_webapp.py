import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image
import plotly.graph_objects as go
import time

# ---------- Configuration ----------
st.set_page_config(page_title="🧠 Brain Tumor Classifier", layout="centered")

# Load the model
MODEL_PATH = r'C:\Users\aatka\Downloads\brain_tumor_model_vgg16.keras'
model = load_model(MODEL_PATH, compile=False)

# Class labels
class_labels = ['glioma', 'meningioma', 'no_tumor', 'pituitary']

#  Header 
st.markdown("""
    <h1 style='text-align: center; color: #4a4a4a;'>🧠 Brain Tumor Detection</h1>
    <p style='text-align: center; font-size: 18px;'> <b>Upload a brain MRI image to detect the Type of Brain Tumor</b>.</p>
    <hr>
""", unsafe_allow_html=True)

# Image Upload 
uploaded_file = st.file_uploader(" Upload a brain MRI image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Show the image
    img = Image.open(uploaded_file).convert('RGB')
    st.image(img, caption='Uploaded Image', use_column_width=True)

    # Preprocessing
    img_resized = img.resize((224,224)) 
    img_array = image.img_to_array(img_resized) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Simulate prediction delay
    with st.spinner('🔍 Analyzing Image...'):
        time.sleep(1.5)
        preds = model.predict(img_array)[0]

    pred_class_idx = np.argmax(preds)
    pred_class = class_labels[pred_class_idx]
    confidence = preds[pred_class_idx] * 100

    #  Prediction Output
    col1, col2 = st.columns([1, 2])
    with col1:
        st.markdown(f"###  **Prediction**")
        emoji = "✅" if pred_class == 'no_tumor' else "⚠️"
        st.markdown(f"**{emoji} {pred_class.capitalize()}**")

    with col2:
        st.markdown("###  Confidence")
        st.markdown(f"<span style='font-size: 24px; color: #007ACC;'><b>{confidence:.2f}%</b></span>", unsafe_allow_html=True)

    #  Confidence Score Plot 
    st.markdown("###  Class-wise Confidence Scores")
    fig = go.Figure(go.Bar(
        x=class_labels,
        y=preds,
        marker_color=['#ff6361', '#ffa600', '#58508d', '#bc5090']
    ))
    fig.update_layout(height=400, xaxis_title="Tumor Type", yaxis_title="Confidence", yaxis=dict(tickformat=".0%"))
    st.plotly_chart(fig)

   
    st.markdown("---")
    st.info("💡 The model is trained to recognize four tumor types. Always consult a medical professional for final diagnosis.")
else:
    st.warning("Please upload a brain MRI image to proceed.")
