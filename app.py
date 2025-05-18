import streamlit as st
from ultralytics import YOLO
import cv2
from PIL import Image
import numpy as np

st.title("🧠 Object Detection with YOLOv8")
model = YOLO("yolov8n.pt")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Convert image for OpenCV
    image_array = np.array(image)

    # Run detection
    results = model(image_array)[0]
    res_plotted = results.plot()

    st.image(res_plotted, caption="Detected Objects", use_column_width=True)
