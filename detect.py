# detect.py
from ultralytics import YOLO
import cv2

# Load pre-trained model
model = YOLO("yolov8n.pt")

# Load an image
image_path = "images/test.jpg"
results = model(image_path)

# Show results
results[0].show()  # Opens image with bounding boxes
