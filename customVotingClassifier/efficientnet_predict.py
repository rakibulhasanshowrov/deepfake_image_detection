import pickle
import cv2 as cv
import numpy as np
from tensorflow.keras.utils import img_to_array
from keras.applications.vgg16 import preprocess_input
from mtcnn import MTCNN
import os
import sys
import io

# Load the saved EfficientNet model
with open('efficientnet_model.pkl', 'rb') as f:
    model = pickle.load(f)

# Initialize face detector
detector = MTCNN()

def detect_and_crop_face(image_path):
    """Detect and crop face from an image."""
    image = cv.imread(image_path)
    if image is None:
        print(f"Error: Could not read the image at {image_path}")
        return None
    image_rgb = cv.cvtColor(image, cv.COLOR_BGR2RGB)
    try:
        detections = detector.detect_faces(image_rgb)
    except UnicodeEncodeError as e:
        print(f"Encoding error: {e}", flush=True)
        return None

    if detections:
        x, y, width, height = detections[0]['box']
        face = image_rgb[y:y + height, x:x + width]
        face_resized = cv.resize(face, (224, 224))  # Resize to match EfficientNet input size
        return face_resized
    else:
        print("No face detected in the image.")
        return None

def predict_image(image_path):
    """Predict if an image is fake or real, and return prediction and confidence score."""
    # Detect and preprocess the face
    face = detect_and_crop_face(image_path)
    if face is not None:
        face_array = img_to_array(face)
        face_array = np.expand_dims(face_array, axis=0)  # Add batch dimension
        face_array = preprocess_input(face_array)  # Preprocess for EfficientNet
        
        # Make prediction (assuming sigmoid output)
        prediction = model.predict(face_array)
        confidence_score = prediction[0][0]
        
        # Classify as 1 for Real (confidence > 0.5) and 0 for Fake (confidence <= 0.5)
        predicted_class = 1 if confidence_score > 0.5 else 0
        return predicted_class, confidence_score
    else:
        return None, None  # If no face is detected

def save_result(output_file, predicted_class, confidence_score):
    """Save the result to a file."""
    with open(output_file, 'w') as file:
        file.write(f"{predicted_class},{confidence_score:.4f}")

if __name__ == "__main__":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

    if len(sys.argv) != 3:
        print("Usage: python efficientnet_predict.py <image_path> <output_file>", flush=True)
        sys.exit(1)

    image_path = sys.argv[1]
    output_file = sys.argv[2]

    if os.path.exists(image_path):
        predicted_class, confidence_score = predict_image(image_path)
        if predicted_class is not None:
            # Save prediction and confidence score to the output file
            save_result(output_file, predicted_class, confidence_score)
        else:
            print("Face could not be detected for prediction.")
    else:
        print(f"Error: Image path {image_path} does not exist.")

