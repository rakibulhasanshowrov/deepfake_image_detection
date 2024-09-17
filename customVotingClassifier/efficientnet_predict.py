import pickle
import cv2 as cv
import numpy as np
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications.efficientnet import preprocess_input
from mtcnn import MTCNN

# Load the saved EfficientNet model
with open('efficientnet_model.pkl', 'rb') as f:
    model = pickle.load(f)

# Initialize face detector
detector = MTCNN()

def detect_and_crop_face(image_path):
    """Detect and crop face from an image."""
    image = cv.imread(image_path)
    image_rgb = cv.cvtColor(image, cv.COLOR_BGR2RGB)
    detections = detector.detect_faces(image_rgb)
    
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
        
        # Make prediction
        prediction = model.predict(face_array)
        confidence_score = prediction[0][0]
        
        if confidence_score > 0.5:
            return 0, confidence_score  # Fake image (0), confidence score
        else:
            return 1, 1 - confidence_score  # Real image (1), confidence score
    else:
        return None, None  # If no face is detected

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        print("Usage: python efficientnet_predict.py <image_path>")
        sys.exit(1)

    image_path = sys.argv[1]
    prediction, confidence = predict_image(image_path)

    if prediction is not None:
        # Output prediction and confidence as comma-separated values
        print(f"{prediction},{confidence:.4f}")
    else:
        print("Face could not be detected for prediction.")
