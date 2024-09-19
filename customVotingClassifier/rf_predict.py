import cv2
import numpy as np
import pickle
from skimage.restoration import estimate_sigma

# Function to detect and crop face using OpenCV
def crop_face(image):
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
    if len(faces) == 0:
        return None
    x, y, w, h = faces[0]
    return image[y:y+h, x:x+w]

# Function to calculate blurriness (Laplacian variance)
def calculate_blurriness(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    variance_of_laplacian = cv2.Laplacian(gray, cv2.CV_64F).var()
    return variance_of_laplacian

# Function to estimate noise patterns (using skimage)
def calculate_noise(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    sigma_est = estimate_sigma(gray, average_sigmas=True)
    return sigma_est

# Function to preprocess image and extract features
def preprocess_image(image_path):
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError("Image not found or unable to load.")
    face = crop_face(image)
    if face is None:
        raise ValueError("No face detected in the image.")
    blurriness = calculate_blurriness(face)
    noise = calculate_noise(face)
    return np.array([[blurriness, noise]])

# Function to load model and make prediction
def predict(image_path, model_path='random_forest_model.pkl'):
    # Load the trained model
    with open(model_path, 'rb') as model_file:
        model = pickle.load(model_file)
    
    # Preprocess the image
    features = preprocess_image(image_path)
    
    # Make prediction
    prediction = model.predict(features)
    confidence = model.predict_proba(features).max()
    
    # Map prediction to class labels
    class_labels = ['Fake', 'Real']
    predicted_class = 1 if prediction[0] == 1 else 0  # 1 = Real, 0 = Fake
    
    return predicted_class, confidence

# Function to save prediction and confidence to a file
def save_result(output_file, predicted_class, confidence_score):
    """Save the result to a file."""
    with open(output_file, 'w') as file:
        file.write(f"{predicted_class},{confidence_score:.4f}")

if __name__ == "__main__":
    import sys

    if len(sys.argv) != 3:
        print("Usage: python rf_predict.py <image_path> <output_file>", flush=True)
        sys.exit(1)

    image_path = sys.argv[1]
    output_file = sys.argv[2]

    try:
        predicted_class, confidence = predict(image_path)
        # Save prediction and confidence score to the output file
        save_result(output_file, predicted_class, confidence)
    except ValueError as e:
        print(f"Error: {e}")
    except Exception as e:
        print(f"Unexpected error: {e}")

