
import os
import cv2
import numpy as np
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

# Load images and extract features (blurriness and noise)
def load_images_and_extract_features(directory):
    features = []
    labels = []
    for label in ['Fake', 'Real']:
        folder_path = os.path.join(directory, label)
        for file_name in os.listdir(folder_path):
            file_path = os.path.join(folder_path, file_name)
            image = cv2.imread(file_path)
            if image is not None:
                face = crop_face(image)
                if face is not None:
                    blurriness = calculate_blurriness(face)
                    noise = calculate_noise(face)
                    features.append([blurriness, noise])
                    labels.append(0 if label == 'Fake' else 1)
    return np.array(features), np.array(labels)

# Main function for preprocessing
def main():
    # Define dataset directory
    dataset_directory = 'E:/498R/Dataset3/'

    # Extract features and labels
    X, y = load_images_and_extract_features(dataset_directory)

    # Save the extracted features and labels
    np.save('X.npy', X)
    np.save('y.npy', y)
    print(f"Features and labels have been saved. Total samples: {len(X)}")

if __name__ == "__main__":
    main()
