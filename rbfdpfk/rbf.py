import os
import cv2
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score
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

# Main function
def main():
    # Load dataset and extract features
    dataset_directory = 'E:/498R/Dataset3_simple/'
    X, y = load_images_and_extract_features(dataset_directory)
    
    # Split data into training, validation, and testing sets
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)
    
    # Train Random Forest model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Save the model and dataset
    with open('random_forest_model.pkl', 'wb') as model_file:
        pickle.dump(model, model_file)
    np.save('X_train.npy', X_train)
    np.save('X_val.npy', X_val)
    np.save('X_test.npy', X_test)
    np.save('y_train.npy', y_train)
    np.save('y_val.npy', y_val)
    np.save('y_test.npy', y_test)
    
    # Evaluate model on validation set
    y_pred = model.predict(X_val)
    accuracy = accuracy_score(y_val, y_pred)
    precision = precision_score(y_val, y_pred)
    recall = recall_score(y_val, y_pred)
    f1 = f1_score(y_val, y_pred)
    
    # Print evaluation report
    print("Classification Report:\n", classification_report(y_val, y_pred, target_names=['Fake', 'Real']))
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")

if __name__ == "__main__":
    main()
