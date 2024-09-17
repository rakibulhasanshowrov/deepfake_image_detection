import os
import cv2
import numpy as np
import dlib
from skimage.feature import local_binary_pattern, hog
import pickle
from joblib import load

# Initialize face detector
detector = dlib.get_frontal_face_detector()

# Fixed face size for feature extraction
fixed_face_size = (128, 128)

# Load saved SVM model and scaler
with open('svm_model.pkl', 'rb') as model_file:
    clf = pickle.load(model_file)

scaler = load('scaler.joblib')

# Define parameters for LBP, HOG, and color histograms
radius = 1
n_points = 8 * radius
method = 'uniform'

# Feature extraction functions
def extract_lbp(image, radius=1, n_points=8, method='uniform'):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    lbp = local_binary_pattern(gray, n_points, radius, method)
    lbp_hist, _ = np.histogram(lbp, bins=np.arange(0, n_points + 3), range=(0, n_points + 2))
    lbp_hist = lbp_hist.astype("float")
    lbp_hist /= (lbp_hist.sum() + 1e-6)  # Normalize
    return lbp_hist

def extract_hog(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    hog_features = hog(gray, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2), block_norm='L2-Hys')
    return hog_features

def extract_color_histogram(image):
    hist = cv2.calcHist([image], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
    hist = cv2.normalize(hist, hist).flatten()
    return hist

# Function to detect face, resize it, and extract features
def extract_features(image_path):
    image = cv2.imread(image_path)
    rects = detector(image, 1)

    if len(rects) > 0:
        x, y, w, h = rects[0].left(), rects[0].top(), rects[0].width(), rects[0].height()
        face = image[y:y+h, x:x+w]  # Crop to the face region

        # Resize face to fixed size (128x128)
        face_resized = cv2.resize(face, fixed_face_size)

        # Extract features
        lbp_features = extract_lbp(face_resized)
        hog_features = extract_hog(face_resized)
        color_histogram = extract_color_histogram(face_resized)

        # Combine all features
        features = np.hstack([lbp_features, hog_features, color_histogram])
        return features
    else:
        print("No face detected in the image.")
        return None

# Function to predict class and confidence score
def predict_image(image_path):
    features = extract_features(image_path)

    if features is not None:
        # Preprocess the features
        features = scaler.transform([features])

        # Predict the class
        prediction = clf.predict(features)
        confidence = clf.predict_proba(features)

        # Get confidence score for the predicted class
        predicted_class = 1 if prediction[0] == 1 else 0  # 1 = Real, 0 = Fake
        confidence_score = confidence[0][prediction[0]]
        # print("SVM Prediction:{prediction}".format(prediction))
        # print("SVM Confidence ScoreBF:{confidence_score}".format(confidence_score))

        return predicted_class, confidence_score
    else:
        return None, None

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        print("Usage: python svm_predict.py <image_path>")
        sys.exit(1)

    image_path = sys.argv[1]
    if os.path.exists(image_path):
        predicted_class, confidence = predict_image(image_path)
        # print("SVM Returun to parentSystem:{prediction},{confidence}".format(prediction,confidence))
        if predicted_class is not None:
            # Output prediction and confidence as comma-separated values
            print(f"{predicted_class},{confidence:.4f}")
        else:
            print("Face could not be detected for prediction.")
    else:
        print(f"Error: Image path {image_path} does not exist.")
