import os
import cv2
import numpy as np
import dlib
import mahotas
from sklearn import svm
from skimage.feature import local_binary_pattern, hog
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, classification_report
from sklearn.preprocessing import StandardScaler
from joblib import dump
import pickle

# Initialize face detector
detector = dlib.get_frontal_face_detector()

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

# Function to detect face and extract features
def extract_features(image_path):
    image = cv2.imread(image_path)
    rects = detector(image, 1)

    if len(rects) > 0:
        x, y, w, h = rects[0].left(), rects[0].top(), rects[0].width(), rects[0].height()
        face = image[y:y+h, x:x+w]  # Crop to the face region

        # Extract features
        lbp_features = extract_lbp(face)
        hog_features = extract_hog(face)
        color_histogram = extract_color_histogram(face)

        # Combine all features
        features = np.hstack([lbp_features, hog_features, color_histogram])
        return features
    else:
        return None

# Directory paths
dataset_dir = 'E:/498R/Dataset3_simple'
fake_dir = os.path.join(dataset_dir, 'Fake')
real_dir = os.path.join(dataset_dir, 'Real')

# Load and process images
data = []
labels = []

for category, label in [('Fake', 0), ('Real', 1)]:
    category_dir = os.path.join(dataset_dir, category)
    for file_name in os.listdir(category_dir):
        file_path = os.path.join(category_dir, file_name)
        features = extract_features(file_path)
        if features is not None:
            data.append(features)
            labels.append(label)

data = np.array(data)
labels = np.array(labels)

# Normalize the data
scaler = StandardScaler()
data = scaler.fit_transform(data)

# Save the extracted features and labels
np.save('features.npy', data)
np.save('labels.npy', labels)

# Split dataset into training, validation, and test sets
X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state=42)

# Train SVM classifier
clf = svm.SVC(kernel='linear', probability=True)
clf.fit(X_train, y_train)

# Save the trained model
with open('svm_model.pkl', 'wb') as model_file:
    pickle.dump(clf, model_file)

# Evaluate the model
y_pred_test = clf.predict(X_test)
y_pred_val = clf.predict(X_val)

# Calculate and display evaluation metrics
metrics = {
    'Accuracy': accuracy_score(y_test, y_pred_test),
    'Recall': recall_score(y_test, y_pred_test),
    'Precision': precision_score(y_test, y_pred_test),
    'F1 Score': f1_score(y_test, y_pred_test)
}

for metric, value in metrics.items():
    print(f"{metric}: {value:.4f}")

# Detailed classification report
print("\nClassification Report (Test Set):")
print(classification_report(y_test, y_pred_test))

# Save the report in a text file
report = classification_report(y_test, y_pred_test)
with open('svm_classification_report.txt', 'w') as report_file:
    report_file.write(report)

# Save the StandardScaler for future use
dump(scaler, 'scaler.joblib')
