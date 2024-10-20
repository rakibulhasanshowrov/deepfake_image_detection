import os
import cv2
import numpy as np
import dlib
from skimage.feature import local_binary_pattern, hog

# Initialize face detector
detector = dlib.get_frontal_face_detector()

# Directories for fake and real images
fake_dir = "E:/498R/20k dataset/20K Splitted Dataset/train/Fake"
real_dir = "E:/498R/20k dataset/20K Splitted Dataset/train/Real"

# Fixed face size for feature extraction
fixed_face_size = (128, 128)


# Feature extraction functions
def extract_lbp(image, radius=1, n_points=8, method="uniform"):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    lbp = local_binary_pattern(gray, n_points, radius, method)
    lbp_hist, _ = np.histogram(
        lbp, bins=np.arange(0, n_points + 3), range=(0, n_points + 2)
    )
    lbp_hist = lbp_hist.astype("float")
    lbp_hist /= lbp_hist.sum() + 1e-6  # Normalize
    return lbp_hist


def extract_hog(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    hog_features = hog(
        gray,
        orientations=9,
        pixels_per_cell=(8, 8),
        cells_per_block=(2, 2),
        block_norm="L2-Hys",
    )
    return hog_features


def extract_color_histogram(image):
    hist = cv2.calcHist([image], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
    hist = cv2.normalize(hist, hist).flatten()
    return hist


def extract_features(image_path):
    image = cv2.imread(image_path)
    rects = detector(image, 1)

    if len(rects) > 0:
        x, y, w, h = (
            rects[0].left(),
            rects[0].top(),
            rects[0].width(),
            rects[0].height(),
        )
        face = image[y : y + h, x : x + w]  # Crop to the face region
        if face.size == 0:
            print(f"Error: Face region is empty in {image_path}")
            return None
        face_resized = cv2.resize(face, fixed_face_size)
        lbp_features = extract_lbp(face_resized)
        hog_features = extract_hog(face_resized)
        color_histogram = extract_color_histogram(face_resized)
        features = np.hstack([lbp_features, hog_features, color_histogram])
        return features
    else:
        return None


def load_images():
    data = []
    labels = []
    invalid_images = []

    # Process all Fake images
    for file_name in os.listdir(fake_dir):
        file_path = os.path.join(fake_dir, file_name)
        features = extract_features(file_path)
        if features is not None and len(features) > 0:
            data.append(features)
            labels.append(0)  # Fake label
        else:
            invalid_images.append(file_path)

    # Process all Real images
    for file_name in os.listdir(real_dir):
        file_path = os.path.join(real_dir, file_name)
        features = extract_features(file_path)
        if features is not None and len(features) > 0:
            data.append(features)
            labels.append(1)  # Real label
        else:
            invalid_images.append(file_path)

    if invalid_images:
        print(f"Could not extract features from the following images:")
        for img_path in invalid_images:
            print(img_path)

    data = np.array(data)
    labels = np.array(labels)

    return data, labels


# Process all images
features, labels = load_images()

# Save features and labels
np.save("features.npy", features)
np.save("labels.npy", labels)

print(f"Processed and saved {len(features)} images.")
