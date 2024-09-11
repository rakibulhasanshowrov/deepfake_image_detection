from mtcnn.mtcnn import MTCNN
import cv2 as cv
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np
import pywt  # For wavelet transforms
import pandas as pd
import pickle

# Directory Path
dataset_dir = 'E:/498R/dataset_simple/Train'

class Faceloading:
    def __init__(self, directory):
        self.directory = directory
        self.target_size = (160, 160)
        self.featureI = []
        self.classL = []
        self.detector = MTCNN()

    def extractFace(self, dir):
        image = cv.imread(dir)
        image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
        result = self.detector.detect_faces(image)
        if result:  # Ensure a face is detected
            x, y, w, h = result[0]['box']
            x, y = abs(x), abs(y)
            face = image[y:y + h, x:x + w]
            face_arr = cv.resize(face, self.target_size)
            return face_arr
        else:
            return None  # Handle cases where no face is detected

    def calculate_laplacian_variance(self, face_image):
        """Calculate the Laplacian variance for blur detection"""
        gray = cv.cvtColor(face_image, cv.COLOR_BGR2GRAY)
        laplacian = cv.Laplacian(gray, cv.CV_64F)
        variance = laplacian.var()
        return variance

    def calculate_wavelet_noise(self, face_image):
        """Calculate wavelet noise feature using Wavelet Transform"""
        gray = cv.cvtColor(face_image, cv.COLOR_BGR2GRAY)
        coeffs = pywt.wavedec2(gray, 'haar', level=2)
        cH2, cV2, cD2 = coeffs[1]  # Detail coefficients
        noise_std = np.std(cD2)
        return noise_std

    def calculate_dct_frequency(self, face_image):
        """Calculate frequency feature using Discrete Cosine Transform (DCT)"""
        gray = cv.cvtColor(face_image, cv.COLOR_BGR2GRAY)
        dct = cv.dct(np.float32(gray))
        dct_high_freq = dct[10:, 10:]  # Extract high-frequency components
        high_freq_sum = np.sum(np.abs(dct_high_freq))
        return high_freq_sum

    def extract_features(self, face_array):
        """Combine all features (blur, noise, frequency) into a single vector"""
        blur_feature = self.calculate_laplacian_variance(face_array)
        noise_feature = self.calculate_wavelet_noise(face_array)
        frequency_feature = self.calculate_dct_frequency(face_array)
        return [blur_feature, noise_feature, frequency_feature]

    def load_faces(self, dir):
        """Load faces and extract features"""
        Faces = []
        for image_dir in dir.iterdir():
            if image_dir.suffix == '.jpg':
                face_array = self.extractFace(str(image_dir))
                if face_array is not None:
                    # Extract features from the face image
                    features = self.extract_features(face_array)
                    Faces.append(features)
                else:
                    print(f"No face detected in {image_dir}")
            else:
                print('Error Occurred in Load Faces Method')
        return Faces

    def load_classes(self):
        """Load all classes and corresponding feature vectors"""
        for dir in self.directory.iterdir():
            if dir.is_dir():
                print('Class Name is: {}'.format(dir.name))
                label = dir.name
                Faces = self.load_faces(dir)
                # Extend features and classes
                self.featureI.extend(Faces)
                self.classL.extend([label] * len(Faces))  # Extend the label for each face
            else:
                print('Error Occurred in Load Classes Method')
        return np.asarray(self.featureI), np.asarray(self.classL)


# Instantiate and load features with classes
faceloading = Faceloading(Path(dataset_dir))
features, labels = faceloading.load_classes()
print("Extracted Features Shape:", features.shape)
print("Labels Shape:", labels.shape)


# Save the features and labels to files
np.save('E:/498R/Code/Result/simple/features.npy', features)
np.save('E:/498R/Code/Result/simple/labels.npy', labels)
print("Features and labels have been saved as .npy files.")




# Save the DataFrame as a CSV file
df = pd.DataFrame(features)
df['label'] = labels
df.to_csv('E:/498R/Code/Result/simple/features_and_labels.csv', index=False)
print("Features and labels have been saved as a CSV file.")


# Save using pickle
with open('E:/498R/Code/Result/simple/features_and_labels.pkl', 'wb') as f:
    pickle.dump((features, labels), f)

print("Features and labels have been saved as a Pickle file.")