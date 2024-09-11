import cv2 as cv
import numpy as np
import pywt
from mtcnn.mtcnn import MTCNN

class Faceloading:
    def __init__(self, directory=None):
        self.directory = directory
        self.target_size = (160, 160)
        self.featureI = []
        self.classL = []
        self.detector = MTCNN()

    def extractFace(self, image):
        """Extracts face from the image"""
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

    def extract_features_from_image(self, image_path):
        """Process a single image and extract features for model prediction"""
        # Load the image
        image = cv.imread(image_path)
        image = cv.cvtColor(image, cv.COLOR_BGR2RGB)  # Convert to RGB

        # Extract face from the image
        face_array = self.extractFace(image)
        if face_array is not None:
            # Extract features from the face
            features = self.extract_features(face_array)
            return np.array(features).reshape(1, -1)  # Return features reshaped for model prediction
        else:
            print("No face detected in the image.")
            return None

    # Other methods remain the same...
