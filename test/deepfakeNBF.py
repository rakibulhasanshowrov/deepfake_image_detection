import os
import cv2 as cv
import numpy as np
from mtcnn import MTCNN
import pywt

class Faceloading:
    def __init__(self, directory, output_dir):
        self.directory = directory
        self.output_dir = output_dir
        self.target_size = (160, 160)
        self.detector = MTCNN()

        os.makedirs(self.output_dir, exist_ok=True)  # Create output directory if it doesn't exist

    def save_image(self, image, step_name):
        """Save the image at a specific processing step."""
        output_path = os.path.join(self.output_dir, f'{step_name}.png')
        cv.imwrite(output_path, cv.cvtColor(image, cv.COLOR_RGB2BGR))
        print(f"Saved {step_name} image to {output_path}")

    def extractFace(self, dir):
        image = cv.imread(dir)
        image_rgb = cv.cvtColor(image, cv.COLOR_BGR2RGB)
        result = self.detector.detect_faces(image_rgb)

        if result:  # Ensure a face is detected
            x, y, w, h = result[0]['box']
            x, y = abs(x), abs(y)
            face = image_rgb[y:y + h, x:x + w]
            face_arr = cv.resize(face, self.target_size)
            self.save_image(face_arr, '01_face_extracted')
            return face_arr
        else:
            print("No face detected.")
            return None

    def calculate_laplacian_variance(self, face_image):
        """Calculate the Laplacian variance for blur detection and save the result."""
        gray = cv.cvtColor(face_image, cv.COLOR_RGB2GRAY)
        laplacian = cv.Laplacian(gray, cv.CV_64F)
        variance = laplacian.var()
        
        # Normalize the Laplacian for visualization
        laplacian_normalized = cv.normalize(np.abs(laplacian), None, 0, 255, cv.NORM_MINMAX, cv.CV_8U)
        self.save_image(cv.cvtColor(laplacian_normalized, cv.COLOR_GRAY2RGB), '02_laplacian_variance')
        
        return variance

    def calculate_wavelet_noise(self, face_image):
        """Calculate wavelet noise feature and save the result."""
        gray = cv.cvtColor(face_image, cv.COLOR_RGB2GRAY)
        coeffs = pywt.wavedec2(gray, 'haar', level=2)
        cH2, cV2, cD2 = coeffs[1]  # Detail coefficients
        noise_std = np.std(cD2)
        
        # Visualize wavelet coefficients for saving
        wavelet_image = cv.normalize(np.abs(cD2), None, 0, 255, cv.NORM_MINMAX, cv.CV_8U)
        self.save_image(cv.cvtColor(wavelet_image, cv.COLOR_GRAY2RGB), '03_wavelet_noise')
        
        return noise_std

    def calculate_dct_frequency(self, face_image):
        """Calculate frequency feature using Discrete Cosine Transform (DCT) and save the result."""
        gray = cv.cvtColor(face_image, cv.COLOR_RGB2GRAY)
        dct = cv.dct(np.float32(gray))
        dct_high_freq = dct[10:, 10:]  # Extract high-frequency components
        
        # Normalize and save the DCT image for visualization
        dct_normalized = cv.normalize(np.abs(dct_high_freq), None, 0, 255, cv.NORM_MINMAX, cv.CV_8U)
        self.save_image(cv.cvtColor(dct_normalized, cv.COLOR_GRAY2RGB), '04_dct_frequency')
        
        high_freq_sum = np.sum(np.abs(dct_high_freq))
        return high_freq_sum

    def extract_features(self, face_array):
        """Combine all features (blur, noise, frequency) into a single vector."""
        blur_feature = self.calculate_laplacian_variance(face_array)
        noise_feature = self.calculate_wavelet_noise(face_array)
        frequency_feature = self.calculate_dct_frequency(face_array)
        return [blur_feature, noise_feature, frequency_feature]

    def process_single_image(self, image_path):
        """Process a single image and extract features."""
        face_array = self.extractFace(image_path)
        if face_array is not None:
            features = self.extract_features(face_array)
            print(f"Extracted features: {features}")
        else:
            print("No face detected in the image.")

# Example usage:
image_path = 'E:/498R/Code/Testingimage/r.jpg'  # Replace with your image path
output_dir = 'E:/498R/Code/test'  # Replace with your desired output folder

faceloading = Faceloading(directory=None, output_dir=output_dir)
faceloading.process_single_image(image_path)
