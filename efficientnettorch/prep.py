import os
import cv2 as cv
from mtcnn import MTCNN

# Initialize face detector
detector = MTCNN()

# Directories
fake_dir = "E:/498R/Dataset3/Fake"
real_dir = "E:/498R/Dataset3/Real"
preprocessed_fake_dir = "E:/498R/Code/efficientnettorch/Preprocessed/Fake"
preprocessed_real_dir = "E:/498R/Code/efficientnettorch/Preprocessed/Real"

# Create directories for preprocessed images if they don't exist
os.makedirs(preprocessed_fake_dir, exist_ok=True)
os.makedirs(preprocessed_real_dir, exist_ok=True)


def detect_and_crop_face(image_path):
    image = cv.imread(image_path)
    image_rgb = cv.cvtColor(image, cv.COLOR_BGR2RGB)
    detections = detector.detect_faces(image_rgb)
    if detections:
        x, y, width, height = detections[0]["box"]
        face = image_rgb[y : y + height, x : x + width]
        return face
    else:
        return None  # Return None if no face is detected


def preprocess_images(input_dir, output_dir):
    for img_file in os.listdir(input_dir):
        if img_file.endswith(("jpg", "png", "jpeg")):
            img_path = os.path.join(input_dir, img_file)
            face = detect_and_crop_face(img_path)
            if face is not None:
                # Resize and save the cropped face
                face_resized = cv.resize(face, (224, 224))
                save_path = os.path.join(output_dir, img_file)
                cv.imwrite(
                    save_path, cv.cvtColor(face_resized, cv.COLOR_RGB2BGR)
                )  # Convert back to BGR for saving
            else:
                print(f"No face detected in {img_file}, skipping...")


# Preprocess and save fake and real images
preprocess_images(fake_dir, preprocessed_fake_dir)
preprocess_images(real_dir, preprocessed_real_dir)

print("Preprocessing completed.")
