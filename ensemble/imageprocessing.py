from imp import *
from dir import *

image_size = (160, 160)  # Image size should match your model input size

# Initialize the face detector (Haar Cascade Classifier or another method)
face_cascade = cv.CascadeClassifier(cv.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Function to detect and crop faces from images
def detect_and_crop_face(image):
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)  # Convert to grayscale for face detection
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    
    if len(faces) > 0:
        for (x, y, w, h) in faces:
            face = image[y:y+h, x:x+w]  # Crop the detected face
            face_resized = cv.resize(face, image_size)  # Resize to match input size
            return face_resized
    return None  # If no face is detected, return None

# Use ImageDataGenerator to load and preprocess images
datagen = ImageDataGenerator(rescale=1.0/255.0)  # Rescale pixel values to [0, 1]

# Load images from folders ('fake' and 'real')
generator = datagen.flow_from_directory(
    dataset_dir,
    target_size=(256, 256),  # Initial larger size for better face detection
    class_mode='binary',     # Binary classification ('fake' or 'real')
    batch_size=32,
    shuffle=True,
)

# Process and store the images and labels
X, y = [], []
for _ in range(len(generator)):
    images, labels = next(generator)  # Get a batch of images and labels

    for img, label in zip(images, labels):
        img_uint8 = (img * 255).astype(np.uint8)  # Convert image back to uint8 for OpenCV
        face = detect_and_crop_face(img_uint8)    # Detect and crop the face
        
        if face is not None:
            face_normalized = face / 255.0  # Normalize pixel values to [0, 1]
            X.append(face_normalized)       # Add the face image to X
            y.append(label)                 # Add the corresponding label to y

# Convert lists to NumPy arrays
X = np.array(X)
y = np.array(y)

# Print shapes for verification
print("Shape of dataset (X):", X.shape)
print("Shape of labels (y):", y.shape)

# Split data into training, validation, and test sets
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# Save the NumPy arrays
np.save('X_train.npy', X_train)
np.save('X_val.npy', X_val)
np.save('X_test.npy', X_test)

np.save('y_train.npy', y_train)
np.save('y_val.npy', y_val)
np.save('y_test.npy', y_test)

# Alternatively, save all arrays in a compressed file
np.savez('dataset_split.npz', X_train=X_train, X_val=X_val, X_test=X_test, 
         y_train=y_train, y_val=y_val, y_test=y_test)

print("Face images and labels saved successfully.")
print("Training data shape:", X_train.shape)
print("Validation data shape:", X_val.shape)
print("Test data shape:", X_test.shape)