from imp import *
from dir import *

image_size = (64, 64)  # Image size should match your model input size

# Use ImageDataGenerator to load images from the directories
datagen = ImageDataGenerator(rescale=1.0/255.0)  # Rescale pixel values to [0, 1]

# Load images from folders ('fake' and 'real') and convert to NumPy arrays
generator = datagen.flow_from_directory(
    dataset_dir,
    target_size=image_size,
    class_mode='binary',  # Binary classification ('fake' or 'real')
    batch_size=32,
    shuffle=True,
)

# Get the total number of images and labels
X, y = [], []
for _ in range(len(generator)):
    images, labels = next(generator)
    X.extend(images)
    y.extend(labels)

# Convert to NumPy arrays
X = np.array(X)
y = np.array(y)

# Split data into training, validation, and test sets (e.g., 70% training, 15% validation, 15% test)
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)



# Save the individual NumPy arrays
np.save('X_train.npy', X_train)
np.save('X_val.npy', X_val)
np.save('X_test.npy', X_test)

np.save('y_train.npy', y_train)
np.save('y_val.npy', y_val)
np.save('y_test.npy', y_test)

# Alternatively, save them all together in a compressed file
np.savez('dataset_split.npz', X_train=X_train, X_val=X_val, X_test=X_test, 
         y_train=y_train, y_val=y_val, y_test=y_test)

print("Arrays saved successfully.")

print("Training data shape:", X_train.shape)
print("Validation data shape:", X_val.shape)
print("Test data shape:", X_test.shape)