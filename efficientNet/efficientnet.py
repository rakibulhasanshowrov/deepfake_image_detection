from imp import *
from dir import *
# Initialize face detector
detector = MTCNN()

def detect_and_crop_face(image_path):
    """Detect and crop face from an image."""
    image = cv.imread(image_path)
    image_rgb = cv.cvtColor(image, cv.COLOR_BGR2RGB)
    detections = detector.detect_faces(image_rgb)
    if detections:
        x, y, width, height = detections[0]['box']
        face = image_rgb[y:y + height, x:x + width]
        face_resized = cv.resize(face, (224, 224))  # Resize to match EfficientNet input size
        return img_to_array(face_resized)
    return None

# Prepare dataset
def load_dataset():
    """Load dataset, detect faces, and save features and labels."""
    features, labels = [], []
    for category, label in [('Fake', 0), ('Real', 1)]:
        folder = fake_dir if category == 'Fake' else real_dir
        for filename in os.listdir(folder):
            image_path = os.path.join(folder, filename)
            face = detect_and_crop_face(image_path)
            if face is not None:
                features.append(face)
                labels.append(label)
    
    # Convert lists to NumPy arrays
    features = np.array(features, dtype='float32') / 255.0  # Normalize pixel values
    labels = np.array(labels)
    
    # Save features and labels
    np.save('features.npy', features)
    np.save('labels.npy', labels)
    
    return features, labels

# Load the dataset
features, labels = load_dataset()

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)
unique, counts = np.unique(y_train, return_counts=True)
print("Class distribution in training set:", dict(zip(unique, counts)))


# Load EfficientNetB0 model and add custom layers
def build_efficientnet_model():
    base_model = EfficientNetB0(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(128, activation='relu')(x)
    output = Dense(1, activation='sigmoid')(x)  # Binary classification output
    model = Model(inputs=base_model.input, outputs=output)
    
    # Unfreeze the last few layers of EfficientNet
    for layer in model.layers[-20:]:
        layer.trainable = True

    # Compile the model again after unfreezing
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

# Build and train the EfficientNet model
model = build_efficientnet_model()
model.fit(X_train, y_train, epochs=10, validation_split=0.2)

# Get predicted probabilities
y_prob = model.predict(X_test)

# Calculate precision-recall curve
precision, recall, thresholds = precision_recall_curve(y_test, y_prob)

# Find the optimal threshold based on the highest F1 score
f1_scores = 2 * (precision * recall) / (precision + recall)
best_threshold = thresholds[np.argmax(f1_scores)]

print("Best threshold:", best_threshold)

# Convert probabilities to binary using the optimal threshold
y_pred = (y_prob > best_threshold).astype("int32")
print("Predicted labels:", y_pred[:10])

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
recall = recall_score(y_test, y_pred, zero_division=1)
precision = precision_score(y_test, y_pred, zero_division=1)
f1 = f1_score(y_test, y_pred, zero_division=1)

# Save the model using pickle
with open('efficientnet_model.pkl', 'wb') as f:
    pickle.dump(model, f)

# Save the metrics in a table
metrics_df = pd.DataFrame({
    'Metric': ['Accuracy', 'Precision', 'Recall', 'F1 Score'],
    'Score': [accuracy, precision, recall, f1]
})
metrics_df.to_csv('Efficientnet_model_metrics.csv', index=False)

# Print the metrics table
print(metrics_df)