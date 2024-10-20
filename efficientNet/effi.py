
from dir import *
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, models
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, precision_recall_curve
import cv2 as cv
import numpy as np
from mtcnn import MTCNN
import pandas as pd
from PIL import Image
import os

# Initialize the MTCNN face detector
detector = MTCNN()

# Function to detect and crop face from an image
def detect_and_crop_face(image_path):
    """Detect and crop face from an image."""
    image = cv.imread(image_path)
    image_rgb = cv.cvtColor(image, cv.COLOR_BGR2RGB)
    detections = detector.detect_faces(image_rgb)
    if detections:
        x, y, width, height = detections[0]['box']
        face = image_rgb[y:y + height, x:x + width]
        face_resized = cv.resize(face, (224, 224))  # Resize to match EfficientNet input size
        face_resized = Image.fromarray(face_resized)  # Convert to PIL image
        return face_resized
    return None

# Define transformations (normalization and resizing)
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Custom dataset for loading face images
class FaceDataset(Dataset):
    def __init__(self, image_dir, label, transform=None):
        self.image_dir = image_dir
        self.label = label
        self.transform = transform

    def __len__(self):
        return len(self.label)

    def __getitem__(self, idx):
        image_path, label = self.label[idx]
        face_image = detect_and_crop_face(image_path)
        if self.transform:
            face_image = self.transform(face_image)
        return face_image, label

# Load and preprocess dataset
def load_dataset(fake_limit=None, real_limit=None):
    """Load dataset, detect faces, and save features and labels."""
    features, labels = [], []
    
    # Process Fake images
    fake_count = 0
    for filename in os.listdir(fake_dir):
        if fake_limit and fake_count >= fake_limit:
            break
        image_path = os.path.join(fake_dir, filename)
        face = detect_and_crop_face(image_path)
        if face is not None:
            features.append((image_path, 0))  # Label for Fake
            fake_count += 1
    
    # Process Real images
    real_count = 0
    for filename in os.listdir(real_dir):
        if real_limit and real_count >= real_limit:
            break
        image_path = os.path.join(real_dir, filename)
        face = detect_and_crop_face(image_path)
        if face is not None:
            features.append((image_path, 1))  # Label for Real
            real_count += 1
    
    return features

# Load dataset
features = load_dataset(fake_limit=550, real_limit=550)

# Split dataset into training and test sets
train_size = int(0.8 * len(features))
train_features = features[:train_size]
test_features = features[train_size:]

# Create Dataset and DataLoader for training and testing
train_data = FaceDataset(train_features, transform=transform)
test_data = FaceDataset(test_features, transform=transform)

train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
test_loader = DataLoader(test_data, batch_size=32, shuffle=False)

# Load pre-trained EfficientNet and modify the classifier for binary classification
model = models.efficientnet_b0(pretrained=True)
model.classifier = nn.Sequential(
    nn.Linear(model.classifier[1].in_features, 128),
    nn.ReLU(),
    nn.Linear(128, 1),  # Binary classification
    nn.Sigmoid()  # For binary classification output
)

# Move model to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# Loss function and optimizer
criterion = nn.BCELoss()  # Binary cross-entropy for binary classification
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
for epoch in range(10):
    model.train()
    running_loss = 0.0
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device).float()
        
        optimizer.zero_grad()
        outputs = model(inputs).squeeze()
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
    
    print(f"Epoch {epoch+1}, Loss: {running_loss / len(train_loader)}")

# Evaluation on the test set
model.eval()
y_true = []
y_prob = []

with torch.no_grad():
    for inputs, labels in test_loader:
        inputs, labels = inputs.to(device), labels.to(device).float()
        outputs = model(inputs).squeeze()
        y_true.extend(labels.cpu().numpy())
        y_prob.extend(outputs.cpu().numpy())

# Convert predictions to numpy arrays
y_true = np.array(y_true)
y_prob = np.array(y_prob)

# Calculate precision-recall curve
precision, recall, thresholds = precision_recall_curve(y_true, y_prob)

# Find the optimal threshold based on F1 score
f1_scores = 2 * (precision * recall) / (precision + recall)
best_threshold = thresholds[np.argmax(f1_scores)]

# Convert probabilities to binary using the optimal threshold
y_pred = (y_prob > best_threshold).astype(int)

# Calculate accuracy, precision, recall, and F1
accuracy = accuracy_score(y_true, y_pred)
precision = precision_score(y_true, y_pred)
recall = recall_score(y_true, y_pred)
f1 = f1_score(y_true, y_pred)

print(f"Accuracy: {accuracy}, Precision: {precision}, Recall: {recall}, F1 Score: {f1}")

# Save the metrics in a table
metrics_df = pd.DataFrame({
    'Metric': ['Accuracy', 'Precision', 'Recall', 'F1 Score'],
    'Score': [accuracy, precision, recall, f1]
})
metrics_df.to_csv('EfficientNet_model_metrics_pytorch.csv', index=False)

# Print the metrics table
print(metrics_df)

