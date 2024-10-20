import os
import numpy as np
import cv2 as cv
from mtcnn import MTCNN
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms
import pandas as pd
from sklearn.metrics import precision_recall_curve, accuracy_score, recall_score, precision_score, f1_score

# Initialize face detector
detector = MTCNN()

# Transform for image preprocessing
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Custom dataset class
class FaceDataset(Dataset):
    def __init__(self, image_dir, label, transform=None):
        self.image_dir = image_dir
        self.label = label
        self.transform = transform
        self.image_paths = [os.path.join(image_dir, img) for img in os.listdir(image_dir) if img.endswith(('jpg', 'png', 'jpeg'))]

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        face = self.detect_and_crop_face(image_path)
        if self.transform:
            face = self.transform(face)
        return face, self.label

    def detect_and_crop_face(self, image_path):
        image = cv.imread(image_path)
        image_rgb = cv.cvtColor(image, cv.COLOR_BGR2RGB)
        detections = detector.detect_faces(image_rgb)
        if detections:
            x, y, width, height = detections[0]['box']
            face = image_rgb[y:y + height, x:x + width]
            return face
        else:
            return np.zeros((224, 224, 3), dtype=np.uint8)  # Return a blank image if no face is detected

# Load dataset
def load_dataset(fake_dir, real_dir, transform=None):
    fake_dataset = FaceDataset(fake_dir, 0, transform=transform)
    real_dataset = FaceDataset(real_dir, 1, transform=transform)
    
    return fake_dataset, real_dataset

# Prepare datasets
fake_dir = "E:/498R/Dataset3_simple/Fake"
real_dir = "E:/498R/Dataset3_simple/Real"
fake_dataset, real_dataset = load_dataset(fake_dir, real_dir, transform=transform)

# Combine the datasets
full_dataset = torch.utils.data.ConcatDataset([fake_dataset, real_dataset])

# Split into train and test sets
train_size = int(0.8 * len(full_dataset))
test_size = len(full_dataset) - train_size
train_dataset, test_dataset = torch.utils.data.random_split(full_dataset, [train_size, test_size])

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=4)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# Check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load EfficientNet model
class EfficientNetModel(nn.Module):
    def __init__(self):
        super(EfficientNetModel, self).__init__()
        self.base_model = models.efficientnet_b0(pretrained=True)
        self.base_model.classifier[1] = nn.Linear(self.base_model.classifier[1].in_features, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.base_model(x)
        return self.sigmoid(x)

# Move the model to GPU if available
model = EfficientNetModel().to(device)

# Define loss function and optimizer
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
def train_model(model, train_loader, criterion, optimizer, epochs=10):
    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.float().to(device), labels.float().unsqueeze(1).to(device)  # Move data to GPU
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {running_loss/len(train_loader)}")

# Train the model
train_model(model, train_loader, criterion, optimizer)

# Evaluate the model
def evaluate_model(model, test_loader):
    model.eval()
    y_true, y_prob = [], []
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.float().to(device)  # Move data to GPU
            outputs = model(inputs)
            y_true.extend(labels.numpy())
            y_prob.extend(outputs.cpu().numpy().flatten())  # Move outputs to CPU for further processing

    return np.array(y_true), np.array(y_prob)

y_true, y_prob = evaluate_model(model, test_loader)

# Calculate precision-recall curve
precision, recall, thresholds = precision_recall_curve(y_true, y_prob)

# Find the optimal threshold based on the highest F1 score
f1_scores = 2 * (precision * recall) / (precision + recall)
best_threshold = thresholds[np.argmax(f1_scores)]

print("Best threshold:", best_threshold)

# Convert probabilities to binary using the optimal threshold
y_pred = (y_prob > best_threshold).astype("int32")

# Evaluate the model
accuracy = accuracy_score(y_true, y_pred)
recall = recall_score(y_true, y_pred, zero_division=1)
precision = precision_score(y_true, y_pred, zero_division=1)
f1 = f1_score(y_true, y_pred, zero_division=1)

# Save the metrics in a table
metrics_df = pd.DataFrame({
    'Metric': ['Accuracy', 'Precision', 'Recall', 'F1 Score'],
    'Score': [accuracy, precision, recall, f1]
})
metrics_df.to_csv('EfficientNet_model_metrics.csv', index=False)

# Print the metrics table
print(metrics_df)
