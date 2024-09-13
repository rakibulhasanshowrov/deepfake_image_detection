
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import confusion_matrix, accuracy_score, recall_score, f1_score, precision_score
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pickle
from predict import Faceloading 

# Paths to the combined dataset file (features and labels)
combined_features_path = 'E:/498R/Code/Result/Dataset3/dataset3_simple_features.npy'
combined_labels_path = 'E:/498R/Code/Result/Dataset3/dataset3_simple_labels.npy'
# Enable interactive mode
plt.ion()

# Load the features and labels
features = np.load(combined_features_path)
labels = np.load(combined_labels_path)
print(labels)

# Convert labels from 'Fake'/'Real' to 0/1 for binary classification
label_mapping = {'Fake': 0, 'Real': 1}
labels = np.array([label_mapping[label] for label in labels])

# 1. Split the dataset into train, test, and validation sets
train_features, temp_features, train_labels, temp_labels = train_test_split(
    features, labels, test_size=0.3, random_state=42, stratify=labels)

# Split temp set into validation and test sets
test_features, val_features, test_labels, val_labels = train_test_split(
    temp_features, temp_labels, test_size=0.5, random_state=42, stratify=temp_labels)

print(f"Train set size: {train_features.shape[0]}")
print(f"Test set size: {test_features.shape[0]}")
print(f"Validation set size: {val_features.shape[0]}")

# 2. Train an SVM model
clf = svm.SVC(kernel='poly', probability=True)  # Set probability=True to get confidence scores later
clf.fit(train_features, train_labels)

# 3. Evaluate the model
# Predict on the train, test, and validation sets
train_predictions = clf.predict(train_features)
test_predictions = clf.predict(test_features)
val_predictions = clf.predict(val_features)

# 4. Calculate and save metrics (Accuracy, Recall, F1, Precision) for train, test, and validation sets
metrics = {
    'train_accuracy': accuracy_score(train_labels, train_predictions),
    'train_recall': recall_score(train_labels, train_predictions),
    'train_precision': precision_score(train_labels, train_predictions),
    'train_f1': f1_score(train_labels, train_predictions),
    'test_accuracy': accuracy_score(test_labels, test_predictions),
    'test_recall': recall_score(test_labels, test_predictions),
    'test_precision': precision_score(test_labels, test_predictions),
    'test_f1': f1_score(test_labels, test_predictions),
    'val_accuracy': accuracy_score(val_labels, val_predictions),
    'val_recall': recall_score(val_labels, val_predictions),
    'val_precision': precision_score(val_labels, val_predictions),
    'val_f1': f1_score(val_labels, val_predictions)
}

# Format the metrics to two decimal points
formatted_metrics = {k: f"{v:.2f}" for k, v in metrics.items()}

# Create a table to display the metrics
fig, ax = plt.subplots()
ax.axis('tight')
ax.axis('off')
table_data = [
    ["Metric", "Train", "Test", "Validation"],
    ["Accuracy", formatted_metrics['train_accuracy'], formatted_metrics['test_accuracy'], formatted_metrics['val_accuracy']],
    ["Recall", formatted_metrics['train_recall'], formatted_metrics['test_recall'], formatted_metrics['val_recall']],
    ["Precision", formatted_metrics['train_precision'], formatted_metrics['test_precision'], formatted_metrics['val_precision']],
    ["F1 Score", formatted_metrics['train_f1'], formatted_metrics['test_f1'], formatted_metrics['val_f1']]
]

table = ax.table(cellText=table_data, loc='center', cellLoc='center', colWidths=[0.2]*4)
table.auto_set_font_size(False)
table.set_fontsize(12)
table.scale(1.2, 1.2)
plt.savefig('E:/498R/Code/Result/Dataset3/resultTable.png', format='png', dpi=300, bbox_inches='tight')
plt.show()

# Save the metrics
results_path = 'E:/498R/Code/Result/Dataset3/svm_metrics_simple.pkl'
with open(results_path, 'wb') as f:
    pickle.dump(metrics, f)

print("Metrics saved to:", results_path)

# 5. Confusion Matrix Visualization
def plot_confusion_matrix(true_labels, pred_labels, dataset_type):
    cm = confusion_matrix(true_labels, pred_labels)
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Fake', 'Real'], yticklabels=['Fake', 'Real'])
    plt.title(f'Confusion Matrix - {dataset_type}')
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.savefig(f'E:/498R/Code/Result/Dataset3/confusion_matrix_{dataset_type}.png')
    plt.show()

# Plot confusion matrices for test and validation sets
plot_confusion_matrix(test_labels, test_predictions, 'Test')
plot_confusion_matrix(val_labels, val_predictions, 'Validation')

# 6. Save the trained model using pickle
model_save_path = 'E:/498R/Code/Result/Dataset3/svm_model_simple.pkl'
with open(model_save_path, 'wb') as f:
    pickle.dump(clf, f)

print("SVM model saved to:", model_save_path)

# 7. Make a prediction on a new image
def predict_new_image(features, model_path='E:/498R/Code/Result/Dataset3/svm_model_simple.pkl'):
    """Load the saved model and predict the class of the new image's features."""
    with open(model_path, 'rb') as f:
        loaded_model = pickle.load(f)
    
    # Make a prediction
    prediction = loaded_model.predict(features)
    confidence_scores = loaded_model.predict_proba(features)
    
    predicted_label = 'Real' if prediction[0] == 1 else 'Fake'
    confidence = confidence_scores[0][prediction[0]]
    
    print(f"Prediction: {predicted_label}, Confidence: {confidence:.2f}")
    return predicted_label, confidence

# Example usage for prediction on a new image
# Assuming you already have the features for the new image
test_image = 'E:/498R/Code/Testingimage/r.jpg'
faceloading = Faceloading()
new_image_features = faceloading.extract_features_from_image(test_image)  # Replace with real feature extraction logic
predict_new_image(new_image_features)


