
from predict import Faceloading
import numpy as np
from sklearn import svm
from sklearn.metrics import confusion_matrix, accuracy_score, recall_score, f1_score, precision_score
import matplotlib.pyplot as plt
import seaborn as sns
import pickle

# Paths to the saved .npy files for features and labels
train_features_path = 'E:/498R/Code/Result/simple/Train/train_features.npy'
train_labels_path = 'E:/498R/Code/Result/simple/Train/train_labels.npy'

test_features_path = 'E:/498R/Code/Result/simple/Test/test_features.npy'
test_labels_path = 'E:/498R/Code/Result/simple/Test/test_labels.npy'

val_features_path = 'E:/498R/Code/Result/simple/Validation/validation_features.npy'
val_labels_path = 'E:/498R/Code/Result/simple/Validation/validation_labels.npy'

# Load the features and labels for train, test, and validation datasets
train_features = np.load(train_features_path)
train_labels = np.load(train_labels_path)

test_features = np.load(test_features_path)
test_labels = np.load(test_labels_path)

val_features = np.load(val_features_path)
val_labels = np.load(val_labels_path)

# Convert labels from 'fake'/'real' to 0/1 for binary classification
label_mapping = {'Fake': 0, 'Real': 1}
train_labels = np.array([label_mapping[label] for label in train_labels])
test_labels = np.array([label_mapping[label] for label in test_labels])
val_labels = np.array([label_mapping[label] for label in val_labels])

# 1. Train an SVM model
clf = svm.SVC(probability=True)  # Set probability=True to get confidence scores later
clf.fit(train_features, train_labels)

# 2. Evaluate the model
# Predict on the test set
test_predictions = clf.predict(test_features)

# Predict on the validation set
val_predictions = clf.predict(val_features)

# 3. Calculate and save metrics (Accuracy, Recall, F1, Precision) for test and validation sets
metrics = {}

# Calculate metrics for test set
metrics['test_accuracy'] = accuracy_score(test_labels, test_predictions)
metrics['test_recall'] = recall_score(test_labels, test_predictions)
metrics['test_precision'] = precision_score(test_labels, test_predictions)
metrics['test_f1'] = f1_score(test_labels, test_predictions)

# Calculate metrics for validation set
metrics['val_accuracy'] = accuracy_score(val_labels, val_predictions)
metrics['val_recall'] = recall_score(val_labels, val_predictions)
metrics['val_precision'] = precision_score(val_labels, val_predictions)
metrics['val_f1'] = f1_score(val_labels, val_predictions)

# Save the results
results_path = 'E:/498R/Code/Result/simple/svm_metrics.pkl'
with open(results_path, 'wb') as f:
    pickle.dump(metrics, f)

print("Metrics saved to:", results_path)

# 4. Confusion Matrix Visualization
def plot_confusion_matrix(true_labels, pred_labels, dataset_type):
    cm = confusion_matrix(true_labels, pred_labels)
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Fake', 'Real'], yticklabels=['Fake', 'Real'])
    plt.title(f'Confusion Matrix - {dataset_type}')
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.savefig(f'E:/498R/Code/Result/simple/confusion_matrix_{dataset_type}.png')
    plt.show()

# Plot confusion matrix for test and validation sets
plot_confusion_matrix(test_labels, test_predictions, 'Test')
plot_confusion_matrix(val_labels, val_predictions, 'Validation')

# 5. Save the trained model using pickle
model_save_path = 'E:/498R/Code/Result/simple/svm_model.pkl'
with open(model_save_path, 'wb') as f:
    pickle.dump(clf, f)

print("SVM model saved to:", model_save_path)

# 6. Make a prediction on a new image
def predict_new_image(features, model_path='E:/498R/Code/Result/simple/svm_model.pkl'):
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
# Assuming you already have the features for the new image (since face extraction is done already)
# For example:
# new_image_features = np.load('path_to_new_image_features.npy')  # Replace with actual features array

# Just replace the path to the saved features array if needed
test_image='E:/498R/Code/Testingimage/r.jpg'
faceloading=Faceloading()
new_image_features = faceloading.extract_features_from_image(test_image)  # Replace with real features from your system
predict_new_image(new_image_features)

