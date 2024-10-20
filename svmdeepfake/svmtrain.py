import numpy as np
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, classification_report
from sklearn.preprocessing import StandardScaler
from joblib import dump
import pickle

# Load saved features and labels
features = np.load('features.npy')
labels = np.load('labels.npy')

# Normalize the data
scaler = StandardScaler()
features = scaler.fit_transform(features)

# Save the StandardScaler for future use
dump(scaler, 'scaler.joblib')

# Split dataset into training, validation, and test sets
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state=42)

# Train SVM classifier
clf = svm.SVC(kernel='linear', probability=True)
clf.fit(X_train, y_train)

# Save the trained model
with open('svm_model.pkl', 'wb') as model_file:
    pickle.dump(clf, model_file)

# Evaluate the model
y_pred_test = clf.predict(X_test)
y_pred_val = clf.predict(X_val)

# Calculate and display evaluation metrics
metrics = {
    'Accuracy': accuracy_score(y_test, y_pred_test),
    'Recall': recall_score(y_test, y_pred_test),
    'Precision': precision_score(y_test, y_pred_test),
    'F1 Score': f1_score(y_test, y_pred_test)
}

for metric, value in metrics.items():
    print(f"{metric}: {value:.4f}")

# Detailed classification report
print("\nClassification Report (Test Set):")
report = classification_report(y_test, y_pred_test)
print(report)

# Save the report in a text file
with open('svm_classification_report.txt', 'w') as report_file:
    report_file.write(report)
