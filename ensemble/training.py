from imageprocessing import *
from cnn import *


# Alternatively, load all arrays from the compressed file
data = np.load('dataset_split.npz')
X_train = data['X_train']
X_val = data['X_val']
X_test = data['X_test']

y_train = data['y_train']
y_val = data['y_val']
y_test = data['y_test']

print("Arrays loaded successfully.")

# Create and train the CNN model
cnn_model = create_cnn_model()
cnn_model.fit(X_train, y_train, epochs=10, batch_size=32, verbose=0)

# Create Random Forest and SVM models
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
svm_model = SVC(kernel='linear', probability=True)

# Flatten images for traditional ML models
X_train_flattened = X_train.reshape(X_train.shape[0], -1)
X_val_flattened = X_val.reshape(X_val.shape[0], -1)
X_test_flattened = X_test.reshape(X_test.shape[0], -1)

# Train Random Forest and SVM models
rf_model.fit(X_train_flattened, y_train)
svm_model.fit(X_train_flattened, y_train)

# Save individual models
joblib.dump(rf_model, 'random_forest_model.pkl')
joblib.dump(svm_model, 'svm_model.pkl')

# Save the Keras CNN model
cnn_model.save('cnn_model.h5')

# Function to generate results for a dataset
def evaluate_model(model, X, y_true, dataset_type='Test'):
    y_pred = model.predict(X)
    accuracy = accuracy_score(y_true, y_pred)
    report = classification_report(y_true, y_pred, output_dict=True)
    # Handle dynamic class labels
    classes = list(report.keys())
    metrics = report[classes[1]] if len(classes) > 1 else report[classes[0]]
    return {
        'Dataset': dataset_type,
        'Accuracy': accuracy,
        'Precision': metrics['precision'],
        'Recall': metrics['recall'],
        'F1-Score': metrics['f1-score']
    }

# Function to evaluate CNN model
def evaluate_cnn_model(cnn_model, X, y_true, dataset_type='Test'):
    y_pred = (cnn_model.predict(X) > 0.5).astype(int).flatten()
    accuracy = accuracy_score(y_true, y_pred)
    report = classification_report(y_true, y_pred, output_dict=True)
    # Handle dynamic class labels
    classes = list(report.keys())
    metrics = report[classes[1]] if len(classes) > 1 else report[classes[0]]
    return {
        'Dataset': dataset_type,
        'Accuracy': accuracy,
        'Precision': metrics['precision'],
        'Recall': metrics['recall'],
        'F1-Score': metrics['f1-score']
    }

# Evaluate individual models
train_results_cnn = evaluate_cnn_model(cnn_model, X_train, y_train, dataset_type='Train')
val_results_cnn = evaluate_cnn_model(cnn_model, X_val, y_val, dataset_type='Validation')
test_results_cnn = evaluate_cnn_model(cnn_model, X_test, y_test, dataset_type='Test')

train_results_rf = evaluate_model(rf_model, X_train_flattened, y_train, dataset_type='Train')
val_results_rf = evaluate_model(rf_model, X_val_flattened, y_val, dataset_type='Validation')
test_results_rf = evaluate_model(rf_model, X_test_flattened, y_test, dataset_type='Test')

train_results_svm = evaluate_model(svm_model, X_train_flattened, y_train, dataset_type='Train')
val_results_svm = evaluate_model(svm_model, X_val_flattened, y_val, dataset_type='Validation')
test_results_svm = evaluate_model(svm_model, X_test_flattened, y_test, dataset_type='Test')

# Collect all the results in a DataFrame
results_df = pd.DataFrame([
    train_results_cnn, val_results_cnn, test_results_cnn,
    train_results_rf, val_results_rf, test_results_rf,
    train_results_svm, val_results_svm, test_results_svm
])

# Save the results to a CSV file
results_df.to_csv('model_evaluation_results.csv', index=False)

# Print the results for verification
print(results_df)