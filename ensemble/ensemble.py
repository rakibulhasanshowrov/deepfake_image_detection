from imp import *
from dir import *
# Load saved models
cnn_model = load_model('cnn_model.h5')
rf_model = joblib.load('random_forest_model.pkl')
svm_model = joblib.load('svm_model.pkl')

# Load saved data
data = np.load('dataset_split.npz')
X_train = data['X_train']
X_val = data['X_val']
X_test = data['X_test']
y_train = data['y_train']
y_val = data['y_val']
y_test = data['y_test']

# Flatten images for traditional ML models
X_train_flattened = X_train.reshape(X_train.shape[0], -1)
X_val_flattened = X_val.reshape(X_val.shape[0], -1)
X_test_flattened = X_test.reshape(X_test.shape[0], -1)

# Normalize data for RF and SVM
scaler = StandardScaler()
X_train_flattened = scaler.fit_transform(X_train_flattened)
# Save the scaler
joblib.dump(scaler, 'scaler.pkl')
X_train_flattened = scaler.fit_transform(X_train_flattened)
X_val_flattened = scaler.transform(X_val_flattened)
X_test_flattened = scaler.transform(X_test_flattened)

# Define the Custom Voting Classifier
class CustomVotingClassifier:
    def __init__(self, cnn_model, rf_model, svm_model):
        self.cnn_model = cnn_model
        self.rf_model = rf_model
        self.svm_model = svm_model

    def predict(self, X):
        # Unpack inputs
        X_images, X_flattened = X

        # Get CNN predictions
        cnn_prob = self.cnn_model.predict(X_images)
        
        # Get RF and SVM predictions
        rf_prob = self.rf_model.predict_proba(X_flattened)[:, 1]
        svm_prob = self.svm_model.predict_proba(X_flattened)[:, 1]
        
        # Combine predictions using average
        avg_prob = (cnn_prob.flatten() + rf_prob + svm_prob) / 3
        
        # Convert probabilities to class labels
        return (avg_prob > 0.5).astype(int)

    def save(self, file_path):
        with open(file_path, 'wb') as f:
            pickle.dump({
                'cnn_model_path': 'cnn_model.h5',
                'rf_model_path': 'random_forest_model.pkl',
                'svm_model_path': 'svm_model.pkl'
            }, f)

    @classmethod
    def load(cls, file_path):
        with open(file_path, 'rb') as f:
            config = pickle.load(f)
        cnn_model = load_model(config['cnn_model_path'])
        rf_model = joblib.load(config['rf_model_path'])
        svm_model = joblib.load(config['svm_model_path'])
        return cls(cnn_model, rf_model, svm_model)

# Instantiate and use the CustomVotingClassifier
ensemble_model = CustomVotingClassifier(cnn_model, rf_model, svm_model)

# Make predictions with the ensemble model
ensemble_predictions_train = ensemble_model.predict((X_train, X_train_flattened))
ensemble_predictions_val = ensemble_model.predict((X_val, X_val_flattened))
ensemble_predictions_test = ensemble_model.predict((X_test, X_test_flattened))

# Function to generate results for a dataset
def evaluate_model(y_true, y_pred, dataset_type='Test'):
    accuracy = accuracy_score(y_true, y_pred)
    report = classification_report(y_true, y_pred, output_dict=True)
    classes = list(report.keys())
    metrics = report[classes[1]] if len(classes) > 1 else report[classes[0]]
    return {
        'Dataset': dataset_type,
        'Accuracy': accuracy,
        'Precision': metrics['precision'],
        'Recall': metrics['recall'],
        'F1-Score': metrics['f1-score']
    }

# Evaluate the ensemble model
train_results_ensemble = evaluate_model(y_train, ensemble_predictions_train, dataset_type='Train')
val_results_ensemble = evaluate_model(y_val, ensemble_predictions_val, dataset_type='Validation')
test_results_ensemble = evaluate_model(y_test, ensemble_predictions_test, dataset_type='Test')

# Collect all the results in a DataFrame
results_df = pd.DataFrame([
    train_results_ensemble,
    val_results_ensemble,
    test_results_ensemble
])

# Save the results to a CSV file
results_df.to_csv('ensemble_model_evaluation_results.csv', index=False)

# Save the ensemble model
ensemble_model.save('custom_voting_classifier.pkl')

# Print the results for verification
print(results_df)