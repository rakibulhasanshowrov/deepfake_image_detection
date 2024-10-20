import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score

# Main function for training and evaluating the model
def main():
    # Load preprocessed data
    X = np.load('X.npy')
    y = np.load('y.npy')

    # Split data into training, validation, and testing sets
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

    # Train Random Forest model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Save the model
    with open('random_forest_model.pkl', 'wb') as model_file:
        pickle.dump(model, model_file)
    
    # Save the split datasets
    np.save('X_train.npy', X_train)
    np.save('X_val.npy', X_val)
    np.save('X_test.npy', X_test)
    np.save('y_train.npy', y_train)
    np.save('y_val.npy', y_val)
    np.save('y_test.npy', y_test)

    # Evaluate the model on the validation set
    y_pred = model.predict(X_val)
    accuracy = accuracy_score(y_val, y_pred)
    precision = precision_score(y_val, y_pred)
    recall = recall_score(y_val, y_pred)
    f1 = f1_score(y_val, y_pred)

    # Classification report
    report = classification_report(y_val, y_pred, target_names=['Fake', 'Real'])

    # Print evaluation report
    print("Classification Report:\n", report)
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")

    # Save evaluation metrics to a file
    with open('evaluation_report.txt', 'w') as report_file:
        report_file.write("Classification Report:\n")
        report_file.write(report + "\n")
        report_file.write(f"Accuracy: {accuracy:.4f}\n")
        report_file.write(f"Precision: {precision:.4f}\n")
        report_file.write(f"Recall: {recall:.4f}\n")
        report_file.write(f"F1 Score: {f1:.4f}\n")

if __name__ == "__main__":
    main()

