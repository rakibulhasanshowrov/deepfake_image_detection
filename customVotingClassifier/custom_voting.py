import subprocess
import numpy as np

# Function to run external Python scripts for each model
def get_prediction_from_script(script_name, image_path):
    # Run the script and capture the output
    result = subprocess.run(['python', script_name, image_path], capture_output=True, text=True)
    prediction, confidence = result.stdout.strip().split(',')
    return int(prediction), float(confidence)

# Custom voting classifier
class CustomVotingClassifier:
    def __init__(self, model_scripts):
        self.model_scripts = model_scripts  # List of script names for each model

    def predict(self, image_path):
        predictions = []
        confidences = []

        # Loop through each model script and get predictions
        for script in self.model_scripts:
            prediction, confidence = get_prediction_from_script(script, image_path)
            predictions.append(prediction)
            confidences.append(confidence)

        # Custom voting logic (e.g., majority vote)
        final_prediction = max(set(predictions), key=predictions.count)  # Majority vote

        # Confidence score (average confidence from all models)
        avg_confidence = np.mean(confidences)

        return final_prediction, avg_confidence

# Example usage
if __name__ == "__main__":
    image_path = 'path_to_your_image.jpg'
    
    # List of scripts for EfficientNet, SVM, and Random Forest
    model_scripts = ['efficientnet_predict.py', 'svm_predict.py', 'rf_predict.py']

    # Create the voting classifier
    voting_classifier = CustomVotingClassifier(model_scripts)

    # Get the prediction and confidence score
    prediction, confidence = voting_classifier.predict(image_path)

    # Output result
    print(f"Prediction: {'Fake' if prediction == 0 else 'Real'}, Confidence: {confidence:.2f}")
