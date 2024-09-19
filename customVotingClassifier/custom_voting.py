import subprocess
import numpy as np
import os

# Function to run external Python scripts for each model
def run_script_and_wait(script_name, image_path, output_file):
    """Runs the script without expecting output directly, but waits for it to finish and saves results in an output file."""
    # Run the script with subprocess and pass the image path
    result = subprocess.run(['python', script_name, image_path, output_file], capture_output=True, text=True, encoding='utf-8')

    # Check for any errors in the subprocess
    if result.stderr:
        print(f"Subprocess encountered an error in {script_name}: {result.stderr}")

# Function to retrieve the result from an output file
def get_result_from_file(output_file):
    """Reads the prediction and confidence score from a file."""
    if os.path.exists(output_file):
        with open(output_file, 'r') as file:
            content = file.read().strip()
            if ',' in content:
                prediction, confidence = content.split(',')
                return int(prediction), float(confidence)
            else:
                print(f"Error: Invalid format in output file {output_file}")
    else:
        print(f"Error: Output file {output_file} not found")
    
    return None, None

# Custom voting classifier
class CustomVotingClassifier:
    def __init__(self, model_scripts):
        self.model_scripts = model_scripts  # List of script names for each model

    def predict(self, image_path):
        predictions = []
        confidences = []
        output_files = []

        # Loop through each model script, run it, and retrieve results from the file
        for i, script in enumerate(self.model_scripts):
            output_file = f'output_{i}.txt'  # Generate a unique output file for each model
            output_files.append(output_file)

            # Run the prediction script (which will save results to the file)
            run_script_and_wait(script, image_path, output_file)

        # Now retrieve predictions from output files
        for output_file in output_files:
            prediction, confidence = get_result_from_file(output_file)
            if prediction is not None and confidence is not None:
                predictions.append(prediction)
                confidences.append(confidence)

        # Custom voting logic (e.g., majority vote)
        final_prediction = max(set(predictions), key=predictions.count)  # Majority vote

        # Confidence score (average confidence from all models)
        avg_confidence = np.mean(confidences)

        return final_prediction, avg_confidence

# Example usage
if __name__ == "__main__":
    image_path = 'E:/498R/Code/Testingimage/pr.jpg'
    
    # List of scripts for EfficientNet, SVM, and Random Forest
    model_scripts = ['efficientnet_predict.py', 'svm_predict.py', 'rf_predict.py']

    # Create the voting classifier
    voting_classifier = CustomVotingClassifier(model_scripts)

    # Get the prediction and confidence score
    prediction, confidence = voting_classifier.predict(image_path)

    # Output result
    print(f"Prediction: {'Fake' if prediction == 0 else 'Real'}, Confidence: {confidence:.2f}")

