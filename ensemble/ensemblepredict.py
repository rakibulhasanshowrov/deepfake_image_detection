from imp import *
from dir import *
# Define the image size
image_size = (64, 64)

# Define the CustomVotingClassifier class
class CustomVotingClassifier:
    def __init__(self, cnn_model, rf_model, svm_model):
        self.cnn_model = cnn_model
        self.rf_model = rf_model
        self.svm_model = svm_model

    def predict_with_confidence(self, X):
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
        predicted_label = (avg_prob > 0.5).astype(int)
        
        # Get the confidence score (average probability)
        confidence_score = avg_prob[0]
        
        return predicted_label[0], confidence_score

    @classmethod
    def load(cls, file_path):
        with open(file_path, 'rb') as f:
            config = pickle.load(f)
        cnn_model = load_model(config['cnn_model_path'])
        rf_model = joblib.load(config['rf_model_path'])
        svm_model = joblib.load(config['svm_model_path'])
        return cls(cnn_model, rf_model, svm_model)

# Load models
cnn_model = load_model('cnn_model.h5')
rf_model = joblib.load('random_forest_model.pkl')
svm_model = joblib.load('svm_model.pkl')

# Load scaler
scaler = joblib.load('scaler.pkl')

# Load the custom ensemble model
ensemble_model = CustomVotingClassifier(cnn_model, rf_model, svm_model)

# Load and preprocess the image
def preprocess_image(image_path, target_size):
    # Load the image
    image = load_img(image_path, target_size=target_size)
    
    # Convert the image to a NumPy array
    image_array = img_to_array(image)
    
    # Rescale pixel values to [0, 1]
    image_array = image_array / 255.0
    
    # Add batch dimension
    image_array = np.expand_dims(image_array, axis=0)
    
    return image_array

# Define the path to your image
image_path = 'E:/498R/Dataset3_simple/Real/real_00001.jpg'

# Preprocess the image
img_preprocessed = preprocess_image(image_path, image_size)

# Flatten the preprocessed image for traditional models
img_flattened = img_preprocessed.reshape(img_preprocessed.shape[0], -1)

# Normalize the flattened image (Use the scaler used during training)
img_flattened = scaler.transform(img_flattened)

# Make prediction using the ensemble model
predicted_label, confidence_score = ensemble_model.predict_with_confidence((img_preprocessed, img_flattened))

# Output the result
print('Prediction:', 'Fake' if predicted_label == 1 else 'Real')
print('Confidence Score:', confidence_score)