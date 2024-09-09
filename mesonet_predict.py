import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
import matplotlib.pyplot as plt

# Load the saved model
model = tf.keras.models.load_model('mesonet_deepfake_detector_model.h5')

def preprocess_image(img_path, target_size):
    # Load the image
    img = image.load_img(img_path, target_size=target_size)
    # Convert the image to a numpy array
    img_array = image.img_to_array(img)
    # Expand dimensions to match the input shape of the model
    img_array = np.expand_dims(img_array, axis=0)
    # Preprocess the image (e.g., normalize)
    img_array = img_array / 255.0  # Example normalization
    return img_array

def predict_image(img_path):
    # Define the target size (this should match the input size of your model)
    target_size = (256, 256)  # Example size, adjust if necessary

    # Preprocess the image
    img_array = preprocess_image(img_path, target_size)
    
    # Make a prediction
    prediction = model.predict(img_array)
     # Print prediction
    print(f"Raw prediction: {prediction}")
    
    # Get the class labels
    class_labels = ['Fake', 'Real']
    
    # Get the class with the highest probability and its confidence score
    confidence_scores = prediction[0]
    predicted_class_index = np.argmax(confidence_scores)
    predicted_class = class_labels[predicted_class_index]
    confidence_score = confidence_scores[predicted_class_index]

    # Print the result
    print(f"Predicted class: {predicted_class}")
    print(f"Confidence score: {confidence_score * 100:.2f}%")
    print(f"Prediction probabilities: {confidence_scores}")

    # Display the image
    img = image.load_img(img_path)
    plt.imshow(img)
    plt.title(f'Prediction: {predicted_class} ({confidence_score * 100:.2f}%)')
    plt.axis('off')
    plt.show()

# Example usage
img_path = 'E:/498R/Testingimage/r.jpg'
predict_image(img_path)

img_path = 'E:/498R/Testingimage/f.jpg'
predict_image(img_path)
