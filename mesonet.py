import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, recall_score, f1_score, precision_score
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# # Path to your dataset full
# train_dir = 'E:/498R/dataset/Train'
# validation_dir = 'E:/498R/dataset/Validation/'
# test_dir = 'E:/498R/dataset/Test/'
# output_dir='E:/498R/result/full/'
Path to your dataset simple
train_dir = 'E:/498R/dataset_simple/Train'
validation_dir = 'E:/498R/dataset_simple/Validation/'
test_dir = 'E:/498R/dataset_simple/Test/'
output_dir='E:/498R/result/simple/'
output_dir='E:/498R/result/simple/'

# Parameters
IMG_SIZE = (256, 256)  # Slightly bigger image size for MesoNet
BATCH_SIZE = 32
EPOCHS = 20
LEARNING_RATE = 0.0001

# Data Augmentation and Preprocessing
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

validation_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)

# Load the dataset from directory
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='binary'
)

validation_generator = validation_datagen.flow_from_directory(
    validation_dir,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='binary'
)

test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='binary',
    shuffle=False  # Do not shuffle to evaluate the results properly
)

# MesoNet model definition
def MesoNet():
    model = Sequential()

    # First convolutional block
    model.add(Conv2D(8, (3, 3), padding='same', activation='relu', input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3)))
    model.add(Conv2D(8, (3, 3), padding='same', activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))

    # Second convolutional block
    model.add(Conv2D(16, (5, 5), padding='same', activation='relu'))
    model.add(Conv2D(16, (5, 5), padding='same', activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))

    # Flatten and Fully Connected layers
    model.add(Flatten())

    model.add(Dense(16, activation='relu'))
    model.add(Dropout(0.5))  # Dropout for regularization

    model.add(Dense(1, activation='sigmoid'))  # Binary classification

    return model

# Build and compile the MesoNet model
model = MesoNet()
model.compile(
    optimizer = Adam(learning_rate=LEARNING_RATE),
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# Print the model summary
model.summary()

# Train the model
history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // BATCH_SIZE,
    epochs=EPOCHS,
    validation_data=validation_generator,
    validation_steps=validation_generator.samples // BATCH_SIZE
)

# Function to evaluate model and display confusion matrix and metrics
def evaluate_model(generator, dataset_name):
    # Predict labels
    y_true = generator.classes
    y_pred = (model.predict(generator, steps=len(generator)) > 0.5).astype("int32").flatten()

    # Calculate confusion matrix
    conf_matrix = confusion_matrix(y_true, y_pred)

    # Plot confusion matrix
    plt.figure(figsize=(6, 6))
    sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=['Fake', 'Real'], yticklabels=['Fake', 'Real'])
    plt.title(f'Confusion Matrix - {dataset_name}')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    #plt.show(block=False)
     # Save plot as an image
    plot_path = os.path.join(output_dir, f'confusion_matrix_{dataset_name}.png')
    plt.savefig(plot_path)
    plt.close()

    # Classification metrics
    acc = accuracy_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)

    print(f"--- {dataset_name} Set Metrics ---")
    print(f"Accuracy: {acc * 100:.2f}%")
    print(f"Recall: {recall * 100:.2f}%")
    print(f"Precision: {precision * 100:.2f}%")
    print(f"F1-Score: {f1 * 100:.2f}%")
    print("\nClassification Report:\n", classification_report(y_true, y_pred, target_names=['Fake', 'Real']))

# Evaluate on training set
evaluate_model(train_generator, "Training")

# Evaluate on validation set
evaluate_model(validation_generator, "Validation")

# Evaluate on test set
evaluate_model(test_generator, "Test")

# Save the model
model.save('mesonet_deepfake_detector_model_big.h5')
