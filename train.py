# Import packages for data manipulation
import numpy as np
import pandas as pd
import random
import pickle

# Import packages for data visualization
import matplotlib.pyplot as plt
import seaborn as sns

# Import packages for building the model
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (Input, Conv2D, MaxPooling2D, Dropout, Dense, Flatten)

# Import packages for data preprocessing and model performance
import cv2
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             confusion_matrix, ConfusionMatrixDisplay, classification_report)

# Clone the Data
!git clone https://bitbucket.org/jadslim/german-traffic-signs

# Get a list of available files
!ls german-traffic-signs

# Unpickle the training set
with open('german-traffic-signs/train.p', 'rb') as train:
  train_data = pickle.load(train)

# Unpickle the validation set
with open('german-traffic-signs/valid.p', 'rb') as val:
  val_data = pickle.load(val)

# Unpickle the test set
with open('german-traffic-signs/test.p', 'rb') as test:
  test_data = pickle.load(test)

# Separate the features and target
X_train, y_train = train_data['features'], train_data['labels']
X_val, y_val = val_data['features'], val_data['labels']
X_test, y_test = test_data['features'], test_data['labels']

# Store the class names in a list
sign_names = pd.read_csv('/content/german-traffic-signs/signnames.csv')
class_names = list(sign_names['SignName'])

# Iterate through the sign names and their associated class
names_dict = dict()
for index, name in enumerate(class_names):
  names_dict[index] = name
names_dict

# Display the shape of the datasets
print(f'Shape of the Training set: {X_train.shape, y_train.shape}')
print(f'Shape of the Validation set: {X_val.shape, y_val.shape}')
print(f'Shape of the Test set: {X_test.shape, y_test.shape}')

# Display 25 random images from the training set and display the class name below each image
plt.figure(figsize=(10,10))
for i in range(25):
  plt.subplot(5,5,i+1)
  idx = random.randint(0, len(X_train) - 1)
  plt.xticks([])
  plt.yticks([])
  plt.grid(False)
  plt.imshow(X_train[idx])
  plt.xlabel(class_names[y_train[idx]])
plt.tight_layout()
plt.show();

# Plot the distribution of the classes
plt.title('Distribution of Classes')
plt.xlabel('Sign Classes')
plt.ylabel('Frequency')
sns.countplot(x=y_train)
plt.xticks(ticks=range(len(class_names)), labels=class_names, rotation=90)
plt.show();

# Function for preprocessing the images
def preprocess(img):
  img = cv2.resize(img, (32, 32))
  img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
  img = cv2.equalizeHist(img)
  img = np.divide(img, 255)
  return img

# Apply the preprocessing to the images
X_train = np.array(list(map(preprocess, X_train)))
X_val = np.array(list(map(preprocess, X_val)))
X_test = np.array(list(map(preprocess, X_test)))

# Display a sample of the preprocessed images
idx = random.randint(0, len(X_train) - 1)
plt.imshow(X_train[idx])
plt.title(class_names[y_train[idx]])
plt.show();

# Reshape train, validation and test images to add a channel dimension
X_train = np.expand_dims(X_train, axis=-1)
X_val = np.expand_dims(X_val, axis=-1)
X_test = np.expand_dims(X_test, axis=-1)

print(f'Shape of the Training-set after reshaping: {X_train.shape}')
print(f'Shape of the Validation-set after reshaping: {X_val.shape}')
print(f'Shape of the Testing-set after reshaping: {X_test.shape}')

# Make sure all sets have a consistent target shape
len(np.unique(y_train)) == len(np.unique(y_val)) == len(np.unique(y_test))

# One-Hot Encode the target variable
n_classes = len(np.unique(y_train))

y_train = tf.keras.utils.to_categorical(y_train, n_classes)
y_val = tf.keras.utils.to_categorical(y_val, n_classes)
y_test = tf.keras.utils.to_categorical(y_test, n_classes)

# Function to plot images in the form of a grid with 1 row and 5 columns where images are placed in each column
def plotImages(images_arr):
  fig, axes = plt.subplots(1, 5)
  axes = axes.flatten()
  for img, ax in zip(images_arr, axes):
    ax.imshow(img, cmap=plt.cm.binary)
  plt.tight_layout()
  plt.show();

# Create a transformation object
datagen = ImageDataGenerator(
    rotation_range=10,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.1,
    zoom_range=0.1,
    horizontal_flip=False,
    vertical_flip=False,
    fill_mode='nearest'
)

# Apply the transformations to the training set
datagen.fit(X_train)

# Generate a batch of augmented images
augmented_images = [datagen.random_transform(X_train[random.randint(0, len(X_train) - 1)]) for i in range(5)]

# Plot the augmented images
plotImages(augmented_images)

# Define the parameters
shape = X_train.shape[1:]
classes = n_classes

# Build the model
def neural_network(input_shape, n_classes):

  # Input layer
  X_input = Input(input_shape)

  # First conv layer
  X = Conv2D(108, kernel_size=(5, 5), activation='relu')(X_input)
  X = MaxPooling2D(pool_size=(2, 2))(X)
  X = Dropout(0.15)(X)

  # Second conv layer
  X = Conv2D(108, kernel_size=(5, 5), activation='relu')(X)
  X = MaxPooling2D(pool_size=(2, 2))(X)
  X = Dropout(0.2)(X)

  # FC & Output layers
  X = Flatten()(X)
  X = Dense(100, activation='relu')(X)
  X = Dense(100, activation='relu')(X)
  X = Dense(n_classes, activation='softmax')(X)

  # Build the model
  model = Model(inputs=X_input, outputs=X, name='Network')
  return model

# Call the model
model = neural_network(shape, classes)

# Compile the model
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Display the number of parameters
model.summary()

# Train the model using the generator
epochs = 15
history = model.fit(datagen.flow(X_train, y_train, batch_size=128),
                    validation_data=(X_val, y_val),
                    epochs=epochs, shuffle=True)

# Extract the loss and accuracy from the dictionary and plot them
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(epochs)

plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy', color='blue')
plt.plot(epochs_range, val_acc, label='Validation Accuracy', color='cyan')
plt.legend()
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('Training vs. Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss', color='blue')
plt.plot(epochs_range, val_loss, label='Validation Loss', color='cyan')
plt.legend()
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training vs. Validation Loss')
plt.tight_layout()
plt.show();

# Function to display the predicted images
def display_predictions(model, X_data, y_data, n=None, color=None):

  # Make predictions
  predicted_labels = model.predict(X_data)

  # Converts predictions & true labels to class indices
  predicted_labels = np.argmax(predicted_labels, axis=1)
  true_labels = np.argmax(y_data, axis=1)

  plt.figure(figsize=(10, 10))
  for i in range(n):
    plt.subplot(5,5,i+1)
    idx = random.randint(0, len(X_data) - 1)
    plt.xticks([])
    plt.yticks([])
    plt.imshow(X_data[idx], cmap=color)
    true_label = true_labels[idx]
    predicted_label = predicted_labels[idx]
    plt.title(f'True: {true_label}, Pred: {predicted_label}')

  plt.tight_layout()
  plt.show();

# Display a few couple of predictions from the validation set
display_predictions(model, X_val, y_val, n=10, color='rainbow')

# Evaluate the model on the test data
loss, accuracy = model.evaluate(X_test, y_test)

print(f'Accuracy on test dataset: {accuracy:.4}')
print(f'Loss on test dataset: {loss:.4}')

# Initialize a list to track correctness
correct_predictions = []

# Make predictions on the test data
predictions = model.predict(X_test)

# Display the first 10 random predictions along with their true labels
n = 10
for i in range(n):
    y_pred, y_true = np.argmax(predictions[i]), np.argmax(y_test[i])
    correct_predictions.append(y_pred == y_true)
    print(f'Pred: {class_names[y_pred]} , Actual: {class_names[y_true]}')

# Calculate accuracy for the first 10 images
accuracy = np.mean(correct_predictions) * 100
print('--------------------------------------------------------')
print(f'The Accuracy on the first {n} images: {accuracy}%')

# Convert ground truth & predictions from one-hot encoding to class indices
predictions = np.argmax(predictions, axis=1)
ground_truth = np.argmax(y_test, axis=1)

# Compute the number of wrong predictions
num_wrong_predictions = np.sum(predictions != ground_truth)

print(f'Number of wrong predictions: {num_wrong_predictions}')

# Create a confusion matrix to analyze the performance of the model
cm = confusion_matrix(ground_truth, predictions)

# Normalize the confusion matrix to show percentages
cm_percentage = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100

# Plot the confusion matrix
plt.figure(figsize=(22, 12))
sns.heatmap(cm_percentage, annot=True, fmt='.1f', cmap='Purples',
            xticklabels=class_names, yticklabels=class_names)

plt.title('Confusion Matrix (Normalized)', fontsize=24)
plt.xlabel('Predicted Labels', fontsize=18)
plt.ylabel('True Labels', fontsize=18)
plt.xticks(rotation=90)
plt.yticks(rotation=0)
plt.show();

# Get a report of metrics on different classes
print(classification_report(ground_truth, predictions,
                            target_names=class_names))

# Call the function to display predicted test images
display_predictions(model, X_test, y_test, n=15, color='coolwarm')

# Save the model
clf = model.save('model.h5')