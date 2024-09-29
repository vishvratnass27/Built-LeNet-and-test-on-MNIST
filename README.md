This repository contains a Convolutional Neural Network (CNN) implementation in Python using the Keras library. The model is trained on the MNIST dataset, a collection of 60,000 28x28 grayscale images of handwritten digits, ranging from 0 to 9. The task is to classify these images into one of 10 classes (0-9).
:

# Imports:

Libraries like keras.models.Sequential, Conv2D, Dense, MaxPooling2D, etc., are used to create and train the CNN.

mnist dataset is loaded directly from Keras.

# Data Preprocessing:

The MNIST dataset contains images of digits (0-9), each of size 28x28 pixels.

The dataset is reshaped from (60000, 28, 28) to (60000, 28, 28, 1) to add a 4th dimension, as required by Keras for processing images (the last 1 indicates that the images are grayscale).

Data is normalized by dividing the pixel values by 255, changing the range from [0, 255] to [0, 1], which speeds up the training process.

Labels are one-hot encoded using to_categorical (e.g., label 2 becomes [0, 0, 1, 0, 0, 0, 0, 0, 0, 0]).

# Model Architecture:

The model is built using the Sequential API.

# First Convolutional Layer:

A Conv2D layer with 20 filters of size (5x5), using "same" padding to preserve image dimensions.

ReLU activation is used to introduce non-linearity.

A MaxPooling layer with a 2x2 pool size is added for down-sampling.

# Second Convolutional Layer:

A Conv2D layer with 50 filters of size (5x5) followed by ReLU activation.

MaxPooling is used again to down-sample the image.

# Fully Connected Layer:
The model is flattened (converted from 2D to 1D) and passed through a Dense layer with 500 units.

ReLU activation is applied.

# Output Layer:

A Dense layer with num_classes (10 for MNIST) and Softmax activation, which converts the output into probability distributions across the 10 classes.

# Compiling the Model:

The model uses categorical_crossentropy as the loss function (suitable for multi-class classification problems).

The optimizer used is Adam, a widely-used adaptive learning rate optimization algorithm.

The metric used to monitor performance is accuracy.

# Training the Model:

The model is trained on the training data (x_train, y_train) for 10 epochs with a batch size of 128.

Validation is performed using the test data (x_test, y_test) to monitor performance during training.

# Saving the Model:

The trained model is saved to a file named mnist_LeNet.h5 for future use.

# Evaluating the Model:

The test data is used to evaluate the model's performance, and both the test loss and accuracy are printed.
