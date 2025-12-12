# CNN for MNIST Digit Classification

This document describes the `cnn.py` script, which implements a Convolutional Neural Network (CNN) for classifying digits from the MNIST dataset using TensorFlow and Keras.

## Overview

The script demonstrates a complete machine learning workflow:
1.  **Dataset Loading**: Loads the MNIST dataset of handwritten digits.
2.  **Data Preprocessing**: Reshapes and normalizes the image data.
3.  **Model Definition**: Constructs a sequential CNN model.
4.  **Model Compilation**: Configures the model for training with an optimizer, loss function, and metrics.
5.  **Model Training**: Trains the CNN on the training data.
6.  **Model Evaluation**: Evaluates the model's performance on the test data.
7.  **Prediction**: Demonstrates how to make a prediction on a single sample.

## Prerequisites

To run this script, you need to have TensorFlow installed:

```bash
pip install tensorflow
```

## Model Architecture

The CNN model is a `Sequential` model consisting of the following layers:

*   **Conv2D Layer 1**: 32 filters, 3x3 kernel, ReLU activation, input shape `(28, 28, 1)`.
*   **MaxPooling2D Layer 1**: 2x2 pooling filter.
*   **Conv2D Layer 2**: 64 filters, 3x3 kernel, ReLU activation.
*   **MaxPooling2D Layer 2**: 2x2 pooling filter.
*   **Flatten Layer**: Flattens the output from the convolutional layers into a 1D vector.
*   **Dense Layer 1**: 128 units, ReLU activation.
*   **Dense Layer 2 (Output)**: 10 units (for 10 digit classes), Softmax activation.

## Data Preprocessing

The MNIST dataset images are 28x28 grayscale, so they are reshaped to `(-1, 28, 28, 1)` to add a channel dimension. The pixel values are then normalized to the range [0, 1] by dividing by 255.0.

## Training and Evaluation

The model is compiled with:
*   **Optimizer**: `'adam'`
*   **Loss Function**: `'sparse_categorical_crossentropy'` (suitable for integer labels in multi-class classification)
*   **Metrics**: `'accuracy'`

The model is trained for 5 epochs with a batch size of 64 using the `x_train` and `y_train` datasets. After training, its accuracy is evaluated on the `x_test` and `y_test` datasets.

## Example Usage

To run the script, save the code below as `cnn.py` and execute it from your terminal:

```bash
python cnn.py
```

### Output Example

After training for 5 epochs, you will see output similar to this (exact accuracy may vary):

```
...
Epoch 5/5
938/938 [==============================] - 14s 15ms/step - loss: 0.0402 - accuracy: 0.9877
1875/1875 [==============================] - 2s 1ms/step - loss: 0.0336 - accuracy: 0.9882
Test Accuracy: 0.9882000088691711
1/1 [==============================] - 0s 54ms/step
Predicted digit: 7
```

## `cnn.py` Source Code

```python
# CNN for MNIST Digit Classification
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

# Load dataset
mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Reshape (samples, height, width, channels)
x_train = x_train.reshape(-1, 28, 28, 1) / 255.0
x_test = x_test.reshape(-1, 28, 28, 1) / 255.0

# Build CNN model
model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(28, 28, 1)),
    MaxPooling2D((2,2)),
    
    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D((2,2)),
    
    Flatten(),
    Dense(128, activation='relu'),
    Dense(10, activation='softmax')   # 10 categories (0–9)
])

# Compile model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Train model
model.fit(x_train, y_train, epochs=5, batch_size=64)

# Evaluate accuracy
test_loss, test_acc = model.evaluate(x_test, y_test)
print("Test Accuracy:", test_acc)

# Make prediction on a sample image
pred = model.predict(x_test[:1])
print("Predicted digit:", pred.argmax())
```