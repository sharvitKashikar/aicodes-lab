# Convolutional Neural Network (CNN) for MNIST Digit Classification

This example demonstrates how to build and train a Convolutional Neural Network (CNN) using TensorFlow/Keras to classify handwritten digits from the MNIST dataset.

## CNN Overview

CNNs are a class of deep neural networks most commonly applied to analyzing visual imagery. They are particularly effective for tasks like image classification, object detection, and image segmentation due to their ability to automatically learn spatial hierarchies of features through convolutional layers.

## Code Explanation

1.  **Imports**: Uses `tensorflow` and `tensorflow.keras` for building and training the neural network.
2.  **Dataset Loading**: The `mnist` dataset (handwritten digits) is loaded using `tf.keras.datasets.mnist`. Data is split into training and testing sets.
3.  **Data Preprocessing**: 
    *   Images are reshaped from `(samples, height, width)` to `(samples, height, width, channels)` for CNN input (adding a channel dimension for grayscale images).
    *   Pixel values are normalized to the range `[0, 1]` by dividing by 255.0.
4.  **Model Building (`Sequential` API)**:
    *   `Conv2D`: Applies convolution filters to the input, followed by a ReLU activation.
    *   `MaxPooling2D`: Downsamples the feature maps, reducing dimensionality and making the model more robust to small shifts in image features.
    *   `Flatten`: Converts the 2D feature maps into a 1D vector to be fed into dense layers.
    *   `Dense`: Standard fully connected neural network layers. The final `Dense` layer has 10 units (for 10 digit classes) and a `softmax` activation for multi-class classification.
5.  **Model Compilation**: The model is configured with an `adam` optimizer, `sparse_categorical_crossentropy` loss (suitable for integer labels), and `accuracy` as the metric.
6.  **Model Training**: The `fit()` method trains the model on the training data for a specified number of `epochs` and `batch_size`.
7.  **Evaluation**: The `evaluate()` method calculates the loss and accuracy on the test dataset.
8.  **Prediction**: Demonstrates how to make a prediction on a single test image and prints the predicted digit.

## How to Run

1.  Save the code as `cnn.py`.
2.  Ensure you have `tensorflow` installed:
    ```bash
    pip install tensorflow
    ```
3.  Run the script from your terminal:
    ```bash
    python cnn.py
    ```

## Expected Output

(Output will vary slightly due to training progress, but will be similar to this.)

```
Epoch 1/5
938/938 [==============================] - 15s 16ms/step - loss: 0.1776 - accuracy: 0.9470
Epoch 2/5
938/938 [==============================] - 15s 16ms/step - loss: 0.0577 - accuracy: 0.9822
Epoch 3/5
938/938 [==============================] - 15s 16ms/step - loss: 0.0401 - accuracy: 0.9877
Epoch 4/5
938/938 [==============================] - 15s 16ms/step - loss: 0.0305 - accuracy: 0.9904
Epoch 5/5
938/938 [==============================] - 15s 16ms/step - loss: 0.0245 - accuracy: 0.9922
313/313 [==============================] - 1s 4ms/step - loss: 0.0371 - accuracy: 0.9880
Test Accuracy: 0.9880000007152557
Predicted digit: 7
```

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