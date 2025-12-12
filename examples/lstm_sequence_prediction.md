# Long Short-Term Memory (LSTM) for Sequence Prediction

This example demonstrates using an LSTM (Long Short-Term Memory) recurrent neural network in TensorFlow/Keras to predict the next number in a simple sequence (e.g., 0→1, 1→2, ..., 9→10).

## LSTM Overview

LSTMs are a type of Recurrent Neural Network (RNN) capable of learning long-term dependencies. They are widely used for sequential data tasks such as time series prediction, natural language processing, and speech recognition. LSTMs address the vanishing/exploding gradient problem common in simple RNNs by incorporating 'gates' (forget, input, output) that control the flow of information into and out of the cell state, allowing them to selectively remember or forget information over long sequences.

## Code Explanation

1.  **Imports**: Uses `numpy` for data handling and `tensorflow.keras` for building the LSTM model.
2.  **Dataset Creation**: 
    *   `X`: Input sequence (0 to 9).
    *   `y`: Target sequence (1 to 10), which is `X + 1`.
3.  **Data Reshaping**: LSTM layers in Keras expect input in the format `(samples, timesteps, features)`. The `X` array is reshaped from `(10,)` to `(10, 1, 1)`.
4.  **Model Building (`Sequential` API)**:
    *   `LSTM(64, activation='tanh', input_shape=(1, 1))`: An LSTM layer with 64 units, using `tanh` activation. `input_shape=(1, 1)` indicates 1 timestep and 1 feature per timestep.
    *   `Dense(1)`: A final dense layer with 1 unit to output the predicted next number.
5.  **Model Compilation**: The model is compiled with the `adam` optimizer and `mse` (Mean Squared Error) as the loss function, suitable for regression tasks.
6.  **Model Training**: The `fit()` method trains the LSTM on the generated sequence data for 300 `epochs`. `verbose=0` keeps the training output silent.
7.  **Model Testing**: 
    *   A `test_value` (e.g., 10) is prepared in the correct LSTM input format.
    *   The `predict()` method is used to get the model's prediction for the next number in the sequence.
    *   The predicted value is printed.

## How to Run

1.  Save the code as `lstm.py`.
2.  Ensure you have `tensorflow` installed:
    ```bash
    pip install tensorflow
    ```
3.  Run the script from your terminal:
    ```bash
    python lstm.py
    ```

## Expected Output

```
Input: 10
Predicted next number: 10.999...
```
(The predicted value will be very close to 11.0, with minor floating point variations).

```python
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

# Create dataset
# Example: 0→1, 1→2, 2→3 ... 8→9
X = np.array([[i] for i in range(10)], dtype=float)
y = np.array([i + 1 for i in range(10)], dtype=float)

# Reshape input to LSTM format: (samples, timesteps, features)
X = X.reshape((X.shape[0], 1, 1))

# Build LSTM model
model = Sequential([
    LSTM(64, activation='tanh', input_shape=(1, 1)),
    Dense(1)
])

# Compile model
model.compile(optimizer='adam', loss='mse')

# Train the LSTM
model.fit(X, y, epochs=300, verbose=0)

# Test the model
test_value = np.array([10]).reshape((1, 1, 1))
prediction = model.predict(test_value)

print("Input: 10")
print("Predicted next number:", float(prediction))
```