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
