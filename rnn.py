import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import SimpleRNN, Dense

# Create dataset
# Sequence: 0 → 1, 1 → 2, 2 → 3, ..., 8 → 9
X = np.array([[i] for i in range(10)], dtype=float)
y = np.array([i + 1 for i in range(10)], dtype=float)

# Reshape input to RNN format (samples, timesteps, features)
X = X.reshape((X.shape[0], 1, 1))

# Build RNN model
model = Sequential([
    SimpleRNN(32, activation='tanh', input_shape=(1, 1)),
    Dense(1)
])

# Compile
model.compile(optimizer='adam', loss='mse')

# Train
model.fit(X, y, epochs=200, verbose=0)

# Test the model
test_input = np.array([10]).reshape((1, 1, 1))
prediction = model.predict(test_input)

print("Input: 10")
print("Predicted next number:", float(prediction))
