# RNN Sequence Prediction Example

This example demonstrates a basic Recurrent Neural Network (RNN) built with TensorFlow/Keras to learn and predict the next number in a simple sequence.

## Purpose

The `rnn.py` script trains a `SimpleRNN` model to predict `y = x + 1`. It is trained on sequences like `0 -> 1`, `1 -> 2`, up to `8 -> 9`. After training, it predicts the output for an input `10`.

## Model Details

- **Architecture**: A `Sequential` model consisting of:
    - `SimpleRNN` layer with 32 units and `tanh` activation.
    - `Dense` output layer with 1 unit.
- **Input Shape**: `(1, 1)` representing (timesteps, features).
- **Optimizer**: `adam`
- **Loss Function**: Mean Squared Error (`mse`)
- **Epochs**: 200

## Dataset

The model is trained on the following sequence mapping:

| Input (X) | Output (y) |
|-----------|------------|
| 0         | 1          |
| 1         | 2          |
| ...       | ...        |
| 8         | 9          |

The input `X` is reshaped to `(samples, timesteps, features)` which is required for RNN layers, specifically `(10, 1, 1)`.

## How to Run

1. Ensure you have Python and TensorFlow installed. If not, you can install TensorFlow using pip:
   ```bash
   pip install tensorflow numpy
   ```
2. Navigate to the directory containing `rnn.py`.
3. Execute the script:
   ```bash
   python rnn.py
   ```

## Expected Output

After running the script, the model will train and then make a prediction for the input `10`. The output will be similar to (due to the nature of neural networks, the exact prediction might vary slightly):

```
Input: 10
Predicted next number: 10.9xxxxxxxx
```

This output indicates that the model has successfully learned the `x + 1` pattern and predicts a value close to `11` for an input of `10`.