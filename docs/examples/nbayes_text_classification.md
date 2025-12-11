# Naive Bayes Text Classification Example

This example demonstrates a simple Naive Bayes text classification pipeline using `scikit-learn` in Python.
It showcases how to train a model to classify text as 'positive' or 'negative' based on a small sample dataset.

## Prerequisites

To run this example, you need to have `scikit-learn` installed. You can install it using pip:

```bash
pip install scikit-learn
```

## `nbayes.py` Script

The `nbayes_text_classification.py` script performs the following steps:

1.  **Imports necessary libraries**: `TfidfVectorizer` for text feature extraction, `MultinomialNB` for the Naive Bayes classifier, and `make_pipeline` to combine them.
2.  **Defines sample training data**: A list of texts (`texts`) and their corresponding labels (`labels`).
3.  **Builds a model pipeline**: It creates a pipeline that first converts text into TF-IDF features and then applies the Multinomial Naive Bayes classifier.
4.  **Trains the model**: The model is trained using the `texts` and `labels`.
5.  **Tests prediction**: A new text is provided to the trained model for classification, and the prediction is printed.

```python
# nbayes.py
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline

# Sample training data
texts = [
    "I love this product",
    "This is amazing",
    "Very bad experience",
    "I hate this",
    "Totally fantastic",
    "Worst thing ever"
]

labels = ["positive", "positive", "negative", "negative", "positive", "negative"]

# Build model
model = make_pipeline(TfidfVectorizer(), MultinomialNB())
model.fit(texts, labels)

# Test prediction
test = ["This product is good"]
print("Prediction:", model.predict(test)[0])
```

## How to Run

1.  Save the code above into a file named `nbayes.py`.
2.  Open your terminal or command prompt.
3.  Navigate to the directory where you saved the file.
4.  Run the script using Python:

    ```bash
    python nbayes.py
    ```

Upon execution, you will see the predicted label for the test text "This product is good" (which should be 'positive').
