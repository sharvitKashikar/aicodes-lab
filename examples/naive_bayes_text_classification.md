# Naive Bayes Text Classification

This example demonstrates basic text classification using the Naive Bayes algorithm with TF-IDF vectorization. It predicts whether a given text has a 'positive' or 'negative' sentiment.

## Algorithm Overview

1.  **TF-IDF Vectorization**: Text data is first converted into numerical feature vectors using TF-IDF. This weights words based on their frequency in a document and across the entire corpus.
2.  **Multinomial Naive Bayes Classifier**: This classifier is particularly suitable for classification with discrete features (like word counts or TF-IDF values). It's a probabilistic classifier based on Bayes' theorem, assuming strong independence between features (words), which is 'naive' but often effective in practice for text.

## Code Explanation

1.  **Imports**: Uses `TfidfVectorizer` and `MultinomialNB` from `sklearn` for text processing and classification, and `make_pipeline` to chain these steps.
2.  **Sample Training Data**: `texts` contains training review snippets, and `labels` are their corresponding sentiments ('positive' or 'negative').
3.  **Model Building**: `make_pipeline` creates a sequence: `TfidfVectorizer` first transforms raw text into TF-IDF features, and then `MultinomialNB` trains a classifier on these features.
4.  **Model Training**: The `fit()` method trains the pipeline on the sample `texts` and `labels`.
5.  **Test Prediction**: A new `test` phrase is provided, and the model's `predict()` method is used to determine its sentiment.

## How to Run

1.  Save the code as `nbayes.py`.
2.  Ensure you have `scikit-learn` installed:
    ```bash
    pip install scikit-learn
    ```
3.  Run the script from your terminal:
    ```bash
    python nbayes.py
    ```

## Expected Output

```
Prediction: positive
```

```python
# Naive Bayes Text Classification
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