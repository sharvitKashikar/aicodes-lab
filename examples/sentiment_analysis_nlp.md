# Sentiment Analysis using TF-IDF + Naive Bayes

This example demonstrates how to perform sentiment analysis on text data using a combination of TF-IDF (Term Frequency-Inverse Document Frequency) for feature extraction and the Multinomial Naive Bayes classifier.

## Algorithm Overview

1.  **TF-IDF Vectorization**: Converts text documents into numerical feature vectors. TF-IDF gives higher weights to words that are frequent in a document but rare across all documents, highlighting words that are more indicative of a document's content.
2.  **Multinomial Naive Bayes**: A probabilistic classifier based on Bayes' theorem, particularly well-suited for classification with discrete features (like word counts or TF-IDF values), commonly used in text classification.

## Code Explanation

1.  **Imports**: Utilizes `TfidfVectorizer` and `MultinomialNB` from `sklearn` for NLP and classification, `train_test_split` for data partitioning, and `accuracy_score` for evaluation.
2.  **Sample Dataset**: `documents` is a list of movie review texts, and `labels` are their corresponding sentiment ('pos' for positive, 'neg' for negative).
3.  **Data Splitting**: The dataset is divided into training and testing sets using `train_test_split` to evaluate the model's performance on unseen data.
4.  **Pipeline Creation**: `make_pipeline` is used to create a chained sequence of operations: first `TfidfVectorizer` to transform text into TF-IDF features, then `MultinomialNB` to classify the sentiment.
5.  **Model Training**: The pipeline's `fit()` method is called on the training data (`X_train`, `y_train`).
6.  **Model Evaluation**: Predictions are made on the test set (`X_test`), and the `accuracy_score` is printed to show the model's performance.
7.  **New Predictions**: The model is tested with `test_samples` to demonstrate its ability to predict sentiment for new, unseen texts.

## How to Run

1.  Save the code as `ainlp.py`.
2.  Ensure you have `scikit-learn` installed:
    ```bash
    pip install scikit-learn
    ```
3.  Run the script from your terminal:
    ```bash
    python ainlp.py
    ```

## Expected Output

```
Model Accuracy: 0.6666666666666666
Text: 'The movie was full of suspense and very enjoyable' → Sentiment: pos
Text: 'I hated the storyline, it was terrible' → Sentiment: neg
Text: 'Amazing film with great acting!' → Sentiment: pos
```

```python
# Sentiment Analysis using TF-IDF + Naive Bayes

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.pipeline import make_pipeline

# Sample dataset
documents = [
    "I loved the movie, it was amazing!",
    "Fantastic film with a great story.",
    "Terrible movie, I hated it.",
    "The plot was boring and predictable.",
    "What a wonderful experience!",
    "Worst movie I've seen in years.",
    "That was a very enjoyable movie!",
    "The suspense in the film was thrilling.",
    "Not a single exciting moment in it.",
    "Enjoyable and full of surprises."
]

labels = [
    'pos', 'pos', 'neg', 'neg', 'pos',
    'neg', 'pos', 'pos', 'neg', 'pos'
]

# Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(
    documents, labels, test_size=0.3, random_state=42
)

# Create a pipeline (TF-IDF + Naive Bayes)
model = make_pipeline(TfidfVectorizer(), MultinomialNB())

# Train the model
model.fit(X_train, y_train)

# Evaluate model
y_pred = model.predict(X_test)
print("Model Accuracy:", accuracy_score(y_test, y_pred))

# Test with new inputs
test_samples = [
    "The movie was full of suspense and very enjoyable",
    "I hated the storyline, it was terrible",
    "Amazing film with great acting!"
]

predictions = model.predict(test_samples)

# Show output
for text, sentiment in zip(test_samples, predictions):
    print(f"Text: '{text}' → Sentiment: {sentiment}")
```