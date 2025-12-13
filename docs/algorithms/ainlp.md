```markdown
# Sentiment Analysis using TF-IDF + Naive Bayes

This module (`ainlp.py`) demonstrates sentiment analysis using a combination of TF-IDF (Term Frequency-Inverse Document Frequency) for feature extraction and a Multinomial Naive Bayes classifier. It's designed to classify short text documents into 'positive' or 'negative' sentiments.

## Implementation Details

The script utilizes `sklearn.feature_extraction.text.TfidfVectorizer` to convert text into numerical TF-IDF features and `sklearn.naive_bayes.MultinomialNB` for the classification. A `sklearn.pipeline.make_pipeline` is used to streamline the process of vectorization and classification.

**Key steps include:**
1.  **Sample Dataset:** A small set of predefined documents and their corresponding 'pos' (positive) or 'neg' (negative) labels.
2.  **Data Splitting:** The dataset is split into training and testing sets using `train_test_split`.
3.  **Model Creation:** A pipeline is created combining `TfidfVectorizer` and `MultinomialNB`.
4.  **Model Training:** The model is trained on the training data.
5.  **Model Evaluation:** The model's accuracy is printed based on predictions on the test set.
6.  **Prediction on New Data:** Demonstrates how to use the trained model to predict sentiment for new, unseen text samples.

## Example Usage

To run the sentiment analysis model and see its predictions, execute the `ainlp.py` script:

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

## Expected Output

```
Model Accuracy: 0.6666666666666666
Text: 'The movie was full of suspense and very enjoyable' → Sentiment: pos
Text: 'I hated the storyline, it was terrible' → Sentiment: neg
Text: 'Amazing film with great acting!' → Sentiment: pos
```
