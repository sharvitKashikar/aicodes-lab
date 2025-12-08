# Natural Language Processing Examples

This section provides Python code examples demonstrating fundamental concepts in Natural Language Processing (NLP).

## Sentiment Analysis with `ainlp.py`

This script demonstrates a basic sentiment analysis pipeline using scikit-learn. It trains a simple Naive Bayes classifier to classify text as positive or negative.

### How it Works
1. Reads a small dataset of sample texts and their sentiment labels (positive/negative).
2. Uses `TfidfVectorizer` to convert text into numerical features (TF-IDF).
3. Trains a `MultinomialNB` (Naive Bayes) classifier on these features.
4. Predicts the sentiment of a new, unseen text.

### Code (`ainlp.py`)
```python
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline

# Sample training data
texts = [
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
labels = ['pos', 'pos', 'neg', 'neg', 'pos', 'neg', 'pos', 'pos', 'neg', 'pos']  # Sentiment labels

# Create and train model pipeline
model = make_pipeline(TfidfVectorizer(), MultinomialNB())
model.fit(texts, labels)

# Test with new data
test_text = "The movie was full of suspense and very enjoyable"
prediction = model.predict([test_text])

# Show result
print("Predicted sentiment:", prediction[0])
```

### How to Run
```bash
python ainlp.py
```

### Expected Output
```
Predicted sentiment: pos
```

## Text Feature Extraction with `feature.py`

This script illustrates common NLP preprocessing steps and two fundamental text feature extraction techniques: Bag of Words (BoW) and TF-IDF (Term Frequency-Inverse Document Frequency).

### How it Works
1. **Preprocessing**: Downloads NLTK data (punkt, stopwords), tokenizes text, converts to lowercase, removes stopwords, and performs stemming using `PorterStemmer`.
2. **Bag of Words**: Uses `CountVectorizer` to convert processed texts into a matrix where each row represents a document and each column represents a unique word from the vocabulary, with cell values being the word's frequency.
3. **TF-IDF**: Uses `TfidfVectorizer` to convert texts into a matrix where cell values reflect the importance of a word in a document relative to the entire corpus.

### Code (`feature.py`)
```python
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

# Download required NLTK data
nltk.download('punkt')
nltk.download('stopwords')

# Sample raw texts
documents = [
    "Cats are beautiful animals.",
    "Dogs are loyal and friendly.",
    "Cats and dogs can live together peacefully.",
    "Some animals are more intelligent than others."
]

# Preprocessing
stop_words = set(stopwords.words('english'))
stemmer = PorterStemmer()

processed_docs = []

for doc in documents:
    tokens = nltk.word_tokenize(doc.lower())  # lowercase + tokenize
    filtered_tokens = [stemmer.stem(word) for word in tokens if word.isalpha() and word not in stop_words]
    processed_docs.append(" ".join(filtered_tokens))

print("Processed Documents:")
for doc in processed_docs:
    print(doc)

# Feature Extraction using Bag of Words
count_vectorizer = CountVectorizer()
bow_matrix = count_vectorizer.fit_transform(processed_docs)

print("\nBag of Words Feature Names:")
print(count_vectorizer.get_feature_names_out())

print("\nBag of Words Matrix:")
print(bow_matrix.toarray())

# Feature Extraction using TF-IDF
tfidf_vectorizer = TfidfVectorizer()
tfidf_matrix = tfidf_vectorizer.fit_transform(processed_docs)

print("\nTF-IDF Feature Names:")
print(tfidf_vectorizer.get_feature_names_out())

print("\nTF-IDF Matrix:")
print(tfidf_matrix.toarray())
```

### How to Run
```bash
python feature.py
```

### Expected Output (Example Snippets)
```
Processed Documents:
cat beauti anim
dog loyal friendli
cat dog live togeth peac
anim intellig

Bag of Words Feature Names:
['anim' 'beauti' 'cat' 'dog' 'friendli' 'intellig' 'live' 'loyal' 'peac' 'togeth']

Bag of Words Matrix:
[[1 1 1 0 0 0 0 0 0 0]
 [0 0 0 1 1 0 0 1 0 0]
 [0 0 1 1 0 0 1 0 1 1]
 [1 0 0 0 0 1 0 0 0 0]]

TF-IDF Feature Names:
['anim' 'beauti' 'cat' 'dog' 'friendli' 'intellig' 'live' 'loyal' 'peac' 'togeth']

TF-IDF Matrix:
[[0.48045863 0.62779434 0.48045863 0.         0.         0.
  0.         0.         0.         0.        ]
 [0.         0.         0.         0.48045863 0.62779434 0.
  0.         0.62779434 0.         0.        ]
 [0.         0.         0.36854179 0.36854179 0.         0.
  0.48160408 0.         0.48160408 0.48160408]
 [0.55174577 0.         0.         0.         0.         0.833946   0.
  0.         0.         0.        ]]
```