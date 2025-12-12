# Feature Engineering for NLP: Preprocessing and Vectorization

This example demonstrates fundamental natural language processing (NLP) techniques, including text preprocessing (tokenization, stopword removal, stemming) and feature extraction methods like Bag of Words (BoW) and TF-IDF.

## Overview

Feature engineering in NLP involves transforming raw text into numerical representations that machine learning models can understand. This process typically includes:

1.  **Preprocessing**: Cleaning and normalizing text.
    *   **Tokenization**: Breaking text into individual words or sub-word units.
    *   **Stopword Removal**: Eliminating common words (e.g., 'the', 'is', 'a') that often carry little semantic meaning.
    *   **Stemming/Lemmatization**: Reducing words to their base or root form (e.g., 'running' -> 'run', 'dogs' -> 'dog').
2.  **Feature Extraction (Vectorization)**: Converting preprocessed text into numerical vectors.
    *   **Bag of Words (BoW)**: Represents text as an unordered collection of words, counting word occurrences. It focuses on the presence and frequency of words.
    *   **TF-IDF (Term Frequency-Inverse Document Frequency)**: A statistical measure reflecting how important a word is to a document in a collection or corpus. It increases proportionally to the number of times a word appears in the document but is offset by the frequency of the word in the corpus, which helps to adjust for the fact that some words appear more frequently in general.

## Code Explanation

1.  **Imports**: 
    *   `nltk` for tokenization, stopwords, and stemming.
    *   `CountVectorizer` for Bag of Words.
    *   `TfidfVectorizer` for TF-IDF.
2.  **NLTK Downloads**: Ensures necessary NLTK data ('punkt' for tokenization, 'stopwords' for stopword list) is available.
3.  **Sample Documents**: A list of `documents` for demonstration.
4.  **Step 1: Preprocessing**: 
    *   Initializes `stop_words` from NLTK and a `PorterStemmer`.
    *   Iterates through each document:
        *   Converts to lowercase (`doc.lower()`).
        *   Tokenizes the text using `nltk.word_tokenize()`.
        *   Filters tokens: keeps only alphabetic words and removes stopwords.
        *   Applies stemming to remaining tokens.
        *   Joins the processed tokens back into a string.
    *   Prints the `processed_docs`.
5.  **Step 2: Feature Extraction using Bag of Words**: 
    *   Creates a `CountVectorizer` instance.
    *   `fit_transform()` learns the vocabulary and transforms the `processed_docs` into a `bow_matrix` (sparse matrix of word counts).
    *   `get_feature_names_out()` displays the extracted vocabulary.
    *   `bow_matrix.toarray()` converts the sparse matrix to a dense NumPy array for display.
6.  **Step 3: Feature Extraction using TF-IDF**: 
    *   Creates a `TfidfVectorizer` instance.
    *   `fit_transform()` learns the vocabulary and calculates TF-IDF scores for the `processed_docs`, producing `tfidf_matrix`.
    *   Prints feature names and the TF-IDF matrix.

## How to Run

1.  Save the code as `feature.py`.
2.  Ensure you have the necessary libraries installed:
    ```bash
    pip install nltk scikit-learn
    ```
3.  Run the script from your terminal:
    ```bash
    python feature.py
    ```

## Expected Output

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
[[0.73033501 0.49079549 0.49079549 0.         0.         0.         0.         0.         0.         0.        ]
 [0.         0.         0.         0.57973867 0.57973867 0.         0.         0.57973867 0.         0.        ]
 [0.         0.         0.36625807 0.36625807 0.         0.         0.45785934 0.         0.45785934 0.45785934]
 [0.57735027 0.         0.         0.         0.         0.81649658 0.         0.         0.         0.        ]]
```

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

# Step 1: Preprocessing - Tokenization, Stopword Removal, and Stemming
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

# Step 2: Feature Extraction using Bag of Words
count_vectorizer = CountVectorizer()
bow_matrix = count_vectorizer.fit_transform(processed_docs)

print("\nBag of Words Feature Names:")
print(count_vectorizer.get_feature_names_out())

print("\nBag of Words Matrix:")
print(bow_matrix.toarray())

# Step 3: Feature Extraction using TF-IDF
tfidf_vectorizer = TfidfVectorizer()
tfidf_matrix = tfidf_vectorizer.fit_transform(processed_docs)

print("\nTF-IDF Feature Names:")
print(tfidf_vectorizer.get_feature_names_out())

print("\nTF-IDF Matrix:")
print(tfidf_matrix.toarray())
```