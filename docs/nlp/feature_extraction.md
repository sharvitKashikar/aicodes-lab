```markdown
# Natural Language Processing (NLP) Feature Extraction

This module (`feature.py`) illustrates common techniques for preprocessing text and extracting numerical features, which are essential steps in many NLP tasks like text classification, clustering, and information retrieval.

## Implementation Details

The script covers the following key stages:

### 1. Preprocessing

Text preprocessing aims to clean and normalize text data before feature extraction. The following steps are applied:

*   **Tokenization:** Breaking down text into individual words or sub-word units. The script uses `nltk.word_tokenize`.
*   **Lowercasing:** Converting all text to lowercase to treat words like "The" and "the" as the same.
*   **Stopword Removal:** Removing common words (like "is", "a", "the") that often carry little semantic meaning and can add noise to features. A list of English stopwords from `nltk.corpus.stopwords` is used.
*   **Stemming:** Reducing words to their root or base form (e.g., "running", "ran", "runs" -> "run"). The script applies `nltk.stem.PorterStemmer`.

### 2. Feature Extraction

After preprocessing, text needs to be converted into numerical vectors that machine learning models can understand. The script demonstrates two popular methods:

*   **Bag of Words (BoW):**
    *   **Concept:** Represents a text as an unordered collection of words, disregarding grammar and word order but keeping multiplicity. It counts the occurrences of each word in the document.
    *   **Implementation:** Uses `sklearn.feature_extraction.text.CountVectorizer` to convert a collection of text documents to a matrix of token counts.

*   **TF-IDF (Term Frequency-Inverse Document Frequency):**
    *   **Concept:** A statistical measure used to evaluate how important a word is to a document in a collection or corpus. Words that are frequent in a document but rare across the entire corpus often have higher TF-IDF scores.
    *   **Implementation:** Uses `sklearn.feature_extraction.text.TfidfVectorizer` to transform raw documents to a matrix of TF-IDF features.

## Example Usage

To run the NLP feature extraction examples, execute the `feature.py` script:

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
[[0.58919632 0.70710678 0.3957813  0.         0.         0.         0.         0.         0.         0.        ]
 [0.         0.         0.         0.49071017 0.58919632 0.         0.         0.58919632 0.         0.        ]
 [0.         0.         0.33418587 0.33418587 0.         0.         0.40149021 0.         0.40149021 0.40149021]
 [0.81180295 0.         0.         0.         0.         0.58336306 0.         0.         0.         0.        ]]
```
