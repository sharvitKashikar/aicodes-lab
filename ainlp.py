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
