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
