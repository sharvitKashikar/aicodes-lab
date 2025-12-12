# AI/ML and Algorithm Examples

This directory contains various Python implementations and examples for common Artificial Intelligence, Machine Learning, and fundamental Computer Science algorithms. Each script is self-contained and demonstrates a specific concept.

## K-Means Clustering (`K means.py`)

Demonstrates the K-Means clustering algorithm for unsupervised learning. It groups data points into a specified number of clusters based on feature similarity. The example uses a simple Age-Income dataset.

**Key Concepts:** Clustering, Unsupervised Learning, Centroid-based algorithms.
**How to Run:** `python "K means.py"`

**Example Output:**
```
Cluster Centers:
 [[27.   22000.  ]
 [67.   72333.33333333]
 [47.   52000.  ]]
Labels: [0 0 0 2 2 2 1 1 1]
```
(A plot will also be displayed.)

## Sentiment Analysis (TF-IDF + Naive Bayes) (`ainlp.py`)

Implements a sentiment analysis model using TF-IDF for feature extraction and a Multinomial Naive Bayes classifier. It trains on a small dataset of positive and negative movie reviews and then predicts the sentiment of new samples.

**Key Concepts:** Natural Language Processing (NLP), Sentiment Analysis, TF-IDF, Naive Bayes, Text Classification, Scikit-learn Pipelines.
**How to Run:** `python ainlp.py`

**Example Output:**
```
Model Accuracy: 0.6666666666666666
Text: 'The movie was full of suspense and very enjoyable' → Sentiment: pos
Text: 'I hated the storyline, it was terrible' → Sentiment: neg
Text: 'Amazing film with great acting!' → Sentiment: pos
```

## A* Search Algorithm (`astar.py`)

An implementation of the A* search algorithm, a pathfinding algorithm that finds the shortest path between two points in a grid using a heuristic function (Manhattan distance in this case). It navigates around obstacles (represented by '1' in the grid).

**Key Concepts:** Pathfinding, Graph Search Algorithms, Heuristics, Priority Queue (heapq).
**How to Run:** `python astar.py`

**Example Output:**
```
Shortest path using A*: [(0, 0), (0, 1), (0, 2), (0, 3), (1, 3), (2, 3), (3, 3)]
```

## Breadth-First Search (BFS) (`bfs.py`)

Demonstrates the Breadth-First Search (BFS) algorithm for traversing or searching tree or graph data structures. It explores all of the neighbor nodes at the present depth prior to moving on to the nodes at the next depth level.

**Key Concepts:** Graph Traversal, Queue, Shortest Path on unweighted graphs.
**How to Run:** `python bfs.py`

**Example Output:**
```
BFS traversal starting from node 'A':
A B C D E F
```

## Convolutional Neural Network (CNN) for MNIST (`cnn.py`)

An example of a Convolutional Neural Network (CNN) built with TensorFlow/Keras for classifying handwritten digits from the MNIST dataset. It includes data loading, reshaping, model building with Conv2D and MaxPooling2D layers, compilation, training, and evaluation.

**Key Concepts:** Deep Learning, Convolutional Neural Networks, Image Classification, MNIST Dataset, TensorFlow, Keras.
**How to Run:** `python cnn.py`

**Example Output:**
```
(Training output will be displayed during execution)
Test Accuracy: 0.98...
Predicted digit: X (where X is the predicted digit for the first test sample)
```

## Depth-First Search (DFS) (`dfs.py`)

Demonstrates the Depth-First Search (DFS) algorithm for traversing or searching tree or graph data structures. It explores as far as possible along each branch before backtracking.

**Key Concepts:** Graph Traversal, Recursion, Stack.
**How to Run:** `python dfs.py`

**Example Output:**
```
DFS traversal starting from node 'A':
A B D E C F
```

## Decision Tree Classifier (`dt.py`)

Implements a Decision Tree Classifier using Scikit-learn for basic classification tasks. It uses the Iris dataset for training and evaluation and also visualizes the resulting decision tree.

**Key Concepts:** Supervised Learning, Classification, Decision Trees, Gini Impurity, Feature Importance, Scikit-learn.
**How to Run:** `python dt.py`

**Example Output:**
```
Accuracy: 0.9777777777777777
```
(A plot of the decision tree will also be displayed.)

## Feature Extraction for NLP (`feature.py`)

An example demonstrating text preprocessing (tokenization, stopword removal, stemming) and two common NLP feature extraction techniques: Bag of Words (CountVectorizer) and TF-IDF (TfidfVectorizer). 

**Key Concepts:** Natural Language Processing (NLP), Text Preprocessing, Tokenization, Stopwords, Stemming, Bag of Words, TF-IDF.
**How to Run:** `python feature.py`

**Example Output:**
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
 [1 0 0 1 1 0 0 1 0 0]
 [0 0 1 1 0 0 1 0 1 1]
 [1 0 0 0 0 1 0 0 0 0]]

TF-IDF Feature Names:
['anim' 'beauti' 'cat' 'dog' 'friendli' 'intellig' 'live' 'loyal' 'peac' 'togeth']

TF-IDF Matrix:
[...TF-IDF values...]
```

## Linear Regression (`linear.py`)

Demonstrates a basic Linear Regression model using Scikit-learn. It predicts house prices based on house size, showcasing the fundamental principles of linear relationships in data.

**Key Concepts:** Supervised Learning, Regression, Linear Models, Scikit-learn.
**How to Run:** `python linear.py`

**Example Output:**
```
Coefficient (Slope): 0.07
Intercept: -5.0
Predicted price for 900 sq ft house: 58.0 lakh ₹
```

## Naive Bayes Text Classification (`nbayes.py`)

An introductory example of text classification using a Multinomial Naive Bayes classifier combined with TF-IDF for feature extraction. It trains on simple positive/negative text samples and then predicts the sentiment of new input.

**Key Concepts:** Text Classification, Sentiment Analysis, Naive Bayes, TF-IDF, Scikit-learn Pipelines.
**How to Run:** `python nbayes.py`

**Example Output:**
```
Prediction: positive
```
