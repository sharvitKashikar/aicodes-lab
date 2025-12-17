# aicodes-lab

Welcome to `aicodes-lab`! This repository serves as a collection of fundamental algorithms, data structures, game implementations, and AI/NLP techniques implemented in Python. It's designed for learning, experimentation, and quick reference.

## Table of Contents

- [About](#about)
- [Repository Structure](#repository-structure)
- [Setup and Installation](#setup-and-installation)
- [Usage](#usage)
- [Contributing](#contributing)
- [License](#license)

## About

`aicodes-lab` aims to provide clear, concise, and runnable examples of various computational concepts. Whether you're a student learning computer science, a developer brushing up on algorithms, or an AI enthusiast exploring NLP, you'll find accessible code examples here.

## Repository Structure

The repository is organized into categories based on the type of problem or technique:

```
.
├── README.md
├── bfs.py              # Breadth-First Search algorithm
├── dfs.py              # Depth-First Search algorithm
├── queen.py            # N-Queens problem solver
├── tic.py              # Tic-Tac-Toe game implementation
├── tokenization.py     # NLTK-based text tokenization, stopword removal, stemming
├── tfidf.py            # TF-IDF (Term Frequency-Inverse Document Frequency) calculation
├── feature.py          # Comprehensive text feature extraction (BoW, TF-IDF)
├── ainlp.py            # Simple AI NLP sentiment analysis model using TF-IDF and Naive Bayes
└── docs/
    ├── algorithms.md         # Documentation for graph traversal algorithms
    ├── game_implementations.md # Documentation for game logic implementations
    └── nlp.md                  # Documentation for Natural Language Processing examples
```

## Setup and Installation

To run the Python scripts in this repository, you'll need Python 3.x installed. Some scripts also require specific libraries. It's recommended to set up a virtual environment.

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/sharvitKashikar/aicodes-lab.git
    cd aicodes-lab
    ```

2.  **Create and activate a virtual environment (optional but recommended):**
    ```bash
    python3 -m venv venv
    source venv/bin/activate  # On Windows: `venv\Scripts\activate`
    ```

3.  **Install dependencies:**
    Some scripts like `tokenization.py`, `tfidf.py`, `feature.py`, and `ainlp.py` use libraries such as `nltk` and `scikit-learn`. Install them using pip:
    ```bash
    pip install nltk scikit-learn
    ```
    *Note: For `nltk`, you might also need to download specific data packages. The `tokenization.py` and `feature.py` scripts include `nltk.download()` calls to handle this automatically on first run.*

## Usage

Each `.py` file is designed to be run independently. Navigate to the root directory of the repository and execute the desired script:

```bash
python3 <filename>.py
```

For example:

```bash
python3 bfs.py
python3 queen.py
python3 ainlp.py
```

Detailed usage and explanations for each category can be found in the `docs/` directory:

-   [**Algorithms**](docs/algorithms.md): Learn about graph traversal algorithms (BFS, DFS).
-   [**Game Implementations**](docs/game_implementations.md): Explore classic game logic (N-Queens, Tic-Tac-Toe).
-   [**Natural Language Processing**](docs/nlp.md): Dive into text processing and AI models (Tokenization, TF-IDF, Sentiment Analysis).

## Contributing

Contributions are welcome! If you have an algorithm, data structure, or AI/NLP example you'd like to add, please follow these steps:

1.  Fork the repository.
2.  Create a new branch (`git checkout -b feature/your-feature-name`).
3.  Implement your code and add relevant tests (if applicable).
4.  Add appropriate documentation in the `docs/` directory.
5.  Commit your changes (`git commit -m 'Add new feature: brief description'`).
6.  Push to the branch (`git push origin feature/your-feature-name`).
7.  Open a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details (if one exists).