# Breadth-First Search (BFS)

This example demonstrates the Breadth-First Search (BFS) algorithm, a graph traversal algorithm that explores all of the neighbor nodes at the present depth prior to moving on to the nodes at the next depth level.

## Algorithm Overview

BFS starts at a specified node and explores all of its immediate neighbors. Then, for each of those neighbors, it explores their unvisited neighbors, and so on. It uses a queue data structure to keep track of nodes to visit.

## Code Explanation

1.  **Imports**: `collections.deque` is used as an efficient queue for graph traversal.
2.  **`bfs(graph, start)` function**: Takes a graph (represented as an adjacency list) and a starting node.
    *   `visited`: A set to keep track of visited nodes to prevent infinite loops and redundant processing.
    *   `queue`: A `deque` initialized with the `start` node. The `start` node is also marked as visited.
    *   **Traversal Loop**: While the queue is not empty, it dequeues a `node`, prints it, and then enqueues all its unvisited `neighbor`s, marking them as visited.
3.  **Example Usage**: Defines a sample `graph` as a dictionary (adjacency list) and then calls the `bfs` function starting from node 'A'.

## How to Run

1.  Save the code as `bfs.py`.
2.  Run the script from your terminal:
    ```bash
    python bfs.py
    ```

## Expected Output

```
BFS traversal starting from node 'A':
A B C D E F 
```

```python
from collections import deque

def bfs(graph, start):
    visited = set()
    queue = deque([start])
    visited.add(start)

    while queue:
        node = queue.popleft()
        print(node, end=" ")

        for neighbor in graph[node]:
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append(neighbor)

# Example usage:
graph = {
    'A': ['B', 'C'],
    'B': ['D', 'E'],
    'C': ['F'],
    'D': [],
    'E': [],
    'F': []
}
print("BFS traversal starting from node 'A':")
bfs(graph, 'A')
```