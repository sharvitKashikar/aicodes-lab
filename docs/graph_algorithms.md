# Graph Traversal Algorithms

This section provides Python code examples for common graph traversal algorithms.

## Breadth-First Search (BFS) with `bfs.py`

Breadth-First Search (BFS) is an algorithm for traversing or searching tree or graph data structures. It starts at the tree root (or some arbitrary node of a graph, sometimes referred to as a 'search key') and explores all of the neighbor nodes at the present depth prior to moving on to the nodes at the next depth level.

### How it Works
1. Uses a queue to keep track of nodes to visit.
2. Starts from a given `start` node, marks it as visited, and adds it to the queue.
3. While the queue is not empty, it dequeues a node, prints it, and then enqueues all its unvisited neighbors, marking them as visited.

### Code (`bfs.py`)
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

### How to Run
```bash
python bfs.py
```

### Expected Output
```
BFS traversal starting from node 'A':
A B C D E F 
```

## Depth-First Search (DFS) with `dfs.py`

Depth-First Search (DFS) is an algorithm for traversing or searching tree or graph data structures. The algorithm starts at the root node (selecting some arbitrary node as the root in the case of a graph) and explores as far as possible along each branch before backtracking.

### How it Works
1. Uses recursion (or an explicit stack) to explore as far as possible down each branch.
2. Starts from a given `start` node, marks it as visited, and prints it.
3. For each unvisited neighbor of the current node, it recursively calls DFS on that neighbor.

### Code (`dfs.py`)
```python
def dfs(graph, start, visited=None):
    if visited is None:
        visited = set()
    visited.add(start)
    print(start, end=" ")

    for neighbor in graph[start]:
        if neighbor not in visited:
            dfs(graph, neighbor, visited)

# Example usage:
graph = {
    'A': ['B', 'C'],
    'B': ['D', 'E'],
    'C': ['F'],
    'D': [],
    'E': [],
    'F': []
}
print("\nDFS traversal starting from node 'A':")
dfs(graph, 'A')
```

### How to Run
```bash
python dfs.py
```

### Expected Output
```

DFS traversal starting from node 'A':
A B D E C F 
```