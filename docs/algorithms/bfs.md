```markdown
# Breadth-First Search (BFS)

This module (`bfs.py`) implements the Breadth-First Search (BFS) algorithm, a fundamental algorithm for traversing or searching tree or graph data structures. It explores all of the neighbor nodes at the present depth prior to moving on to the nodes at the next depth level.

## Algorithm Details

BFS starts at a specified source node and explores the neighbor nodes first, before moving to the next level neighbors. It typically uses a queue data structure to manage which node to visit next.

### Graph Representation

The graph is represented as an adjacency list using a Python dictionary, where each key represents a node and its value is a list of its direct neighbors.

## Example Usage

To perform a BFS traversal, execute the `bfs.py` script. It will traverse a predefined graph starting from node 'A'.

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

## Expected Output

```
BFS traversal starting from node 'A':
A B C D E F 
```
