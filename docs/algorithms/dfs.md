```markdown
# Depth-First Search (DFS)

This module (`dfs.py`) implements the Depth-First Search (DFS) algorithm, another fundamental algorithm for traversing or searching tree or graph data structures. It explores as far as possible along each branch before backtracking.

## Algorithm Details

DFS starts at a specified source node and explores as far as possible along each branch before exploring other branches. It typically uses a stack data structure or recursion to manage the nodes to visit. This implementation uses recursion.

### Graph Representation

The graph is represented as an adjacency list using a Python dictionary, where each key represents a node and its value is a list of its direct neighbors.

## Example Usage

To perform a DFS traversal, execute the `dfs.py` script. It will traverse a predefined graph starting from node 'A'.

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

## Expected Output

```

DFS traversal starting from node 'A':
A B D E C F 
```
