# Depth-First Search (DFS)

This example demonstrates the Depth-First Search (DFS) algorithm, a graph traversal algorithm that explores as far as possible along each branch before backtracking. It uses a stack (implicitly, via recursion in this example) to keep track of nodes to visit.

## Algorithm Overview

DFS starts at a specified node and explores as far as possible along each branch before backtracking. It recursively visits unvisited adjacent nodes. When it encounters a node with no unvisited neighbors, it backtracks to the previous node and explores a different branch.

## Code Explanation

1.  **`dfs(graph, start, visited=None)` function**: Takes a graph (represented as an adjacency list) and a starting node, along with an optional `visited` set to track explored nodes.
    *   Initializes `visited` set if not provided (for the initial call).
    *   Marks the `start` node as visited and prints it.
    *   **Recursive Traversal**: For each `neighbor` of the current node, if it has not been `visited`, the `dfs` function is called recursively on that `neighbor`.
3.  **Example Usage**: Defines a sample `graph` as a dictionary (adjacency list) and then calls the `dfs` function starting from node 'A'.

## How to Run

1.  Save the code as `dfs.py`.
2.  Run the script from your terminal:
    ```bash
    python dfs.py
    ```

## Expected Output

```

DFS traversal starting from node 'A':
A B D E C F 
```

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