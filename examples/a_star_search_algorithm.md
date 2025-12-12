# A* Search Algorithm

This example implements the A* (A-star) search algorithm, a popular pathfinding and graph traversal algorithm that finds the shortest path between a starting node and a goal node in a graph. It uses a heuristic function to estimate the cost from the current node to the goal, making it more efficient than Dijkstra's algorithm for many problems.

## Algorithm Overview

A* search combines principles of Dijkstra's algorithm (which guarantees the shortest path) and greedy best-first search (which expands nodes closest to the goal). It evaluates nodes using a cost function `f(n) = g(n) + h(n)`:

-   `g(n)`: The cost of the path from the start node to `n`.
-   `h(n)`: The estimated cost (heuristic) from `n` to the goal node.

The algorithm maintains a set of discovered nodes to be evaluated, ordered by `f(n)`, and iteratively expands the node with the lowest `f(n)` until the goal is reached.

## Code Explanation

1.  **Imports**: `heapq` is used for efficient management of the `open_set` (priority queue) of nodes to visit.
2.  **`heuristic(a, b)` function**: Calculates the Manhattan distance between two points `a` and `b`. This acts as an admissible heuristic for a grid (it never overestimates the true cost).
3.  **`astar(grid, start, goal)` function**: This is the core A* implementation.
    *   `open_set`: A min-heap storing tuples `(f_score, node)` for nodes to be explored.
    *   `came_from`: A dictionary to reconstruct the path.
    *   `g_score`: A dictionary storing the cost from the start to each node.
    *   The algorithm iteratively picks the node with the lowest `f_score` from `open_set`.
    *   It explores neighbors of the current node, updates their `g_score` and `f_score` if a shorter path is found, and adds them to `open_set`.
    *   If the `goal` is reached, it reconstructs and returns the path.
4.  **Example Usage**: Defines a `grid` (a 2D array where `0` is a walkable path and `1` is an obstacle), `start` and `goal` coordinates, and then calls the `astar` function.

## How to Run

1.  Save the code as `astar.py`.
2.  Run the script from your terminal:
    ```bash
    python astar.py
    ```

## Expected Output

```
Shortest path using A*: [(0, 0), (0, 1), (0, 2), (0, 3), (1, 3), (2, 3), (3, 3)]
```

```python
# A* Search Algorithm Example

from heapq import heappush, heappop

def heuristic(a, b):
    """Heuristic function: Manhattan distance"""
    return abs(a[0] - b[0]) + abs(a[1] - b[1])

def astar(grid, start, goal):
    rows, cols = len(grid), len(grid[0])
    open_set = []
    heappush(open_set, (0, start))
    
    came_from = {}
    g_score = {start: 0}
    
    while open_set:
        _, current = heappop(open_set)

        if current == goal:
            # reconstruct path
            path = []
            while current in came_from:
                path.append(current)
                current = came_from[current]
            path.append(start)
            return path[::-1]

        x, y = current
        for dx, dy in [(1,0), (-1,0), (0,1), (0,-1)]: 
            nx, ny = x + dx, y + dy

            if 0 <= nx < rows and 0 <= ny < cols and grid[nx][ny] != 1:
                neighbor = (nx, ny)
                tentative_g = g_score[current] + 1

                if tentative_g < g_score.get(neighbor, float('inf')):
                    g_score[neighbor] = tentative_g
                    f_score = tentative_g + heuristic(neighbor, goal)
                    heappush(open_set, (f_score, neighbor))
                    came_from[neighbor] = current

    return None

# Example usage
if __name__ == "__main__":
    grid = [
        [0, 0, 0, 0],
        [0, 1, 1, 0],
        [0, 0, 0, 0],
        [0, 1, 0, 0]
    ]
    
    start = (0, 0)
    goal = (3, 3)
    
    path = astar(grid, start, goal)
    print("Shortest path using A*:", path)
```