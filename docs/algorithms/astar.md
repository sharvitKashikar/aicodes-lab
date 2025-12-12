```markdown
# A* Search Algorithm

This module (`astar.py`) provides an implementation of the A* (A-star) search algorithm, a widely used pathfinding and graph traversal algorithm. It intelligently explores a graph by combining features of Dijkstra's algorithm and greedy best-first search, guaranteeing to find the shortest path in an unweighted graph while also being efficient.

## Algorithm Details

The A* algorithm finds the shortest path between a starting node and a goal node in a graph. It uses a heuristic function to estimate the cost from the current node to the goal. The total cost `f(n)` for a node `n` is calculated as:

`f(n) = g(n) + h(n)`

where:
-   `g(n)` is the cost from the start node to `n`.
-   `h(n)` is the heuristic estimate of the cost from `n` to the goal node.

### Heuristic Function

The implementation uses the **Manhattan distance** as its heuristic function. For two points `(x1, y1)` and `(x2, y2)`, the Manhattan distance is `|x1 - x2| + |y1 - y2|`. This is suitable for grid-based pathfinding where movement is restricted to horizontal and vertical directions.

### Grid Representation

The search space is represented as a 2D list (grid) where:
-   `0` represents an open, traversable cell.
-   `1` represents an obstacle (wall).

## Example Usage

To see the A* algorithm in action, execute the `astar.py` script directly. It will solve a predefined grid pathfinding problem.

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

## Expected Output

```
Shortest path using A*: [(0, 0), (0, 1), (0, 2), (0, 3), (1, 3), (2, 3), (2, 2), (2, 1), (3, 1), (3, 2), (3, 3)]
```
