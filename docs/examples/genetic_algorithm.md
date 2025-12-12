# Genetic Algorithm Example

This document describes the `genetic.py` script, which implements a basic Genetic Algorithm to find the maximum value of the function f(x) = x^2 within a given range.

## Purpose

Genetic Algorithms are a class of optimization algorithms inspired by natural selection. They are used to find optimal or near-optimal solutions to difficult problems by iteratively evolving potential solutions. This script provides a simple, self-contained example demonstrating the core concepts of a genetic algorithm.

## How it Works

The algorithm follows these steps:

1.  **Initialization**: An initial population of random solutions (represented by 'x' values) is generated.
2.  **Fitness Evaluation**: Each solution in the population is evaluated by a fitness function, which quantifies how 'good' the solution is. In this example, the fitness function is `f(x) = x^2`.
3.  **Selection**: A subset of the best-performing solutions (parents) is selected from the current population.
4.  **Crossover (Recombination)**: New solutions (children) are created by combining genetic information from two selected parents.
5.  **Mutation**: Random, small changes are introduced into the children to maintain genetic diversity and explore new solution spaces.
6.  **Iteration**: Steps 2-5 are repeated for a specified number of generations, with the population continuously evolving towards better solutions.

## Core Components

The `genetic.py` script defines the following key functions:

### `fitness(x)`

Calculates the fitness score for a given individual `x`. For this example, it returns `x**2`.

```python
def fitness(x):
    return x**2
```

### `generate_population(size)`

Creates an initial population of `size` individuals. Each individual `x` is a random float between -10 and 10.

```python
def generate_population(size):
    return np.random.uniform(-10, 10, size)
```

### `selection(pop)`

Selects the best half of the current population based on their fitness scores to become parents for the next generation.

```python
def selection(pop):
    scores = np.array([fitness(x) for x in pop])
    selected_idx = scores.argsort()[-len(pop)//2:]
    return pop[selected_idx]
```

### `crossover(parents, size)`

Generates a new generation of `size` individuals by blending two randomly chosen parents. The child's value is the average of the two parents.

```python
def crossover(parents, size):
    next_gen = []
    for _ in range(size):
        p1, p2 = np.random.choice(parents, 2)
        child = (p1 + p2) / 2
        next_gen.append(child)
    return np.array(next_gen)
```

### `mutate(pop, rate=0.1)`

Introduces slight random changes to individuals in the population with a given `rate`. This helps prevent premature convergence and explores new areas of the solution space.

```python
def mutate(pop, rate=0.1):
    for i in range(len(pop)):
        if np.random.rand() < rate:
            pop[i] += np.random.uniform(-1, 1)
    return pop
```

## Running the Script

To run this genetic algorithm example, simply execute the `genetic.py` file:

```bash
python genetic.py
```

The script will print the best `x` value and its corresponding `f(x)` for each generation, concluding with the final best solution found.

## Source Code (`genetic.py`)

```python
import numpy as np

# Fitness function: maximize x^2
def fitness(x):
    return x**2

# Generate initial population
def generate_population(size):
    return np.random.uniform(-10, 10, size)

# Selection: pick best half
def selection(pop):
    scores = np.array([fitness(x) for x in pop])
    selected_idx = scores.argsort()[-len(pop)//2:]
    return pop[selected_idx]

# Crossover: blend two parents
def crossover(parents, size):
    next_gen = []
    for _ in range(size):
        p1, p2 = np.random.choice(parents, 2)
        child = (p1 + p2) / 2
        next_gen.append(child)
    return np.array(next_gen)

# Mutation: slight random change
def mutate(pop, rate=0.1):
    for i in range(len(pop)):
        if np.random.rand() < rate:
            pop[i] += np.random.uniform(-1, 1)
    return pop

# Genetic Algorithm
population_size = 10
generations = 20

population = generate_population(population_size)

for gen in range(generations):
    population = selection(population)
    population = crossover(population, population_size)
    population = mutate(population)
    best = max(population, key=fitness)
    print(f"Generation {gen+1} → Best x: {best:.4f}, f(x): {fitness(best):.4f}")

print("\nFinal Answer:")
print("Best solution:", best)
print("Maximum value f(x) =", fitness(best))
```