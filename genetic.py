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
