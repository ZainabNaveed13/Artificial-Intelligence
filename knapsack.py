import random

# Step 1: Define Problem Variables
weights = [10, 20, 30, 15,25]  # Weight of each item
values = [60,100,120,75,90]  # Value of each item
max_weight = 50  # Max weight of knapsack
population_size = 10  # Number of solutions in population
generations = 50  # Number of generations to evolve

# Step 2: Initialize a Population
def create_individual():
    """Creates a random individual (a solution to the knapsack)."""
    return [random.randint(0, 1) for _ in range(len(weights))]

def create_population():
    """Creates a population of individuals."""
    return [create_individual() for _ in range(population_size)]

# Step 3: Fitness Function
def fitness(individual):
    """Calculates fitness as total value if weight constraint is met, otherwise 0."""
    total_weight = sum([individual[i] * weights[i] for i in range(len(individual))])
    total_value = sum([individual[i] * values[i] for i in range(len(individual))])
    if total_weight <= max_weight:
        return total_value
    else:
        return 0

# Step 4: Selection
def selection(population):
    """Select individuals with better fitness scores."""
    fitness_scores = [(fitness(ind), ind) for ind in population]
    fitness_scores.sort(reverse=True, key=lambda x: x[0])
    return [ind for _, ind in fitness_scores[:population_size // 2]]

# Step 5: Crossover
def crossover(parent1, parent2):
    """Perform crossover between two parents to create offspring."""
    crossover_point = random.randint(0, len(parent1) - 1)
    return parent1[:crossover_point] + parent2[crossover_point:]

# Step 6: Mutation
def mutate(individual):
    """Randomly mutate an individual."""
    mutation_point = random.randint(0, len(individual) - 1)
    individual[mutation_point] = 1 - individual[mutation_point]  # Flip 0 to 1 or 1 to 0

# Step 7: Genetic Algorithm Main Loop
def genetic_algorithm():
    population = create_population()
    for _ in range(generations):
        selected = selection(population)
        next_generation = []

        while len(next_generation) < population_size:
            parent1, parent2 = random.sample(selected, 2)
            child = crossover(parent1, parent2)
            mutate(child)
            next_generation.append(child)

        population = next_generation

    # Get the best solution in the final population
    best_individual = max(population, key=fitness)
    return best_individual, fitness(best_individual)

# Run the Genetic Algorithm
best_solution, best_value = genetic_algorithm()
print("Best solution:", best_solution)
print("Best value:", best_value)
