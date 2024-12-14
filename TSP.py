import random
import numpy as np

# Define the distance matrix
def create_symmetric_distance_matrix(num_cities):
    # Create an empty matrix for distances
    distance_matrix = np.zeros((num_cities, num_cities), dtype=int)
    
    # Fill in the upper triangle of the matrix with random distances
    for i in range(num_cities):
        for j in range(i + 1, num_cities):
            distance = np.random.randint(10, 100)
            distance_matrix[i][j] = distance
            distance_matrix[j][i] = distance  # Ensure symmetry

    np.fill_diagonal(distance_matrix, 0)  # Distance from a city to itself is zero
    return distance_matrix

# Generate a symmetric sample distance matrix for testing
num_cities = 10
distance_matrix = create_symmetric_distance_matrix(num_cities)

# Initialize population with unique random tours
def initialize_population(pop_size, num_cities):
    population = []
    while len(population) < pop_size:
        tour = random.sample(range(num_cities), num_cities)  # Random permutation of city indices
        if tour not in population:  # Check for uniqueness
            population.append(tour)
    return population

# Calculate total distance of a tour
def calculate_distance(tour, distance_matrix):
    total_distance = 0
    for i in range(len(tour) - 1):
        total_distance += distance_matrix[tour[i], tour[i + 1]]
    total_distance += distance_matrix[tour[-1], tour[0]]  # Return to the start city
    return total_distance

# Evaluate fitness for each tour
def evaluate_fitness(population, distance_matrix):
    fitness_scores = []
    for tour in population:
        distance = calculate_distance(tour, distance_matrix)
        fitness = 1 / (distance + 1e-10)  # Avoid division by zero for zero distance
        fitness_scores.append(fitness)
    return fitness_scores

# Select parents based on fitness (Roulette Wheel Selection)
def select_parents(population, fitness_scores):
    total_fitness = sum(fitness_scores)
    selection_probs = [f / total_fitness for f in fitness_scores]
    parent1 = population[np.random.choice(len(population), p=selection_probs)]
    parent2 = population[np.random.choice(len(population), p=selection_probs)]
    return parent1, parent2

# Crossover to create offspring (Ordered Crossover - OX)
def crossover(parent1, parent2):
    start, end = sorted(random.sample(range(len(parent1)), 2))
    child = [-1] * len(parent1)
    child[start:end] = parent1[start:end]

    parent2_remaining = [gene for gene in parent2 if gene not in child]
    child = [parent2_remaining.pop(0) if gene == -1 else gene for gene in child]
    return child

# Mutation to introduce diversity (Swap Mutation)
def mutate(tour, mutation_rate=0.01):
    if random.random() < mutation_rate:
        i, j = random.sample(range(len(tour)), 2)
        tour[i], tour[j] = tour[j], tour[i]
    return tour

# Genetic Algorithm for TSP
def genetic_algorithm_tsp(distance_matrix, pop_size, num_generations):
    num_cities = len(distance_matrix)
    population = initialize_population(pop_size, num_cities)

    for generation in range(num_generations):
        fitness_scores = evaluate_fitness(population, distance_matrix)
        
        # Generate the new population
        new_population = []
        for _ in range(pop_size // 2):
            parent1, parent2 = select_parents(population, fitness_scores)
            child1 = mutate(crossover(parent1, parent2))
            child2 = mutate(crossover(parent2, parent1))
            new_population.extend([child1, child2])
        
        # Update population
        population = new_population

        # Optional: Print the best solution in each generation
        best_fitness = max(fitness_scores)
        best_distance = 1 / best_fitness
        print(f"Generation {generation + 1}: Best Distance = {best_distance}")

    # Evaluate final population fitness and get the best solution
    fitness_scores = evaluate_fitness(population, distance_matrix)
    best_index = fitness_scores.index(max(fitness_scores))
    best_tour = population[best_index]
    return best_tour, 1 / max(fitness_scores)

# Example usage
pop_size = 50
num_generations = 100

best_tour, best_distance = genetic_algorithm_tsp(distance_matrix, pop_size, num_generations)
print("Best tour:", best_tour)
print("Best distance:", best_distance)
