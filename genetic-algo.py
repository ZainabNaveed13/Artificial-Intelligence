import random

# Initialize the population with random binary strings
def initialize_population(pop_size, string_length):
    return [
        ''.join(random.choice(['0', '1']) for _ in range(string_length))
        for _ in range(pop_size)
    ]

# Calculate fitness for an individual
def calculate_fitness(individual):
    return individual.count('1')

# Select parents based on fitness
def select_parents(population, fitness_scores):
    total_fitness = sum(fitness_scores)
    probabilities = [score / total_fitness for score in fitness_scores]
    parent1 = random.choices(population, weights=probabilities, k=1)[0]
    parent2 = random.choices(population, weights=probabilities, k=1)[0]
    return parent1, parent2

# Perform crossover to generate offspring
def crossover(parent1, parent2):
    crossover_point = random.randint(1, len(parent1) - 1)
    child = parent1[:crossover_point] + parent2[crossover_point:]
    return child

# Apply mutation to introduce diversity
def mutate(individual, mutation_rate):
    mutated = ''.join(
        bit if random.random() > mutation_rate else '1' if bit == '0' else '0'
        for bit in individual
    )
    return mutated

# Genetic Algorithm implementation
def genetic_algorithm(string_length, pop_size, num_generations, mutation_rate):
    # Initialize population
    population = initialize_population(pop_size, string_length)

    for generation in range(num_generations):
        # Evaluate fitness
        fitness_scores = [calculate_fitness(ind) for ind in population]
        
        # Find the best solution
        best_solution = max(population, key=calculate_fitness)
        best_fitness = calculate_fitness(best_solution)
        print(f"Generation {generation}: Best Fitness = {best_fitness}, Best String = {best_solution}")
        
        # Create the next generation
        new_population = []
        for _ in range(pop_size // 2):  # Generate pairs of offspring
            parent1, parent2 = select_parents(population, fitness_scores)
            child1 = crossover(parent1, parent2)
            child2 = crossover(parent2, parent1)
            child1 = mutate(child1, mutation_rate)
            child2 = mutate(child2, mutation_rate)
            new_population.extend([child1, child2])

        # Replace old population with new one
        population = new_population

    # Return the best solution found
    return max(population, key=calculate_fitness)

# Parameters
string_length = 10
pop_size = 20
num_generations = 50
mutation_rate = 0.01

# Run the Genetic Algorithm
optimal_solution = genetic_algorithm(string_length, pop_size, num_generations, mutation_rate)
print(f"Optimal Solution: {optimal_solution}")
