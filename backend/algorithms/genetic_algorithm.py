import numpy as np
from concurrent.futures import ProcessPoolExecutor


def mutate_gaussian(genomes, sigma=0.1):
    '''Vectorized no-loop mutation'''
    noise = np.random.normal(0, sigma, genomes.shape)
    return genomes + noise


def perform_individual_crossover(parent_a, parent_b):
    alpha = np.random.rand()
    child = alpha * parent_a + (1 - alpha) * parent_b
    return child


def perform_population_crossover(genomes):
    genomes = genomes.copy()
    np.random.shuffle(genomes)
    children = []
    for i in range(0, len(genomes), 2):
        a = genomes[i]
        b = genomes[i+1]
        child1 = perform_individual_crossover(a, b)
        child2 = perform_individual_crossover(b, a)
        children.append(child1)
        children.append(child2)

    return np.array(children)


def fitness_function(genome):
    x = genome[0]
    return x * np.sin(10 * x) + x * np.cos(2 * x)


def evaluate_population(genomes, fitness_function, workers=8):
    with ProcessPoolExecutor(max_workers=workers) as executor:
        fitness = executor.map(fitness_function, genomes)

    return np.array(list(fitness))


def select(genomes, fitness, k=3):
    '''Perform tournament selection'''
    population = len(genomes)
    selected = []

    for _ in range(population):
        idx = np.random.choice(population, k, replace=False)
        best = idx[np.argmax(fitness[idx])]
        selected.append(genomes[best])

    return np.array(selected)


def run_ga(population, fitness_fn, generations):
    '''
    Example output:
    Gen 1   best=2.4  mean=0.2
    Gen 10  best=5.8  mean=3.1
    Gen 50  best=7.9  mean=7.3
    '''
    for g in range(generations):
        fitness = evaluate_population(population.individuals, fitness_fn)

        # Keep the best citizens intact
        elite_size = 2
        elite_indices = np.argsort(fitness)[-elite_size:]
        elites = population.individuals[elite_indices]

        population.fitness = fitness
        selected = select(population.individuals, fitness)
        offspring = perform_population_crossover(selected)
        sigma = max(0.01, 0.1 * (0.99 ** g))
        mutated = mutate_gaussian(offspring, sigma=sigma)
        mutated[:elite_size] = elites

        population.individuals = mutated
        population.generation += 1

        best_fitness = np.max(fitness)
        mean_fitness = np.mean(fitness)

        print(f'Gen {population.generation}\tbest={best_fitness:.4f}\tmean={mean_fitness:.4f}')

        best_idx = np.argmax(fitness)
        best_genome = population.individuals[best_idx]

        print(f'Gen {population.generation}\tbest={best_fitness:.4f}\tx={best_genome[0]:.4f}')

    return population
