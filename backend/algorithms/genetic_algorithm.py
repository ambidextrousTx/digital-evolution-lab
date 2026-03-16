import numpy as np
from multiprocessing import Pool


def mutate_gaussian(genomes, sigma=0.1):
    '''Vectorized no-loop mutation'''
    noise = np.random.normal(0, sigma, genomes.shape)
    return genomes + noise


def perform_individual_crossover(parent_a, parent_b):
    alpha = np.random.rand()
    child = alpha * parent_a + (1 - alpha) * parent_b
    return child


def perform_population_crossover(genomes):
    np.random.shuffle(genomes)
    children = []
    for i in range(0, len(genomes), 2):
        a = genomes[i]
        b = genomes[i+1]
        child1 = perform_population_crossover(a, b)
        child2 = perform_population_crossover(b, a)
        children.append(child1)
        children.append(child2)

    return np.array(children)


def compute_fitness(genome):
    x = genome[0]
    return x * np.sin(10 * x) + x * np.cos(2 * x)


def evaluate_population(genomes, fitness_function, workers=8):
    with Pool(workers) as pool:
        fitness = pool.map(fitness_function, genomes)

    return np.array(fitness)


def select(genomes, fitness, k=3):
    '''Perform tournament selection'''
    population = len(genomes)
    selected = []

    for _ in range(population):
        idx = np.random.choice(population, k, replace=False)
        best = idx[np.argmax(fitness[idx])]
        selected.append(genomes[best])

    return selected


def run_ga(population, fitness_fn, generations):
    for g in range(generations):
        fitness = evaluate_population(population.genomes, fitness_fn)
        population.fitness = fitness
        selected = select(population.genomes, fitness)
        offspring = perform_population_crossover(selected)
        mutated = mutate_gaussian(offspring)
        population.genomes = mutated
        population.generation += 1

    return population
