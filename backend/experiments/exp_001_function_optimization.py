import numpy as np
from core.population import Population
from algorithms.genetic_algorithm import run_ga, fitness_function


pop_size = 100
genome_dim = 1

if __name__ == "__main__":
    genomes = np.random.uniform(-10, 10, (pop_size, genome_dim))

    population = Population(genomes)

    run_ga(population, fitness_function, generations=200)
