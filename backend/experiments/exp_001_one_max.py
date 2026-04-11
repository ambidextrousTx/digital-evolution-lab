from algorithms.genetic_algorithm import GeneticAlgorithm


def run_onemax_experiment():
    ga = GeneticAlgorithm(
        population_size=200,
        genome_length=100,
        mutation_rate=0.015,
        crossover_rate=0.8,
        elitism=5
    )

    ga.initialize_population()

    # 150 generations
    for gen in range(150):
        ga.evolve_one_generation()
        # Print every 10th generation
        if gen % 10 == 0:
            best = max(ga.population, key=lambda individual: individual.fitness)
            print(f"Gen {gen:3d} | Best fitness: {best.fitness:3d}/{ga.genome_length}")

    return ga


if __name__ == "__main__":
    run_onemax_experiment()
