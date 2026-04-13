import plotly.io as pio
import numpy as np
from algorithms.genetic_algorithm import GeneticAlgorithm
from viz.plotting import (
    create_population_heatmap,
    create_fitness_distribution,
    create_convergence_plot
)


def run_onemax_with_viz():
    ga = GeneticAlgorithm(
        population_size=200,
        genome_length=100,
        mutation_rate=0.015,
        crossover_rate=0.8,
        elitism=5
    )

    ga.initialize_population()

    print("Starting evolution...")

    for gen in range(150):
        ga.evolve_one_generation()

        if gen % 10 == 0 or gen == 149:
            best = max(ga.population, key=lambda ind: ind.fitness)
            print(f"Gen {gen:3d} | Best: {best.fitness:3d}/{ga.genome_length} | "
                  f"Mean: {np.mean([ind.fitness for ind in ga.population]):.1f}")

            # Generate plots
            if gen % 30 == 0 or gen == 149:   # less frequent for speed
                heatmap = create_population_heatmap(ga.population, f"Generation {gen}")
                dist = create_fitness_distribution(ga.population, gen)

                # For now, save HTML
                pio.write_html(heatmap, f"viz_output/heatmap_gen_{gen}.html")
                pio.write_html(dist, f"viz_output/dist_gen_{gen}.html")

    # Final convergence plot
    final_conv = create_convergence_plot(ga.best_history, "OneMax Convergence")
    pio.write_html(final_conv, "viz_output/final_convergence.html")
    print("Visualization files saved to viz_output/")

    return ga


if __name__ == "__main__":
    import os
    os.makedirs("viz_output", exist_ok=True)
    run_onemax_with_viz()
