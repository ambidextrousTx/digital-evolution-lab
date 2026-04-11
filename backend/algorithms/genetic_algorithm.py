import numpy as np
from typing import List, Tuple
from core.individual import Individual
import random


class GeneticAlgorithm:
    def __init__(self,
                 population_size: int = 100,
                 genome_length: int = 50,
                 mutation_rate: float = 0.01,
                 crossover_rate: float = 0.7,
                 elitism: int = 2):
        self.population_size = population_size
        self.genome_length = genome_length
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.elitism = elitism
        self.population: List[Individual] = []
        self.generation = 0
        self.best_history = []

    def initialize_population(self):
        """Random binary genome - like primordial soup"""
        genomes = np.random.randint(0, 2, size=(self.population_size,
                                                self.genome_length))
        self.population = [Individual(genome) for genome in genomes]

    def fitness_one_max(self, genome: np.ndarray) -> float:
        """Number of 1s - simple fitness computation"""
        return np.sum(genome)

    def select_parent(self) -> Individual:
        """Tournament selection"""
        tournament = random.sample(self.population, 3)
        return max(tournament, key=lambda individual: individual.fitness)

    def crossover(self, parent1: Individual,
                  parent2: Individual) -> Tuple[Individual, Individual]:
        """Single-point crossover"""
        if random.random() > self.crossover_rate:
            return parent1, parent2  # No crossover

        point = random.randint(1, self.genome_length - 1)
        child1_genome = np.concatenate([parent1.genome[:point],
                                        parent2.genome[point:]])
        child2_genome = np.concatenate([parent2.genome[:point],
                                        parent1.genome[point:]])
        return Individual(child1_genome), Individual(child2_genome)

    def mutate(self, individual: Individual):
        """Bit-flip mutation - random genetic variation"""
        for i in range(self.genome_length):
            if random.random() < self.mutation_rate:
                individual.genome[i] = 1 - individual.genome[i]

    def evolve_one_generation(self):
        """One full generation of selection + reproduction + mutation"""
        # Evaluate everyone
        for individual in self.population:
            individual.evaluate(self.fitness_one_max)

        # Record the best
        best = max(self.population, key=lambda individual: individual.fitness)
        self.best_history.append(best.fitness)

        # Elitism - best organisms survive automatically
        new_population = sorted(self.population,
                                key=lambda individual: individual.fitness,
                                reverse=True)[:self.elitism]

        while len(new_population) < self.population_size:
            parent1 = self.select_parent()
            parent2 = self.select_parent()
            child1, child2 = self.crossover(parent1, parent2)

            self.mutate(child1)
            self.mutate(child2)

            new_population.extend([child1, child2])

        # Trim if we overshot due to elitism
        self.population = new_population[:self.population_size]
        self.generation += 1

