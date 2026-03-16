import numpy as np


class Population:
    def __init__(self, individuals: np.ndarray):
        self.individuals = individuals
        self.fitness = None
        self.generation = 0
