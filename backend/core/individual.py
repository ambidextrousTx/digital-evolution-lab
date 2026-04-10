import numpy as np
from typing import Callable


class Individual:
    '''Represents one candidate organism
    in the population'''

    def __init__(self, genome: np.ndarray):
        self.genome = genome
        self.fitness = None

    def evaluate(self, fitness_function: Callable):
        if self.fitness is None:
            self.fitness = fitness_function(self.genome)

        return self.fitness
