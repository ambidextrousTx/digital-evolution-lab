import numpy as np
from typing import Callable


class Individual:
    '''Represents one candidate organism
    in the population'''

    def __init__(self, genome: np.ndarray):
        self.genome = genome
        self.fitness = 0

    def evaluate(self, fitness_function: Callable):
        self.fitness = fitness_function(self.genome)
