from dataclasses import dataclass
import numpy as np


@dataclass
class Individual:
    '''A genome is always a Numpy vector.
    The framework is dimension-agnostic'''
    genome: np.ndarray
    fitness: float | None = None
