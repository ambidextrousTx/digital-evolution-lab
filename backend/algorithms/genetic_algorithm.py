import numpy as np


def mutate(x, sigma=0.1):
    return x + np.random.normal(0, sigma)


def crossover(a, b):
    alpha = np.random.rand()
    return alpha * a + (1 - alpha) * b


def fitness(x):
    return x * np.sin(10 * x) + x * np.cos(2 * x)
