import math
import numpy as np

def fitness_function(x):
    return np.sum(x**2)

def mfo(function, dimension, n_moths, max_iter, lb, ub):
    moths_position = np.random.uniform(lb, ub, (n_moths, dimension))
    moths_fitness = np.array([function(i) for i in moths_position])
    best_moth = moths_position[np.argmin(moths_fitness)]
    best_score = np.min(moths_fitness)
    flame_num = n_moths
    b = 1

    for l in range(1, max_iter + 1):
        sorted_indices = np.argsort(moths_fitness)
        flames_position = moths_position[sorted_indices][:flame_num]

        for i in range(n_moths):
            t = np.random.uniform(-1, 1)
            distance = np.linalg.norm(moths_position[i] - flames_position[i % flame_num], ord=2)
            moths_position[i] = distance * math.exp(b * t) * math.cos(2 * math.pi * t) + flames_position[i % flame_num]
            moths_position[i] += np.random.normal(0, 0.001, dimension)

        moths_position = np.clip(moths_position, lb, ub)
        moths_fitness = np.array([function(i) for i in moths_position])

        if np.min(moths_fitness) < best_score:
            best_score = np.min(moths_fitness)
            best_moth = moths_position[np.argmin(moths_fitness)]

        flame_num = int(np.round(n_moths - (l * ((n_moths - 1) / max_iter))))

    return best_moth, best_score

pos, fitness = mfo(fitness_function, 2, 5, 1000, -10, 10)
print(pos)
print(fitness)
