import math
import numpy as np

def fitness_function(x):
    return np.sum(x**2)

def mfo(function, dimension, n_moths, max_iter, lb, ub):
    moths_position = np.random.uniform(lb, ub, (n_moths, dimension))
    moths_position = np.clip(moths_position, lb, ub)
    moths_fitness = np.array([function(i) for i in moths_position])

    flame_num = n_moths
    b = 1

    for l in range(1, max_iter + 1):
        # sap xep flame theo fitness
        sorted_indices = np.argsort(moths_fitness)  # tra ve thu tu index cua mang sau sap xep
        flames_position = moths_position[sorted_indices][ :flame_num]  # sap xep vi tri flames thanh mang tang dan dua vao index o tren

        for i in range(n_moths):
            # cap nhat moth
            t = np.random.uniform(-1, 1)
            distance = np.linalg.norm(moths_position[i] - flames_position[i % flame_num], ord=2)
            moths_position[i] = distance * math.exp(b * t) * np.cos(2 * np.pi * t) + flames_position[i % flame_num]

        #cap nhat moth flame
        moths_fitness = np.array([function(i) for i in moths_position])

        # cap nhat flame_num
        flame_num = int(np.round(n_moths - (l * ((n_moths - 1) / max_iter))))

        # Giới hạn giá trị trong [lb, ub]
        moths_position = np.clip(moths_position, lb, ub)

    sorted_indices = np.argsort(moths_fitness)
    sorted_moth =  moths_position[sorted_indices]
    return sorted_moth[0], function(sorted_moth[0])

# dimension = 2
# n_moths = 3
# max_iter= 3
# lb = -10
# ub = 10
# moths_position = np.array([[4, 3], [1, -2], [0.5, 0.5]])
pos, fitness = mfo(fitness_function, 2, 3, 1000, -10, 10)
print(pos)
print(fitness)