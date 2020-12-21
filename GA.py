import numpy as np

import random


def matrix_to_vector(pop_matrix):
    pop_vector = []

    for sol_index in range(pop_matrix.shape[0]):
        curr_vec = []
        for layer_index in range(pop_matrix.shape[1]):
            tmpvec = np.reshape(pop_matrix[sol_index, layer_index], newshape=pop_matrix[sol_index, layer_index].size)
            curr_vec.extend(tmpvec)

        pop_vector.append(curr_vec)
    return np.array(pop_vector)


def vector_to_matrix(pop_vector, pop_matrix):
    matrix = []

    for sol_index in range(pop_matrix.shape[0]):
        start = 0
        end = 0
        for layer_index in range(pop_matrix.shape[1]):
            end += pop_matrix[sol_index, layer_index].size
            cur_vec = pop_vector[sol_index, start:end]
            layer_matrix = np.reshape(cur_vec, newshape=pop_matrix[sol_index, layer_index].shape)
            matrix.append(layer_matrix)
            start = end
    return np.reshape(matrix, newshape=pop_matrix.shape)


def select_parent(pop, fitness, num_parents):
    parents = np.empty((num_parents, pop.shape[1]))

    for parent_index in range(num_parents):
        best_index = np.where(fitness == np.max(fitness))
        best_index = best_index[0][0]
        parents[parent_index, :] = pop[best_index, :]
        fitness[best_index] = -999999999999999999
    return parents


def crossover(parents, offspring_size):
    offspring = np.empty(offspring_size)

    crossover_point = np.uint8(offspring_size[1] / 2)

    for k in range(offspring_size[0]):
        parent_1_index = k % parents.shape[0]

        parent_2_index = (k + 1) % parents.shape[0]

        offspring[k, 0:crossover_point] = parents[parent_1_index, 0:crossover_point]
        offspring[k, crossover_point:] = parents[parent_2_index, crossover_point:]
    return offspring


def mutation(offspring_crossover, rate):
    num_mutations = np.uint8(rate * offspring_crossover.shape[1] / 100)

    mutation_indices = np.array(random.sample(range(0, offspring_crossover.shape[1]), num_mutations))

    for index in range(offspring_crossover.shape[0]):
        r = np.random.uniform(-1.0, 1.0, 1)
        offspring_crossover[index, mutation_indices] += r

    return offspring_crossover
