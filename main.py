import numpy as np
import GA
import ANN
from dataset.mnist import load_mnist
import matplotlib.pyplot as plt
import time

(x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, one_hot_label=True)

print(x_train.shape)
sol_per_pop = 8

num_parent_mating = 4

num_generations = 5000

mutation_rate = 10

ini_pop_weight = []
start = time.time()
# Creating the initial population
for current_sol in range(sol_per_pop):
    HL1 = 50
    HL1_weight = np.random.uniform(low=-0.1, high=0.1, size=(x_train.shape[1], HL1))

    HL2 = 50
    HL2_weight = np.random.uniform(low=-0.1, high=0.1, size=(HL1, HL2))

    outsize = 10
    out_weight = np.random.uniform(low=-0.1, high=0.1, size=(HL2, outsize))

    ini_pop_weight.append(np.array([HL1_weight, HL2_weight, out_weight]))

pop_matrix = np.array(ini_pop_weight)
pop_vector = GA.matrix_to_vector(pop_matrix)
# print(pop_vector.shape)
best_outputs = []

accuracies = np.empty(shape=num_generations)

for generation in range(num_generations):
    print("Generation: ", generation)

    pop_matrix = GA.vector_to_matrix(pop_vector, pop_matrix)

    # Calculate fitness of each chromosome in population
    batch_size =1000
    batch_mask = np.random.choice(x_train.shape[0], batch_size)
    x_batch = x_train[batch_mask]
    t_batch = t_train[batch_mask]
    fitness = ANN.fitness(pop_matrix, x_batch, t_batch)

    accuracies[generation] = fitness[0]

    print("Fitness:", fitness)

    # Select the best parents in pool
    parents = GA.select_parent(pop_vector, fitness.copy(), num_parent_mating)

    # print("Parents", parents)

    # Generate offspring
    offspring_crossover = GA.crossover(parents,
                                       offspring_size=(pop_vector.shape[0] - parents.shape[0], pop_vector.shape[1]))
    # print("Crossover", offspring_crossover)

    # Mutation
    offspring_mutation = GA.mutation(offspring_crossover, mutation_rate)
    # print("Mutation", offspring_mutation)

    pop_vector[0:parents.shape[0], :] = parents
    pop_vector[parents.shape[0]:, :] = offspring_mutation

pop_matrix = GA.vector_to_matrix(pop_vector, pop_matrix)
end = time.time()
best_weight = pop_matrix[0, :]
ann = ANN.ANN(best_weight)
acc = ann.predict(x_test, t_test)
print(end - start)
plt.plot(accuracies)
print(acc)
plt.xlabel("Iterations", fontsize=20)
plt.ylabel("Fitness", fontsize=20)

plt.show()
