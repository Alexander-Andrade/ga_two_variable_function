import numpy as np
import random
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm


def y(x1, x2):
    return 100*(x1**2 - x2)**2 + (1 - x1)**2


def calc_fitness(f, population):
    return f(population[:, 0], population[:, 1])


def pairs(iterable, n):
    size = len(iterable)
    count = 0

    while True:
        ids = np.random.permutation(size).tolist() + np.random.permutation(size).tolist()
        ids_len = size << 1
        for i in range(0, ids_len, 2):
            yield (iterable[ids[i]], iterable[ids[i+1]])
            count += 1
            if count >= n:
                return


def flat_crossover(parents, n_children):
    children = np.empty((n_children, parents.shape[1]), dtype=float)

    for i, pair in enumerate(pairs(parents, n_children)):
        children[i] = random.uniform(min(pair[0][0], pair[1][0]), max(pair[0][0], pair[1][0])),\
                      random.uniform(min(pair[0][1], pair[1][1]), max(pair[0][1], pair[1][1]))

    return children


def mutation(children, mutation_prob, scope):
    mutated_children = np.copy(children)

    mutation_map = np.random.random(size=children.shape)
    ids = np.nonzero(mutation_map < mutation_prob)
    for i, j in zip(ids[0], ids[1]):
        mutated_children[i][j] = random.uniform(scope[0], scope[1])

    return mutated_children


def reduce_population(parents, children, pop_size):
    n_grandparents = pop_size - children.shape[0]
    new_population = np.empty((pop_size, parents.shape[1]))
    new_population[:n_grandparents, :] = parents[:n_grandparents]
    new_population[n_grandparents:,:] = children
    return new_population


def ga_search(f, n_params, scope, num_generations=100, population_size=8,
              n_parents=4, mutation_prob=0.01, n_grandparents=2):
    population = np.random.uniform(low=scope[0], high=scope[1], size=(population_size, n_params))

    for generation in range(num_generations):
        fitness = calc_fitness(f, population)
        population = population[np.flip(fitness.argsort(), axis=0)]
        print(fitness)
        if generation == num_generations - 1:
            return population[0]
        parents = population[:n_parents]
        children = flat_crossover(parents, population_size - n_grandparents)
        children = mutation(children, mutation_prob, scope)
        population = reduce_population(parents, children, population_size)


if __name__ == '__main__':
    res = ga_search(f=y, n_params=2, scope=(-2.048, 2.048), num_generations=100)

    fig = plt.figure()
    ax = fig.gca(projection='3d')

    X1 = np.arange(-2.048, 2.048, 0.25)
    X2 = np.arange(-2.048, 2.048, 0.25)
    X1, X2 = np.meshgrid(X1, X2)
    Y = y(X1, X2)

    surf = ax.plot_surface(X1, X2, Y, cmap=cm.coolwarm,
                           linewidth=0, antialiased=False)

    fig.colorbar(surf, shrink=0.5, aspect=5)

    ax.plot([res[0]], [res[1]], zs=y(res[0], res[1]), markersize=6, marker='o', color='red')

    plt.show()