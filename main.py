import numpy as np


def y(x1, x2):
    return 100*(x1**2 - x2)**2 + (1 - x1)**2


def calc_fitness(f, population):
    return f(population[:, 0], population[:, 1])

def ga_search(f, n_params, scope, num_generations=100, population_size=8):
    population = np.random.uniform(low=scope[0], high=scope[1], size=(population_size, n_params))

    for generation in range(num_generations):
        fitness = calc_fitness(f, population)
        print(fitness)
        pass



if __name__ == '__main__':
    ga_search(f=y, n_params=2, scope=(-2.048, 2.048), num_generations=100)