from deap import benchmarks, algorithms, base, creator, tools
import random
import math

'''
Created on 13-11-2012
@author: wysek
'''

N = 100
Nbar = 50
GEN = 100
U = 0
V = 1

def my_rand():
    return random.random() * (V - U) - (V + U) / 2

def dtlz5(ind, n_objs):
    from functools import reduce
    g = lambda x: sum([(a - 0.5)**2 for a in x])
    gval = g(ind[n_objs-1:])

    theta = lambda x: math.pi / (4.0 * (1 + gval)) * (1 + 2 * gval * x)
    fit = [(1 + gval) * math.cos(math.pi / 2.0 * ind[0]) *
           reduce(lambda x,y: x*y, [math.cos(theta(a)) for a in ind[1:]])]
    for m in reversed(range(1, n_objs)):
        if m == 1:
            fit.append((1 + gval) * math.sin(math.pi / 2.0 * ind[0]))
        else:
            fit.append((1 + gval) * math.cos(math.pi / 2.0 * ind[0]) *
                       reduce(lambda x,y: x*y, [math.cos(theta(a)) for a in ind[1:m-1]], 1) *
                       math.sin(theta(ind[m-1])))
    return fit

def dtlz6(ind, n_objs):
    from functools import reduce
    gval = sum([a**0.1 for a in ind[n_objs-1:]])
    theta = lambda x: math.pi / (4.0 * (1 + gval)) * (1 + 2 * gval * x)
    fit = [(1 + gval) * math.cos(math.pi / 2.0 * ind[0]) *
           reduce(lambda x,y: x*y, [math.cos(theta(a)) for a in ind[1:]])]
    for m in reversed(range(1, n_objs)):
        if m == 1:
            fit.append((1 + gval) * math.sin(math.pi / 2.0 * ind[0]))
        else:
            fit.append((1 + gval) * math.cos(math.pi / 2.0 * ind[0]) *
                       reduce(lambda x,y: x*y, [math.cos(theta(a)) for a in ind[1:m-1]], 1) *
                       math.sin(theta(ind[m-1])))
    return fit

def dtlz7(ind, n_objs):
    gval = 1 + 9.0 / len(ind[n_objs-1:]) * sum([a for a in ind[n_objs-1:]])
    fit = [i for i in ind[:n_objs-1]]
    fit.append((1 + gval) * (n_objs - sum([a / (1.0 + gval) * (1 + math.sin(3 * math.pi * a)) for a in ind[:n_objs-1]])))
    return fit


def main():
    creator.create("FitnessMax", base.Fitness, weights=(-1.0, -1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMax) #@UndefinedVariable
    toolbox = base.Toolbox()
    toolbox.register("attr_float", my_rand)

    toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_float, n=6) #@UndefinedVariable
    toolbox.register("evaluate", benchmarks.zdt1)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("mate", tools.cxSimulatedBinaryBounded, eta=0.5, low=U, up=V)
    toolbox.register("mutate", tools.mutPolynomialBounded, eta=0.5, low=U, up=V, indpb=1)
    toolbox.register("select", tools.selSPEA2)
    #binary tournament selection
    toolbox.register("selectTournament", tools.selTournament, tournsize=2)


    # Step 1 Initialization
    pop = toolbox.population(n=N)
    archive = []
    curr_gen = 1

    while True:
        # Step 2 Fitness assignement
        for ind in pop:
            ind.fitness.values = toolbox.evaluate(ind)

        for ind in archive:
            ind.fitness.values = toolbox.evaluate(ind)

        # Step 3 Environmental selection
        archive  = toolbox.select(pop + archive, k=Nbar)

        # Step 4 Termination
        if curr_gen >= GEN:
            final_set = archive
            break

        # Step 5 Mating Selection
        mating_pool = toolbox.selectTournament(archive, k=N)
        offspring_pool = map(toolbox.clone, mating_pool)

        # Step 6 Variation
        # crossover 100% and mutation 6%
        for child1, child2 in zip(offspring_pool[::2], offspring_pool[1::2]):
            toolbox.mate(child1, child2)

        for mutant in offspring_pool:
            if random.random() < 0.06:
                toolbox.mutate(mutant)

        pop = offspring_pool

        print curr_gen
        curr_gen += 1


    import matplotlib.pyplot as plt
    import numpy

    front = numpy.array([ind.fitness.values for ind in archive])
    plt.scatter(front[:,0], front[:,1], c="b")
    plt.axis("tight")
    plt.show()


if __name__ == '__main__':
    main()