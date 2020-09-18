import array
import random

from deap import benchmarks, algorithms, base, creator, tools

def my_rand(V=1,U=0):
    return random.random() * (V-U) - (V+U)/2

N=80 # population size
GEN=100 # number of generations
Nbar = 40 # archive size

toolbox = base.Toolbox()

creator.create("FitnessMax", base.Fitness, weights=(-1.0, -1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)
toolbox.register("attr_float", my_rand)
toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_float, n=20)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("mate", tools.cxSimulatedBinaryBounded, eta=0.5, low=0, up=1)
toolbox.register("mutate", tools.mutPolynomialBounded, eta=0.5, low=0, up=1, indpb=1)
toolbox.register("select", tools.selSPEA2)
toolbox.register("selectTournament", tools.selTournament, tournsize=2)
toolbox.register("evaluate", benchmarks.zdt2)

# init population
pop = toolbox.population(N)

# init list for partial results and partial spacing
partial_res = []
partial_spacing = []

# Step 1 Initialization
archive = []
curr_gen = 1

while True:
    print "CURRENT GEN: " + str(curr_gen)
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
    curr_gen += 1

final_front = final_set
front_ = [(ind.fitness.values[0], ind.fitness.values[1]) for ind in final_front]
print front_
