import array
import random

from deap import benchmarks, algorithms, base, creator, tools

def my_rand(V=1,U=0):
    return random.random() * (V-U) - (V+U)/2

N=100
GEN=200

toolbox = base.Toolbox()

creator.create("FitnessMax", base.Fitness, weights=(-1.0,-1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)
toolbox.register("attr_float", my_rand)
toolbox.register("individual", tools.initRepeat, creator.Individual,toolbox.attr_float, n=20)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("mate", tools.cxSimulatedBinaryBounded, eta=0.5, low=0, up=1)
toolbox.register("mutate", tools.mutPolynomialBounded, eta=0.5, low=0, up=1, indpb=1)
toolbox.register("select", tools.selNSGA2)
toolbox.register("selectTournament", tools.selTournamentDCD)
toolbox.register("evaluate", benchmarks.zdt2)

# init population
pop = toolbox.population(N)

# init list for partial results and partial spacing
partial_res = []
partial_spacing = []

# Step 1 Initialization
archive = []
curr_gen = 1

# init population
for ind in pop:
    ind.fitness.values = toolbox.evaluate(ind)

# sort using non domination sort (k is the same as n of population - only sort is applied)
pop = toolbox.select(pop, k=N)

for g in xrange(GEN):
    print "CURRENT GEN: " + str(g)
    #select parent pool with tournament dominated selection
    parent_pool = toolbox.selectTournament(pop, k=N)
    offspring_pool = map(toolbox.clone, parent_pool)

    # Apply crossover and mutation on the offspring
    for child1, child2 in zip(offspring_pool[::2], offspring_pool[1::2]):
        if random.random() < 0.9:
            toolbox.mate(child1, child2)
    for mutant in offspring_pool:
        if random.random() < 0.1:
            toolbox.mutate(mutant)

    # evaluate offsprings
    for ind in offspring_pool:
        ind.fitness.values = toolbox.evaluate(ind)

    # extend base population with offsprings, pop is now 2N size
    pop.extend(offspring_pool)

    # sort and select new population
    pop = toolbox.select(pop, k=N)
    
final_front = tools.sortFastND(pop, k=N)[0]

print final_front
