'''
Created on 21/08/2014

@author: quatrosem
'''
from __future__ import division
import random, array
from deap import base
from deap import creator
from deap import tools
import collections
import numpy as np
from hv import HyperVolume
from deap import benchmarks


toolbox = base.Toolbox()

# Problem definition
# Functions zdt4 has bounds x1 = [0, 1], xn = [-5, 5], with n = 2, ..., 10
# BOUND_LOW, BOUND_UP = [0.0] + [-5.0]*9, [1.0] + [5.0]*9
# Functions zdt1, zdt2, zdt3, zdt6 have bounds [0, 1]
BOUND_LOW, BOUND_UP = 0.0, 1.0

#Fitness of OBJECTIVE #1
def evalOneMax(individual):
    return sum(individual),

def uniform(low, up, size=None):
    try:
        return [random.uniform(a, b) for a, b in zip(low, up)]
    except TypeError:
        return [random.uniform(a, b) for a, b in zip([low] * size, [up] * size)]

class GA:
    #class variables
    description = "This is the Genetic Algorithm class of the problem"
    author = "Daniel Cinalli"

    CXPB  = 0.1
    MUTPB = 0.05
    NGEN  = 70
    num_population = 300

    def __init__(self, CXPB=None, MUTPB=None, NGEN = None,  num_population=None):
        self.CXPB  = CXPB
        self.MUTPB = MUTPB
        self.NGEN  = NGEN
        self.num_population = num_population



    def OneMax_init(self):

        creator.create("FitnessMax", base.Fitness, weights=(1.0,))
        creator.create("Individual", array.array, typecode='b', fitness=creator.FitnessMax)


        toolbox.register("attr_bool", random.randint, 0, 1)
        toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_bool, 100)
        toolbox.register("population", tools.initRepeat, list, toolbox.individual)

        toolbox.register("evaluate", evalOneMax)
        toolbox.register("mate", tools.cxTwoPoint)
        toolbox.register("mutate", tools.mutFlipBit, indpb=0.05)
        toolbox.register("select", tools.selTournament, tournsize=3)

    def ZDT1_init(self):

        creator.create("FitnessMin", base.Fitness, weights=(-1.0, -1.0))
        creator.create("Individual", array.array, typecode='d', fitness=creator.FitnessMin)

        # Functions zdt1, zdt2, zdt3 have 30 dimensions, zdt4 and zdt6 have 10
        NDIM = 30

        toolbox.register("attr_float", uniform, BOUND_LOW, BOUND_UP, NDIM)
        toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.attr_float)
        toolbox.register("population", tools.initRepeat, list, toolbox.individual)

        toolbox.register("evaluate", benchmarks.zdt1)
        toolbox.register("mate", tools.cxSimulatedBinaryBounded, low=BOUND_LOW, up=BOUND_UP, eta=20.0)
        toolbox.register("mutate", tools.mutPolynomialBounded, low=BOUND_LOW, up=BOUND_UP, eta=20.0, indpb=1.0/NDIM)
        toolbox.register("select", tools.selNSGA2)
        #toolbox.register("select", tools.selSPEA2)



    def ZDT1_init_SPEA(self):

        creator.create("FitnessMin", base.Fitness, weights=(-1.0, -1.0))
        creator.create("Individual", array.array, typecode='d', fitness=creator.FitnessMin)

        # Functions zdt1, zdt2, zdt3 have 30 dimensions, zdt4 and zdt6 have 10

        #NDIM = 6
        NDIM = 30

        toolbox.register("attr_float", uniform, BOUND_LOW, BOUND_UP, NDIM)
        toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.attr_float)
        toolbox.register("population", tools.initRepeat, list, toolbox.individual)

        toolbox.register("evaluate", benchmarks.zdt1)
        # toolbox.register("mate", tools.cxSimulatedBinaryBounded, low=BOUND_LOW, up=BOUND_UP, eta=20.0)
        # toolbox.register("mutate", tools.mutPolynomialBounded, low=BOUND_LOW, up=BOUND_UP, eta=20.0, indpb=1.0/NDIM)
        #toolbox.register("select", tools.selNSGA2)

        toolbox.register("mate", tools.cxSimulatedBinaryBounded, low=BOUND_LOW, up=BOUND_UP, eta=0.5)
        toolbox.register("mutate", tools.mutPolynomialBounded, low=BOUND_LOW, up=BOUND_UP, eta=0.5, indpb=1.0)

        #toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_float, n=6) #@UndefinedVariable


        toolbox.register("select", tools.selSPEA2)
        toolbox.register("selectTournament", tools.selTournament, tournsize=2)

    def ZDT1_init_SPEA3D(self):

        creator.create("FitnessMin", base.Fitness, weights=(-1.0, -1.0, -1.0))
        creator.create("Individual", array.array, typecode='d', fitness=creator.FitnessMin)

        # Functions zdt1, zdt2, zdt3 have 30 dimensions, zdt4 and zdt6 have 10

        NDIM = 3

        toolbox.register("attr_float", uniform, BOUND_LOW, BOUND_UP, NDIM)
        toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.attr_float)
        toolbox.register("population", tools.initRepeat, list, toolbox.individual)

        toolbox.register("evaluate", lambda ind: benchmarks.dtlz1(ind, 3))
        toolbox.register("mate", tools.cxSimulatedBinaryBounded, low=BOUND_LOW, up=BOUND_UP, eta=20.0)
        toolbox.register("mutate", tools.mutPolynomialBounded, low=BOUND_LOW, up=BOUND_UP, eta=20.0, indpb=1.0/NDIM)
        #toolbox.register("select", tools.selNSGA2)

        #toolbox.register("mate", tools.cxSimulatedBinaryBounded, low=BOUND_LOW, up=BOUND_UP, eta=0.5)
        #toolbox.register("mutate", tools.mutPolynomialBounded, low=BOUND_LOW, up=BOUND_UP, eta=0.5, indpb=1.0)

        #toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_float, n=6) #@UndefinedVariable


        toolbox.register("select", tools.selSPEA2)
        toolbox.register("selectTournament", tools.selTournament, tournsize=2)


    #create population
    def SetPopulation(self, num):
        pop =toolbox.population(n=num)
        #pop.fitness.values = 3
        return pop

    #get fitness of all individuals
    def GetFitness(self, pop):
        fitnesses = list(map(toolbox.evaluate, pop))
        #fitnesses = list(map(toolbox.evaluate, [[8, 8, 1, 19, 15, 1, 16, 11, 1, 14, 2, 1, 4, 15, 1, 9, 14, 1, 0, 8, 1, -1, -1, -1, -1, -1, -1, -1, -1, -1], [8, 8, 1, 19, 15, 1, 16, 11, 1, 14, 2, 1, 4, 15, 1, 9, 14, 1, 0, 8, 1, -1, -1, -1, -1, -1, -1, -1, -1, -1], [8, 8, 1, 19, 15, 1, 16, 11, 1, 14, 2, 1, 4, 15, 1, 9, 14, 1, 0, 8, 1, -1, -1, -1, -1, -1, -1, -1, -1, -1]]))
        return fitnesses

    #attach fitness to the individual
    def AttachFitness(self, pop, fitnesses):

        for ind, fit in zip(pop, fitnesses):
            ind.fitness.values = fit

    #cross and mutation
    def Selection(self, pop,rate=None):

        # Select the next generation individuals
        offspring = toolbox.select(pop, len(pop))
        # Clone the selected individuals
        offspring = list(map(toolbox.clone, offspring))

        # Apply crossover and mutation on the offspring
        for child1, child2 in zip(offspring[::2], offspring[1::2]): #get every single pair
            if random.random() < self.CXPB:
                toolbox.mate(child1, child2)
                del child1.fitness.values
                del child2.fitness.values
        for mutant in offspring:
            if random.random() < self.MUTPB:
                #print mutant
                toolbox.mutate(mutant)
                #print mutant
                del mutant.fitness.values

        # Evaluate the individuals with an invalid fitness
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]

        # print("  Cross or Mutation: %s" % len(invalid_ind))
        fitnesses = map(toolbox.evaluate, invalid_ind)

        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit


        #print("  Evaluated %i individuals" % len(invalid_ind))

        # The population is entirely replaced by the offspring
        pop[:] = offspring


        # Gather all the fitnesses in one list and print the stats
        fits = [ind.fitness.values[0] for ind in pop]

        length = len(pop)
        mean = sum(fits) / length
        sum2 = sum(x*x for x in fits)
        std = abs(sum2 / length - mean**2)**0.5

        print("  Min %s" % min(fits))
        print("  Max %s" % max(fits))
        print("  Avg %s" % mean)
        print("  Std %s" % std)

    #cross and mutation
    def ZDT1_Selection(self, pop):

        #$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
        # Vary the population
        offspring = tools.selTournamentDCD(pop, len(pop))
        offspring = [toolbox.clone(ind) for ind in offspring]

        # # Select the next generation individuals
        # offspring = toolbox.select(pop, len(pop))
        # # Clone the selected individuals
        # offspring = list(map(toolbox.clone, offspring))

        # Apply crossover and mutation on the offspring
        for child1, child2 in zip(offspring[::2], offspring[1::2]): #get every single pair
            if random.random() < self.CXPB:
                toolbox.mate(child1, child2)
                del child1.fitness.values
                del child2.fitness.values
        for mutant in offspring:
            if random.random() < self.MUTPB:
                #print mutant
                toolbox.mutate(mutant)
                #print mutant
                del mutant.fitness.values



        # Evaluate the individuals with an invalid fitness
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]

        # print("  Cross or Mutation: %s" % len(invalid_ind))
        fitnesses = map(toolbox.evaluate, invalid_ind)

        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit


        # Select the next generation population
        off = toolbox.select(pop + offspring, self.num_population)
        pop[:] = off


  #cross and mutation
    def ZDT1_Selection_SPEA(self, pop, archive):

        # Step 1 Environmental selection -- define external archive
        archive_b  = toolbox.select(pop + archive, k=(len(pop)//2))
        fitnesses = self.GetFitness(archive_b)
        self.AttachFitness(archive_b,fitnesses)


        #$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
        # Vary the population
        #step 2
        offspring = toolbox.selectTournament(archive_b, len(pop))

        #offspring = toolbox.selectTournament(pop, len(pop))
        offspring = [toolbox.clone(ind) for ind in offspring]

        # # Select the next generation individuals
        # offspring = toolbox.select(pop, len(pop))
        # # Clone the selected individuals
        # offspring = list(map(toolbox.clone, offspring))

        # Apply crossover and mutation on the offspring
        for child1, child2 in zip(offspring[::2], offspring[1::2]): #get every single pair
            if random.random() < self.CXPB:
                toolbox.mate(child1, child2)
                del child1.fitness.values
                del child2.fitness.values
        for mutant in offspring:
            if random.random() < self.MUTPB:
                #print mutant
                toolbox.mutate(mutant)
                #print mutant
                del mutant.fitness.values



        # Evaluate the individuals with an invalid fitness
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]

        # print("  Cross or Mutation: %s" % len(invalid_ind))
        fitnesses = map(toolbox.evaluate, invalid_ind)

        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit


        # Select the next generation population
        # off = toolbox.select(pop + offspring, self.num_population)
        # pop[:] = off
        #

        #step 4
        pop[:] = offspring
        archive[:] = archive_b

    #create population
    def SetPopulationFakeBench(self):

        def uniform(size=None):
                return [1]*size

        toolbox.register("attr_float_fake", uniform, 30)


        # Structure initializers
        toolbox.register("individualfake", tools.initIterate, creator.Individual, toolbox.attr_float_fake)
        toolbox.register("populationfake", tools.initRepeat, list, toolbox.individualfake)


        pop = toolbox.populationfake(n=self.num_population)
        #pop.fitness.values = 3

        return pop


    #cross and mutation
    def ACD(self, pop):

        #$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
        # Select the next generation population
        off = toolbox.select(pop, self.num_population)

        # The population is entirely replaced by the offspring
        pop[:] = off

