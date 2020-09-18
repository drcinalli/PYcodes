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
from itertools import chain
from operator import attrgetter, itemgetter

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

    def ZDT4_init(self):



        def uniform(low, up, size=None):
            try:
                return [random.uniform(a, b) for a, b in zip(low, up)]
            except TypeError:
                return [random.uniform(a, b) for a, b in zip([low] * size, [up] * size)]

        #BOUND_LOW, BOUND_UP = 0.0 + (-5.0*9), 1.0 + (5.0*9)
        BOUND_LOW, BOUND_UP = [0.0] + [-5.0]*9, [1.0] + [5.0]*9
        creator.create("FitnessMin", base.Fitness, weights=(-1.0,-1.0))
        creator.create("Individual", array.array, typecode='d', fitness=creator.FitnessMin)


        # Functions zdt1, zdt2, zdt3 have 30 dimensions, zdt4 and zdt6 have 10
        NDIM = 30


        toolbox.register("attr_float", uniform, BOUND_LOW, BOUND_UP, 10)
        toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.attr_float)
        toolbox.register("population", tools.initRepeat, list, toolbox.individual)

        toolbox.register("evaluate", benchmarks.zdt4)

        toolbox.register("mate", tools.cxSimulatedBinaryBounded, low=BOUND_LOW, up=BOUND_UP, eta=20.0)
        toolbox.register("mutate", tools.mutPolynomialBounded, low=BOUND_LOW, up=BOUND_UP, eta=20.0, indpb=1.0/NDIM)
        toolbox.register("select", tools.selNSGA2)
        #toolbox.register("select", tools.selSPEA2)

    def ZDT6_init(self):


        NDIM = 10

        def uniform(low, up, size=None):
            try:
                return [random.uniform(a, b) for a, b in zip(low, up)]
            except TypeError:
                return [random.uniform(a, b) for a, b in zip([low] * size, [up] * size)]

        BOUND_LOW, BOUND_UP = 0.0, 1.0
        creator.create("FitnessMin", base.Fitness, weights=(-1.0,-1.0))
        creator.create("Individual", array.array, typecode='d', fitness=creator.FitnessMin)


        toolbox.register("attr_float", uniform, BOUND_LOW, BOUND_UP, NDIM)
        toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.attr_float)
        toolbox.register("population", tools.initRepeat, list, toolbox.individual)

        toolbox.register("evaluate", benchmarks.zdt6)

        toolbox.register("mate", tools.cxSimulatedBinaryBounded, low=BOUND_LOW, up=BOUND_UP, eta=20.0)
        toolbox.register("mutate", tools.mutPolynomialBounded, low=BOUND_LOW, up=BOUND_UP, eta=20.0, indpb=1.0/NDIM)
        toolbox.register("select", tools.selNSGA2)
        #toolbox.register("select", tools.selSPEA2)
        #toolbox.register("selectNSGACOIN", nsgacoin.selNSGA2COIN, world=my_world)



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

        #### ESSE QUE FUNCIONA
        # toolbox.register("mate", tools.cxSimulatedBinaryBounded, low=BOUND_LOW, up=BOUND_UP, eta=0.5)
        # toolbox.register("mutate", tools.mutPolynomialBounded, low=BOUND_LOW, up=BOUND_UP, eta=0.5, indpb=1.0)

        toolbox.register("mate", tools.cxSimulatedBinaryBounded, low=BOUND_LOW, up=BOUND_UP, eta=2)
        toolbox.register("mutate", tools.mutPolynomialBounded, low=BOUND_LOW, up=BOUND_UP, eta=2, indpb=1.0/NDIM)

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

    def ZDT2_init_SPEA(self):

        creator.create("FitnessMin", base.Fitness, weights=(-1.0, -1.0))
        creator.create("Individual", array.array, typecode='d', fitness=creator.FitnessMin)

        # Functions zdt1, zdt2, zdt3 have 30 dimensions, zdt4 and zdt6 have 10

        #NDIM = 6
        NDIM = 30

        toolbox.register("attr_float", uniform, BOUND_LOW, BOUND_UP, NDIM)
        toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.attr_float)
        toolbox.register("population", tools.initRepeat, list, toolbox.individual)

        toolbox.register("evaluate", benchmarks.zdt2)
        # toolbox.register("mate", tools.cxSimulatedBinaryBounded, low=BOUND_LOW, up=BOUND_UP, eta=20.0)
        # toolbox.register("mutate", tools.mutPolynomialBounded, low=BOUND_LOW, up=BOUND_UP, eta=20.0, indpb=1.0/NDIM)
        #toolbox.register("select", tools.selNSGA2)

        #### ESSE QUE FUNCIONA
        # toolbox.register("mate", tools.cxSimulatedBinaryBounded, low=BOUND_LOW, up=BOUND_UP, eta=0.5)
        # toolbox.register("mutate", tools.mutPolynomialBounded, low=BOUND_LOW, up=BOUND_UP, eta=0.5, indpb=1.0)

        toolbox.register("mate", tools.cxSimulatedBinaryBounded, low=BOUND_LOW, up=BOUND_UP, eta=2)
        toolbox.register("mutate", tools.mutPolynomialBounded, low=BOUND_LOW, up=BOUND_UP, eta=2, indpb=1.0/NDIM)

        #toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_float, n=6) #@UndefinedVariable


        toolbox.register("select", tools.selSPEA2)
        toolbox.register("selectTournament", tools.selTournament, tournsize=2)

    def ZDT3_init_SPEA(self):

        creator.create("FitnessMin", base.Fitness, weights=(-1.0, -1.0))
        creator.create("Individual", array.array, typecode='d', fitness=creator.FitnessMin)

        # Functions zdt1, zdt2, zdt3 have 30 dimensions, zdt4 and zdt6 have 10

        #NDIM = 6
        NDIM = 30

        toolbox.register("attr_float", uniform, BOUND_LOW, BOUND_UP, NDIM)
        toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.attr_float)
        toolbox.register("population", tools.initRepeat, list, toolbox.individual)

        toolbox.register("evaluate", benchmarks.zdt3)
        # toolbox.register("mate", tools.cxSimulatedBinaryBounded, low=BOUND_LOW, up=BOUND_UP, eta=20.0)
        # toolbox.register("mutate", tools.mutPolynomialBounded, low=BOUND_LOW, up=BOUND_UP, eta=20.0, indpb=1.0/NDIM)
        #toolbox.register("select", tools.selNSGA2)

        #### ESSE QUE FUNCIONA
        # toolbox.register("mate", tools.cxSimulatedBinaryBounded, low=BOUND_LOW, up=BOUND_UP, eta=0.5)
        # toolbox.register("mutate", tools.mutPolynomialBounded, low=BOUND_LOW, up=BOUND_UP, eta=0.5, indpb=1.0)

        toolbox.register("mate", tools.cxSimulatedBinaryBounded, low=BOUND_LOW, up=BOUND_UP, eta=2)
        toolbox.register("mutate", tools.mutPolynomialBounded, low=BOUND_LOW, up=BOUND_UP, eta=2, indpb=1.0/NDIM)

        #toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_float, n=6) #@UndefinedVariable


        toolbox.register("select", tools.selSPEA2)
        toolbox.register("selectTournament", tools.selTournament, tournsize=2)



    def ZDT4_init_SPEA(self):





        def uniform(low, up, size=None):
            try:
                return [random.uniform(a, b) for a, b in zip(low, up)]
            except TypeError:
                return [random.uniform(a, b) for a, b in zip([low] * size, [up] * size)]

        #BOUND_LOW, BOUND_UP = 0.0 + (-5.0*9), 1.0 + (5.0*9)
        BOUND_LOW, BOUND_UP = [0.0] + [-5.0]*9, [1.0] + [5.0]*9
        creator.create("FitnessMin", base.Fitness, weights=(-1.0,-1.0))
        creator.create("Individual", array.array, typecode='d', fitness=creator.FitnessMin)


        # Functions zdt1, zdt2, zdt3 have 30 dimensions, zdt4 and zdt6 have 10

        #NDIM = 6
        NDIM = 10

        toolbox.register("attr_float", uniform, BOUND_LOW, BOUND_UP, 10)
        toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.attr_float)
        toolbox.register("population", tools.initRepeat, list, toolbox.individual)

        toolbox.register("evaluate", benchmarks.zdt4)
        # toolbox.register("mate", tools.cxSimulatedBinaryBounded, low=BOUND_LOW, up=BOUND_UP, eta=20.0)
        # toolbox.register("mutate", tools.mutPolynomialBounded, low=BOUND_LOW, up=BOUND_UP, eta=20.0, indpb=1.0/NDIM)
        #toolbox.register("select", tools.selNSGA2)

        #### ESSE QUE FUNCIONA
        # toolbox.register("mate", tools.cxSimulatedBinaryBounded, low=BOUND_LOW, up=BOUND_UP, eta=0.5)
        # toolbox.register("mutate", tools.mutPolynomialBounded, low=BOUND_LOW, up=BOUND_UP, eta=0.5, indpb=1.0)
        toolbox.register("mate", tools.cxSimulatedBinaryBounded, low=BOUND_LOW, up=BOUND_UP, eta=2)
        toolbox.register("mutate", tools.mutPolynomialBounded, low=BOUND_LOW, up=BOUND_UP, eta=2, indpb=1.0/NDIM)
        #toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_float, n=6) #@UndefinedVariable
        toolbox.register("select", tools.selSPEA2)
        toolbox.register("selectTournament", tools.selTournament, tournsize=2)

    def ZDT6_init_SPEA(self):



        NDIM = 10

        def uniform(low, up, size=None):
            try:
                return [random.uniform(a, b) for a, b in zip(low, up)]
            except TypeError:
                return [random.uniform(a, b) for a, b in zip([low] * size, [up] * size)]

        BOUND_LOW, BOUND_UP = 0.0, 1.0
        creator.create("FitnessMin", base.Fitness, weights=(-1.0,-1.0))
        creator.create("Individual", array.array, typecode='d', fitness=creator.FitnessMin)


        toolbox.register("attr_float", uniform, BOUND_LOW, BOUND_UP, NDIM)
        toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.attr_float)
        toolbox.register("population", tools.initRepeat, list, toolbox.individual)

        toolbox.register("evaluate", benchmarks.zdt6)



        #### ESSE QUE FUNCIONA
        # toolbox.register("mate", tools.cxSimulatedBinaryBounded, low=BOUND_LOW, up=BOUND_UP, eta=0.5)
        # toolbox.register("mutate", tools.mutPolynomialBounded, low=BOUND_LOW, up=BOUND_UP, eta=0.5, indpb=1.0)
        toolbox.register("mate", tools.cxSimulatedBinaryBounded, low=BOUND_LOW, up=BOUND_UP, eta=2)
        toolbox.register("mutate", tools.mutPolynomialBounded, low=BOUND_LOW, up=BOUND_UP, eta=2, indpb=1.0/NDIM)
        #toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_float, n=6) #@UndefinedVariable
        toolbox.register("select", tools.selSPEA2)
        toolbox.register("selectTournament", tools.selTournament, tournsize=2)

    def ZDT1_init_SMS(self):


        NDIM = 30

        def uniform(low, up, size=None):
            try:
                return [random.uniform(a, b) for a, b in zip(low, up)]
            except TypeError:
                return [random.uniform(a, b) for a, b in zip([low] * size, [up] * size)]

        BOUND_LOW, BOUND_UP = 0.0, 1.0
        creator.create("FitnessMin", base.Fitness, weights=(-1.0,-1.0))
        creator.create("Individual", array.array, typecode='d', fitness=creator.FitnessMin)


        toolbox.register("attr_float", uniform, BOUND_LOW, BOUND_UP, NDIM)
        toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.attr_float)
        toolbox.register("population", tools.initRepeat, list, toolbox.individual)

        toolbox.register("evaluate", benchmarks.zdt1)

        toolbox.register("mate", tools.cxSimulatedBinaryBounded, low=BOUND_LOW, up=BOUND_UP, eta=20.0)
        toolbox.register("mutate", tools.mutPolynomialBounded, low=BOUND_LOW, up=BOUND_UP, eta=20.0, indpb=1.0/NDIM)
        #toolbox.register("select", tools.selNSGA2)
        #toolbox.register("select", tools.selSPEA2)
        #toolbox.register("selectNSGACOIN", nsgacoin.selNSGA2COIN, world=my_world)
        toolbox.register("select", self.selSMS)


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
    def NSGA_Selection(self, pop):

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
    def SMS_Selection(self, pop):

        #$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
        # Vary the population
        offspring = self.selTournamentHYPER(pop, len(pop)) #TROCAR por Hyper+COIN DIst



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

    #I copied from DEAP implementation
    def selTournamentCOINd(self, individuals, k):
        """Tournament selection based on COIN distance between two individuals, if
        the two individuals do not interdominate the selection is made
        based on COIN distance.

        The *individuals* sequence length has to
        be a multiple of 4. Starting from the beginning of the selected
        individuals, two consecutive individuals will be different (assuming all
        individuals in the input list are unique). Each individual from the input
        list won't be selected more than twice.

        This selection requires the individuals to have a :attr:`crowding_dist`
        attribute, which can be set by the :func:`assignCrowdingDist` function.

        :param individuals: A list of individuals to select from.
        :param k: The number of individuals to select.
        :returns: A list of selected individuals.
        """
        def tourn(ind1, ind2):
            if ind1.fitness.dominates(ind2.fitness):
                return ind1
            elif ind2.fitness.dominates(ind1.fitness):
                return ind2

            #COIN DIstance = is the opposite of Crowding Dist
            if ind1.fitness.crowding_dist < ind2.fitness.crowding_dist:
                return ind1
            elif ind1.fitness.crowding_dist > ind2.fitness.crowding_dist:
                return ind2

            if random.random() <= 0.5:
                return ind1
            return ind2

        individuals_1 = random.sample(individuals, len(individuals))
        individuals_2 = random.sample(individuals, len(individuals))

        chosen = []


        for i in xrange(0, k, 4):
            chosen.append(tourn(individuals_1[i],   individuals_1[i+1]))
            chosen.append(tourn(individuals_1[i+2], individuals_1[i+3]))
            chosen.append(tourn(individuals_2[i],   individuals_2[i+1]))
            chosen.append(tourn(individuals_2[i+2], individuals_2[i+3]))


        return chosen

  #cross and mutation


    def selTournamentHYPER(self, individuals, k):
        """Tournament selection based on dominance (D) between two individuals, if
        the two individuals do not interdominate the selection is made
        based on crowding distance (CD). The *individuals* sequence length has to
        be a multiple of 4. Starting from the beginning of the selected
        individuals, two consecutive individuals will be different (assuming all
        individuals in the input list are unique). Each individual from the input
        list won't be selected more than twice.

        This selection requires the individuals to have a :attr:`crowding_dist`
        attribute, which can be set by the :func:`assignCrowdingDist` function.

        :param individuals: A list of individuals to select from.
        :param k: The number of individuals to select.
        :returns: A list of selected individuals.
        """
        def tourn(ind1, ind2):
            if ind1.fitness.dominates(ind2.fitness):
                return ind1
            elif ind2.fitness.dominates(ind1.fitness):
                return ind2

            if ind1.fitness.crowding_dist < ind2.fitness.crowding_dist:
                return ind1
            elif ind1.fitness.crowding_dist > ind2.fitness.crowding_dist:
                return ind2

            if random.random() <= 0.5:
                return ind1
            return ind2

        individuals_1 = random.sample(individuals, len(individuals))
        individuals_2 = random.sample(individuals, len(individuals))

        chosen = []
        for i in xrange(0, k, 4):
            chosen.append(tourn(individuals_1[i],   individuals_1[i+1]))
            chosen.append(tourn(individuals_1[i+2], individuals_1[i+3]))
            chosen.append(tourn(individuals_2[i],   individuals_2[i+1]))
            chosen.append(tourn(individuals_2[i+2], individuals_2[i+3]))

        return chosen


    def Selection_SPEA(self, pop, archive):

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
    def SetPopulationFakeBench(self, novo=30):

        def uniform(size=None):
                return [1]*size

        toolbox.register("attr_float_fake", uniform, novo)


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

    def assignHyperContribution(self, front):
    # Must use wvalues * -1 since hypervolume use implicit minimization
    # And minimization in deap use max on -obj
        wobj = np.array([ind.fitness.wvalues for ind in front]) * -1
        ref = np.max(wobj, axis=0) + 1
        #print ref

        def contribution(i):
            # The contribution of point p_i in point set P
            # is the hypervolume of P without p_i
            #a = self.Hypervolume2D(front , ref)
            b = self.Hypervolume2D(front[:i]+front[i+1:] , ref)

            return  b

        # Parallelization note: Cannot pickle local function
        contrib_values = map(contribution, range(len(front)))

        # Select the maximum hypervolume value (correspond to the minimum difference)
        #return np.argmax(contrib_values)
        for i, h in enumerate(contrib_values):
            front[i].fitness.crowding_dist = h


    def selSMS(self, individuals, k, nd='standard'):
        """Apply NSGA-II
        :param individuals: A list of individuals to select from.
        :param k: The number of individuals to select.
        :param nd: Specify the non-dominated algorithm to use: 'standard' or 'log'.
        :returns: A list of selected individuals.

        .. [Deb2002] Deb, Pratab, Agarwal, and Meyarivan, "A fast elitist
           non-dominated sorting genetic algorithm for multi-objective
           optimization: NSGA-II", 2002.
        """
        nd='standard'
        if nd == 'standard':
            #print 'ssjsjsjsjsjs'
            pareto_fronts = self.sortNondominated(individuals, k)
        # elif nd == 'log':
        #     pareto_fronts = sortLogNondominated(individuals, k)
        else:
            raise Exception('selNSGA2: The choice of non-dominated sorting '
                            'method "{0}" is invalid.'.format(nd))

        #som=0
        for front in pareto_fronts:
            self.assignHyperContribution(front) ### baseado no HYPERVOLUME e nao na distancia
            #som += len(front)
        #print som

        chosen = list(chain(*pareto_fronts[:-1]))
        k = k - len(chosen)

        #here I have the last front in my hands... READY

        if k > 0:

            #HERE THE REDUCTION based on Hyper+COIN dist
            #sorted_front = sorted(pareto_fronts[-1], key=attrgetter("fitness.crowding_dist"), reverse=True)
            sorted_front = sorted(pareto_fronts[-1], key=attrgetter("fitness.crowding_dist"))
            chosen.extend(sorted_front[:k])

        #return chosen
        return chosen #, sorted_front[:k],sorted_front[k:]

    def sortNondominated(self, individuals, k, first_front_only=False):
        """Sort the first *k* *individuals* into different nondomination levels
        using the "Fast Nondominated Sorting Approach" proposed by Deb et al.,
        see [Deb2002]_. This algorithm has a time complexity of :math:`O(MN^2)`,
        where :math:`M` is the number of objectives and :math:`N` the number of
        individuals.

        :param individuals: A list of individuals to select from.
        :param k: The number of individuals to select.
        :param first_front_only: If :obj:`True` sort only the first front and
                                 exit.
        :returns: A list of Pareto fronts (lists), the first list includes
                  nondominated individuals.

        .. [Deb2002] Deb, Pratab, Agarwal, and Meyarivan, "A fast elitist
           non-dominated sorting genetic algorithm for multi-objective
           optimization: NSGA-II", 2002.
        """
        from collections import defaultdict


        if k == 0:
            return []

        map_fit_ind = defaultdict(list)
        for ind in individuals:
            map_fit_ind[ind.fitness].append(ind)
        fits = map_fit_ind.keys()

        current_front = []
        next_front = []
        dominating_fits = defaultdict(int)
        dominated_fits = defaultdict(list)

        # Rank first Pareto front
        for i, fit_i in enumerate(fits):
            for fit_j in fits[i+1:]:
                if fit_i.dominates(fit_j):
                    dominating_fits[fit_j] += 1
                    dominated_fits[fit_i].append(fit_j)
                elif fit_j.dominates(fit_i):
                    dominating_fits[fit_i] += 1
                    dominated_fits[fit_j].append(fit_i)
            if dominating_fits[fit_i] == 0:
                current_front.append(fit_i)

        fronts = [[]]
        for fit in current_front:
            fronts[-1].extend(map_fit_ind[fit])
        pareto_sorted = len(fronts[-1])

        # Rank the next front until all individuals are sorted or
        # the given number of individual are sorted.
        if not first_front_only:
            N = min(len(individuals), k)
            while pareto_sorted < N:
                fronts.append([])
                for fit_p in current_front:
                    for fit_d in dominated_fits[fit_p]:
                        dominating_fits[fit_d] -= 1
                        if dominating_fits[fit_d] == 0:
                            next_front.append(fit_d)
                            pareto_sorted += len(map_fit_ind[fit_d])
                            fronts[-1].extend(map_fit_ind[fit_d])
                current_front = next_front
                next_front = []

        return fronts

    def assignCrowdingDist(self, individuals):
        """Assign a crowding distance to each individual's fitness. The
        crowding distance can be retrieve via the :attr:`crowding_dist`
        attribute of each individual's fitness.
        """
        if len(individuals) == 0:
            return

        distances = [0.0] * len(individuals)
        crowd = [(ind.fitness.values, i) for i, ind in enumerate(individuals)]

        nobj = len(individuals[0].fitness.values)

        for i in xrange(nobj):
            crowd.sort(key=lambda element: element[0][i])
            distances[crowd[0][1]] = float("inf")
            distances[crowd[-1][1]] = float("inf")
            if crowd[-1][0][i] == crowd[0][0][i]:
                continue
            norm = nobj * float(crowd[-1][0][i] - crowd[0][0][i])
            for prev, cur, next in zip(crowd[:-2], crowd[1:-1], crowd[2:]):
                distances[cur[1]] += (next[0][i] - prev[0][i]) / norm

        for i, dist in enumerate(distances):
            individuals[i].fitness.crowding_dist = dist

    def selNSGA2PURE(self, individuals, k, nd='standard'):
        """Apply NSGA-II selection operator on the *individuals*. Usually, the
        size of *individuals* will be larger than *k* because any individual
        present in *individuals* will appear in the returned list at most once.
        Having the size of *individuals* equals to *k* will have no effect other
        than sorting the population according to their front rank. The
        list returned contains references to the input *individuals*. For more
        details on the NSGA-II operator see [Deb2002]_.

        :param individuals: A list of individuals to select from.
        :param k: The number of individuals to select.
        :param nd: Specify the non-dominated algorithm to use: 'standard' or 'log'.
        :returns: A list of selected individuals.

        .. [Deb2002] Deb, Pratab, Agarwal, and Meyarivan, "A fast elitist
           non-dominated sorting genetic algorithm for multi-objective
           optimization: NSGA-II", 2002.
        """
        nd='standard'
        if nd == 'standard':
            #print 'ssjsjsjsjsjs'
            pareto_fronts = self.sortNondominated(individuals, k)
        # elif nd == 'log':
        #     pareto_fronts = sortLogNondominated(individuals, k)
        else:
            raise Exception('selNSGA2: The choice of non-dominated sorting '
                            'method "{0}" is invalid.'.format(nd))

        for front in pareto_fronts:
            self.assignCrowdingDist(front)

        chosen = list(chain(*pareto_fronts[:-1]))
        k = k - len(chosen)
        if k > 0:
            sorted_front = sorted(pareto_fronts[-1], key=attrgetter("fitness.crowding_dist"), reverse=True)
            chosen.extend(sorted_front[:k])

        #return chosen
        return chosen, sorted_front[:k],sorted_front[k:]

    #get hypervolume
    def Hypervolume2D(self, front, refpoint):

        #transform front fitness to a list of fitness
        local_fit=[]
        for i in front:
            local_fit.append((i.fitness.values[0],i.fitness.values[1]))


        #evaluate the hypervolume
        hyper=HyperVolume(refpoint)
        aux = hyper.compute(local_fit)
        return aux/(refpoint[0]*refpoint[1])

    #get hypervolume
    def Hypervolume3D(self, front, refpoint):

        #transform front fitness to a list of fitness
        local_fit=[]
        for i in front:
            local_fit.append((i.fitness.values[0],i.fitness.values[1], i.fitness.values[2]))


        #evaluate the hypervolume
        hyper=HyperVolume(refpoint)
        aux = hyper.compute(local_fit)
        return aux/(refpoint[0]*refpoint[1]*refpoint[2])
