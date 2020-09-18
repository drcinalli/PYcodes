'''
Created on 21/08/2014

@author: quatrosem
'''
from __future__ import division
import random, array
import math
from deap import base
from deap import creator
from deap import tools
from deap import benchmarks
import collections
import numpy as np
from hv import HyperVolume
import scipy.stats
from nsgaCOIN import NSGACOIN
from speaCOIN import SPEACOIN
from smsCOIN import SMSCOIN


from itertools import chain
from operator import attrgetter, itemgetter



toolbox = base.Toolbox()
nsgacoin = NSGACOIN()
speacoin = SPEACOIN()
smscoin = SMSCOIN()


# Functions zdt1, zdt2, zdt3 have 30 dimensions, zdt4 and zdt6 have 10
NDIM = 30
NDIM_DTLZ = 3
NADIR_SPEA = 10


#Fitness of OBJECTIVE #1
def f1Cost(individual, world):

    ################
    # OBJECTIVE F1 #
    ################
    gates=world.GetGates(individual[0])
    units=world.GetUnits(individual[0])

    #check after cross and/or mutation
    if world.CheckGateways(gates) == 0 or world.CheckUnits(units) == 0 or world.CheckLinks(gates, units) == 0:
        if world.FIX_on_off:
            print 'ERROR: Fix did not work on Selection method'
            print(gates)
            print(units)
            quit()
        return 300,-10

    ##############################
    # ATENCAO, type of units here#
    ##############################
    cost_units= 0
    for i in units:
        if i[2]==0:
            cost_units = cost_units + world.BRL_unit_0
        else:
            cost_units = cost_units + world.BRL_unit_1

    cost_gates= world.BRL_allgates#world.all_areas * world.BRL_gateway
    cost_links= world.CostLinks(gates,units)


    ################
    # OBJECTIVE F2 #
    ################

    #################
    # Prod + 1/dist #
    #################
    # #get all the units linked in the gateway list
    # total_prod=0
    # for i in gates:
        ##############################
        # ATENCAO, type of units here#
        ##############################
    #     #get the link i[2] and calculate the cost.
    #     if units[i[2]-1][2] == 1:
    #         total_prod = total_prod + (world.BRL_unit_1+ 1/np.sqrt( (units[i[2]-1][0] - i[0])**2 + (units[i[2]-1][1] - i[1])**2 ))
    #     if units[i[2]-1][2] == 0:
    #         total_prod = total_prod + (world.BRL_unit_0+ 1/np.sqrt( (units[i[2]-1][0] - i[0])**2 + (units[i[2]-1][1] - i[1])**2 ))
    # #return cost_gates+cost_units+cost_links,total_prod

    ############################################
    # count the number of units by their types #
    ############################################
    unit_ty=[]
    for i in units:
        unit_ty.append(i[2])

    counter=collections.Counter(unit_ty)
    ##############################
    # ATENCAO, type of units here#
    ##############################
    prod=0
    if counter.get(0):
        prod = (counter.get(0)*world.PROD_unit_0)
        #prod = (counter.get(0)*world.PROD_unit_0)/gates_01

    if counter.get(1):

        prod = prod+ (counter.get(1)*world.PROD_unit_1)
        #prod = prod+ (counter.get(1)*world.PROD_unit_1)/gates_02

    return cost_gates+cost_units+cost_links,(prod*-1)
    #return cost_links,(prod*-1)

    #####################################################################
    # count the number of units by their types / mean of their distance #
    #####################################################################
    # sum_dist_per_gate=[]
    # gates_lk=[]
    # for i in units:
    #     sum_dist_per_gate.append(0)
    #
    # for i in gates:
    #     sum_dist_per_gate[i[2]-1] = sum_dist_per_gate[i[2]-1] + np.sqrt( (units[i[2]-1][0] - i[0])**2 + (units[i[2]-1][1] - i[1])**2 )
    #     gates_lk.append(i[2])
    #
    # #get the number of units by links
    # counter=collections.Counter(gates_lk)
    # for i in counter:
    #     sum_dist_per_gate[i-1] = sum_dist_per_gate[i-1]/counter.get(i)
    # #this is the mean for every unit assigned: sum_dist_per_gate
    #
    # #loop units
    # prod=0
    # for i in range(0,len(units)):
        ##############################
        # ATENCAO, type of units here#
        ##############################
    #     if units[i][2]==0:
    #         prod = prod + (world.PROD_unit_0*1/sum_dist_per_gate[i])
    #     if units[i][2]==1:
    #         prod = prod + (world.PROD_unit_1*1/sum_dist_per_gate[i])
    #
    # return cost_gates+cost_units+cost_links,(prod*-1)

    ########
    # MONO #
    ########
    #return cost_gates+cost_units+cost_links,1


def f1CostPAIR(individual, world):

    ################
    # OBJECTIVE F1 #
    ################
    gates=world.GetGates(individual[0])
    units=world.GetUnits(individual[0])

    #check after cross and/or mutation
    if world.CheckGateways(gates) == 0 or world.CheckUnits(units) == 0 or world.CheckLinks(gates, units) == 0:
        if world.FIX_on_off:
            print 'ERROR: Fix did not work on Selection method'
            print(gates)
            print(units)
            quit()
        return 300,-10

    ##############################
    # ATENCAO, type of units here#
    ##############################
    cost_units= 0
    for i in units:
        if i[2]==0:
            cost_units = cost_units + world.BRL_unit_0
        else:
            cost_units = cost_units + world.BRL_unit_1

    cost_gates= world.BRL_allgates#world.all_areas * world.BRL_gateway
    cost_links= world.CostLinks(gates,units)


    ################
    # OBJECTIVE F2 #
    ################

    ############################################
    # count the number of units by their types #
    ############################################
    unit_ty=[]
    for i in units:
        unit_ty.append(i[2])

    counter=collections.Counter(unit_ty)
    ##############################
    # ATENCAO, type of units here#
    ##############################
    prod=0
    if counter.get(0):
        prod = (counter.get(0)*world.PROD_unit_0)
        #prod = (counter.get(0)*world.PROD_unit_0)/gates_01

    if counter.get(1):

        prod = prod+ (counter.get(1)*world.PROD_unit_1)
        #prod = prod+ (counter.get(1)*world.PROD_unit_1)/gates_02

    totalcost= cost_gates+cost_units+cost_links

    ###########################
    # calculate the INCREMENT #
    ###########################


    p_PROcost=scipy.stats.norm(world.mean1, world.sigma1).pdf(totalcost)
    p_ANTIcost=scipy.stats.norm(world.mean2, world.sigma2).pdf(totalcost)

    zscore=0.0
    if p_PROcost>= p_ANTIcost:
        #pro
        #if from the first group (pro-Cost),

        #calculate ZSCORE
        zscore = (totalcost - world.mean1) / world.sigma1


    else:
        #anti
        #if from the second group (pro-Prod),

        #calculate ZSCORE
        zscore = (totalcost - world.mean2) / world.sigma2




    #print abs(zscore)
    #add this zscore to the cost of all that belongs to Mean1 = procost
    #add this zscore to the cost of all that belongs to Mean2 = proPROD
    totalcost += abs(zscore*10)


    #print world.mean1
    # totalcost += abs(totalcost-world.mean1)
    # prod += abs(prod+world.mean2)

    return totalcost,(prod*-1)


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
class GA:
    #class variables
    description = "This is the Genetic Algorithm class of the problem"
    author = "Daniel Cinalli"


    def __init__(self, my_world, CXPB=None, MUTPB=None, NGEN = None,  num_population=None, problem='A', moea_alg='N'):

        if problem == 'A':

            #define fitness of the individual
            creator.create("FitnessMin", base.Fitness, weights=(-1.0,-1.0))
            creator.create("Individual", list, fitness=creator.FitnessMin)

            # Attribute generator
            toolbox.register("individual_gen", my_world.CreateFullindividual)

            # Structure initializers
            toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.individual_gen, 1)
            toolbox.register("population", tools.initRepeat, list, toolbox.individual)

            # Operator registering
            toolbox.register("evaluate", f1Cost, world=my_world)
            toolbox.register("evaluatePAIR", f1CostPAIR, world=my_world)

            #toolbox.register("mate", tools.cxTwoPoint) #changes between 2 points
            #toolbox.register("mate", tools.cxOnePoint) #changes before and after a randon point
            toolbox.register("mate", tools.cxUniform, indpb=CXPB) #changes vector values uniformly by indpb
            #cxPartialyMatched = index vector, it does not fit to my structure
            #cxUniformPartialyMatched = index vector, it does not fit to my stucture
            #cxOrdered = index vector, it does not fit to my stucture
            #cxBlend = expects floating points
            #cxSimulatedBinary =  expects floating points
            #cxSimulatedBinaryBounded =  expects floating points
            #cxMessyOnePoint = change the individuals size.

            toolbox.register("mutate", tools.mutUniformInt,low=0, up=min(my_world.n-1, my_world.m-1), indpb=MUTPB) #gives an integer between 0 or M
            #toolbox.register("mutate", tools.mutShuffleIndexes, indpb=MUTPB) #swap index of individuals
            #mutGaussian= not applicable to my dna
            #mutPolynomialBounded ?
            #mutShuffleIndexes = vector of indexes
            #mutPolynomialBounded = floating points, not

            #toolbox.register("select", tools.selTournament, tournsize=3)
            #toolbox.register("select", tools.selRoulette)
            toolbox.register("select", tools.selNSGA2)
            #toolbox.register("select", tools.selSPEA2)
            toolbox.register("selectNSGACOIN", nsgacoin.selNSGA2COIN, world=my_world)


            toolbox.register("selectTournament", tools.selTournament, tournsize=2)

        elif problem == 'P':

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

            #toolbox.register("mate", tools.cxSimulatedBinaryBounded, low=BOUND_LOW, up=BOUND_UP, eta=20.0)
            #toolbox.register("mutate", tools.mutPolynomialBounded, low=BOUND_LOW, up=BOUND_UP, eta=20.0, indpb=1.0/NDIM)

            toolbox.register("evaluate", benchmarks.zdt1)

            if moea_alg=='P':
                toolbox.register("select", tools.selSPEA2)
                #toolbox.register("select", speacoin.selSPEA2_PURE)
                toolbox.register("selectTournament", tools.selTournament, tournsize=2)
                # toolbox.register("mate", tools.cxSimulatedBinaryBounded, low=BOUND_LOW, up=BOUND_UP, eta=0.5)
                # toolbox.register("mutate", tools.mutPolynomialBounded, low=BOUND_LOW, up=BOUND_UP, eta=0.5, indpb=1.0)
                toolbox.register("mate", tools.cxSimulatedBinaryBounded, low=BOUND_LOW, up=BOUND_UP, eta=4.0)
                toolbox.register("mutate", tools.mutPolynomialBounded, low=BOUND_LOW, up=BOUND_UP, eta=4.0, indpb=1.0/NDIM)
                toolbox.register("selectSPEACOIN", speacoin.selSPEA2COIN, world=my_world)

            elif moea_alg=='S':
                toolbox.register("mate", tools.cxSimulatedBinaryBounded, low=BOUND_LOW, up=BOUND_UP, eta=20.0)
                toolbox.register("mutate", tools.mutPolynomialBounded, low=BOUND_LOW, up=BOUND_UP, eta=20.0, indpb=1.0/NDIM)
                toolbox.register("select", smscoin.selSMS)
                toolbox.register("selectSMSCOIN", smscoin.selSMSCOIN, world=my_world)

            else:
                toolbox.register("mate", tools.cxSimulatedBinaryBounded, low=BOUND_LOW, up=BOUND_UP, eta=20.0)
                toolbox.register("mutate", tools.mutPolynomialBounded, low=BOUND_LOW, up=BOUND_UP, eta=20.0, indpb=1.0/NDIM)
                toolbox.register("select", tools.selNSGA2)
                toolbox.register("selectNSGACOIN", nsgacoin.selNSGA2COIN, world=my_world)
                toolbox.register("selectNSGAPURE", nsgacoin.selNSGA2PURE)



        elif problem == 'Q':

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

            toolbox.register("evaluate", benchmarks.zdt2)


            if moea_alg=='P':
                toolbox.register("select", tools.selSPEA2)
                toolbox.register("selectTournament", tools.selTournament, tournsize=2)
                # toolbox.register("mate", tools.cxSimulatedBinaryBounded, low=BOUND_LOW, up=BOUND_UP, eta=0.5)
                # toolbox.register("mutate", tools.mutPolynomialBounded, low=BOUND_LOW, up=BOUND_UP, eta=0.5, indpb=1.0)
                toolbox.register("mate", tools.cxSimulatedBinaryBounded, low=BOUND_LOW, up=BOUND_UP, eta=2)
                toolbox.register("mutate", tools.mutPolynomialBounded, low=BOUND_LOW, up=BOUND_UP, eta=2, indpb=1.0/NDIM)
                toolbox.register("selectSPEACOIN", speacoin.selSPEA2COIN, world=my_world)

            elif moea_alg=='S':
                toolbox.register("mate", tools.cxSimulatedBinaryBounded, low=BOUND_LOW, up=BOUND_UP, eta=20.0)
                toolbox.register("mutate", tools.mutPolynomialBounded, low=BOUND_LOW, up=BOUND_UP, eta=20.0, indpb=1.0/NDIM)
                toolbox.register("select", smscoin.selSMS)
                toolbox.register("selectSMSCOIN", smscoin.selSMSCOIN, world=my_world)

            else:
                toolbox.register("mate", tools.cxSimulatedBinaryBounded, low=BOUND_LOW, up=BOUND_UP, eta=20.0)
                toolbox.register("mutate", tools.mutPolynomialBounded, low=BOUND_LOW, up=BOUND_UP, eta=20.0, indpb=1.0/NDIM)
                toolbox.register("select", tools.selNSGA2)
                #toolbox.register("select", tools.selSPEA2)
                toolbox.register("selectNSGACOIN", nsgacoin.selNSGA2COIN, world=my_world)



        elif problem == 'R':
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

            toolbox.register("evaluate", benchmarks.zdt3)


            if moea_alg=='P':
                toolbox.register("select", tools.selSPEA2)
                toolbox.register("selectTournament", tools.selTournament, tournsize=2)
                # toolbox.register("mate", tools.cxSimulatedBinaryBounded, low=BOUND_LOW, up=BOUND_UP, eta=0.5)
                # toolbox.register("mutate", tools.mutPolynomialBounded, low=BOUND_LOW, up=BOUND_UP, eta=0.5, indpb=1.0)
                toolbox.register("mate", tools.cxSimulatedBinaryBounded, low=BOUND_LOW, up=BOUND_UP, eta=2)
                toolbox.register("mutate", tools.mutPolynomialBounded, low=BOUND_LOW, up=BOUND_UP, eta=2, indpb=1.0/NDIM)
                toolbox.register("selectSPEACOIN", speacoin.selSPEA2COIN, world=my_world)

            elif moea_alg=='S':
                toolbox.register("mate", tools.cxSimulatedBinaryBounded, low=BOUND_LOW, up=BOUND_UP, eta=20.0)
                toolbox.register("mutate", tools.mutPolynomialBounded, low=BOUND_LOW, up=BOUND_UP, eta=20.0, indpb=1.0/NDIM)
                toolbox.register("select", smscoin.selSMS)
                toolbox.register("selectSMSCOIN", smscoin.selSMSCOIN, world=my_world)

            else:
                toolbox.register("mate", tools.cxSimulatedBinaryBounded, low=BOUND_LOW, up=BOUND_UP, eta=20.0)
                toolbox.register("mutate", tools.mutPolynomialBounded, low=BOUND_LOW, up=BOUND_UP, eta=20.0, indpb=1.0/NDIM)
                toolbox.register("select", tools.selNSGA2)
                #toolbox.register("select", tools.selSPEA2)
                toolbox.register("selectNSGACOIN", nsgacoin.selNSGA2COIN, world=my_world)

            # toolbox.register("mate", tools.cxSimulatedBinaryBounded, low=BOUND_LOW, up=BOUND_UP, eta=20.0)
            # toolbox.register("mutate", tools.mutPolynomialBounded, low=BOUND_LOW, up=BOUND_UP, eta=20.0, indpb=1.0/NDIM)
            # toolbox.register("select", tools.selNSGA2)
            # #toolbox.register("select", tools.selSPEA2)
            # toolbox.register("selectNSGACOIN", nsgacoin.selNSGA2COIN, world=my_world)

        elif problem == 'S':

            NDIM = 10

            def uniform(low, up, size=None):
                try:
                    return [random.uniform(a, b) for a, b in zip(low, up)]
                except TypeError:
                    return [random.uniform(a, b) for a, b in zip([low] * size, [up] * size)]

            #BOUND_LOW, BOUND_UP = 0.0 + (-5.0*9), 1.0 + (5.0*9)
            BOUND_LOW, BOUND_UP = [0.0] + [-5.0]*9, [1.0] + [5.0]*9
            creator.create("FitnessMin", base.Fitness, weights=(-1.0,-1.0))
            creator.create("Individual", array.array, typecode='d', fitness=creator.FitnessMin)


            toolbox.register("attr_float", uniform, BOUND_LOW, BOUND_UP, 10)
            toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.attr_float)
            toolbox.register("population", tools.initRepeat, list, toolbox.individual)

            toolbox.register("evaluate", benchmarks.zdt4)

            if moea_alg=='P':
                toolbox.register("select", tools.selSPEA2)
                toolbox.register("selectTournament", tools.selTournament, tournsize=2)
                # toolbox.register("mate", tools.cxSimulatedBinaryBounded, low=BOUND_LOW, up=BOUND_UP, eta=0.5)
                # toolbox.register("mutate", tools.mutPolynomialBounded, low=BOUND_LOW, up=BOUND_UP, eta=0.5, indpb=1.0)
                toolbox.register("mate", tools.cxSimulatedBinaryBounded, low=BOUND_LOW, up=BOUND_UP, eta=2)
                toolbox.register("mutate", tools.mutPolynomialBounded, low=BOUND_LOW, up=BOUND_UP, eta=2, indpb=1.0/NDIM)
                toolbox.register("selectSPEACOIN", speacoin.selSPEA2COIN, world=my_world)

            elif moea_alg=='S':
                toolbox.register("mate", tools.cxSimulatedBinaryBounded, low=BOUND_LOW, up=BOUND_UP, eta=20.0)
                toolbox.register("mutate", tools.mutPolynomialBounded, low=BOUND_LOW, up=BOUND_UP, eta=20.0, indpb=1.0/NDIM)
                toolbox.register("select", smscoin.selSMS)
                toolbox.register("selectSMSCOIN", smscoin.selSMSCOIN, world=my_world)

            else:

                toolbox.register("mate", tools.cxSimulatedBinaryBounded, low=BOUND_LOW, up=BOUND_UP, eta=20.0)
                toolbox.register("mutate", tools.mutPolynomialBounded, low=BOUND_LOW, up=BOUND_UP, eta=20.0, indpb=1.0/NDIM)
                toolbox.register("select", tools.selNSGA2)
                #toolbox.register("select", tools.selSPEA2)
                toolbox.register("selectNSGACOIN", nsgacoin.selNSGA2COIN, world=my_world)

        elif problem == 'T':

            NDIM = 30
            print "ssssssss"

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


            if moea_alg=='P':
                toolbox.register("select", tools.selSPEA2)
                toolbox.register("selectTournament", tools.selTournament, tournsize=2)
                # toolbox.register("mate", tools.cxSimulatedBinaryBounded, low=BOUND_LOW, up=BOUND_UP, eta=0.5)
                # toolbox.register("mutate", tools.mutPolynomialBounded, low=BOUND_LOW, up=BOUND_UP, eta=0.5, indpb=1.0)
                toolbox.register("mate", tools.cxSimulatedBinaryBounded, low=BOUND_LOW, up=BOUND_UP, eta=2)
                toolbox.register("mutate", tools.mutPolynomialBounded, low=BOUND_LOW, up=BOUND_UP, eta=2, indpb=1.0/NDIM)
                toolbox.register("selectSPEACOIN", speacoin.selSPEA2COIN, world=my_world)

            elif moea_alg=='S':
                toolbox.register("mate", tools.cxSimulatedBinaryBounded, low=BOUND_LOW, up=BOUND_UP, eta=20.0)
                toolbox.register("mutate", tools.mutPolynomialBounded, low=BOUND_LOW, up=BOUND_UP, eta=20.0, indpb=1.0/NDIM)
                toolbox.register("select", smscoin.selSMS)
                toolbox.register("selectSMSCOIN", smscoin.selSMSCOIN, world=my_world)

            else:
                toolbox.register("mate", tools.cxSimulatedBinaryBounded, low=BOUND_LOW, up=BOUND_UP, eta=20.0)
                toolbox.register("mutate", tools.mutPolynomialBounded, low=BOUND_LOW, up=BOUND_UP, eta=20.0, indpb=1.0/NDIM)
                toolbox.register("select", tools.selNSGA2)
                #toolbox.register("select", tools.selSPEA2)
                toolbox.register("selectNSGACOIN", nsgacoin.selNSGA2COIN, world=my_world)

        elif problem == 'F':


            def uniform(low, up, size=None):
                try:
                    return [random.uniform(a, b) for a, b in zip(low, up)]
                except TypeError:
                    return [random.uniform(a, b) for a, b in zip([low] * size, [up] * size)]

            BOUND_LOW, BOUND_UP = 0.0, 1.0

            creator.create("FitnessMin", base.Fitness, weights=(-1.0,-1.0))
            creator.create("Individual", array.array, typecode='d', fitness=creator.FitnessMin)


            toolbox.register("attr_float", uniform, BOUND_LOW, BOUND_UP, NDIM_DTLZ)
            toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.attr_float)
            toolbox.register("population", tools.initRepeat, list, toolbox.individual)

            toolbox.register("evaluate", lambda ind: benchmarks.dtlz1(ind, 2))


            if moea_alg=='P':
                toolbox.register("select", tools.selSPEA2)
                toolbox.register("selectTournament", tools.selTournament, tournsize=2)
                # toolbox.register("mate", tools.cxSimulatedBinaryBounded, low=BOUND_LOW, up=BOUND_UP, eta=0.5)
                # toolbox.register("mutate", tools.mutPolynomialBounded, low=BOUND_LOW, up=BOUND_UP, eta=0.5, indpb=1.0)
                toolbox.register("mate", tools.cxSimulatedBinaryBounded, low=BOUND_LOW, up=BOUND_UP, eta=2)
                toolbox.register("mutate", tools.mutPolynomialBounded, low=BOUND_LOW, up=BOUND_UP, eta=2, indpb=1.0)
                toolbox.register("selectSPEACOIN", speacoin.selSPEA2COIN, world=my_world)

            elif moea_alg=='S':
                toolbox.register("mate", tools.cxSimulatedBinaryBounded, low=BOUND_LOW, up=BOUND_UP, eta=20.0)
                toolbox.register("mutate", tools.mutPolynomialBounded, low=BOUND_LOW, up=BOUND_UP, eta=20.0, indpb=1.0)
                toolbox.register("select", smscoin.selSMS)
                toolbox.register("selectSMSCOIN", smscoin.selSMSCOIN, world=my_world)


            else:

                # toolbox.register("mate", tools.cxSimulatedBinaryBounded, low=BOUND_LOW, up=BOUND_UP, eta=20.0)
                # toolbox.register("mutate", tools.mutPolynomialBounded, low=BOUND_LOW, up=BOUND_UP, eta=20.0, indpb=1.0/NDIM_DTLZ)
                toolbox.register("mate", tools.cxSimulatedBinaryBounded, low=BOUND_LOW, up=BOUND_UP, eta=0.5)
                toolbox.register("mutate", tools.mutPolynomialBounded, low=BOUND_LOW, up=BOUND_UP, eta=0.5, indpb=1)
                toolbox.register("select", tools.selNSGA2)
                #toolbox.register("select", tools.selSPEA2)
                toolbox.register("selectNSGACOIN", nsgacoin.selNSGA2COIN, world=my_world)


        elif problem == 'G':


            def uniform(low, up, size=None):
                try:
                    return [random.uniform(a, b) for a, b in zip(low, up)]
                except TypeError:
                    return [random.uniform(a, b) for a, b in zip([low] * size, [up] * size)]

            BOUND_LOW, BOUND_UP = 0.0, 1.0

            creator.create("FitnessMin", base.Fitness, weights=(-1.0,-1.0,-1.0))
            creator.create("Individual", array.array, typecode='d', fitness=creator.FitnessMin)


            toolbox.register("attr_float", uniform, BOUND_LOW, BOUND_UP, NDIM_DTLZ)
            toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.attr_float)
            toolbox.register("population", tools.initRepeat, list, toolbox.individual)

            toolbox.register("evaluate", lambda ind: benchmarks.dtlz2(ind, 3))



            if moea_alg=='P':
                toolbox.register("select", tools.selSPEA2)
                toolbox.register("selectTournament", tools.selTournament, tournsize=2)
                # toolbox.register("mate", tools.cxSimulatedBinaryBounded, low=BOUND_LOW, up=BOUND_UP, eta=0.5)
                # toolbox.register("mutate", tools.mutPolynomialBounded, low=BOUND_LOW, up=BOUND_UP, eta=0.5, indpb=1.0)
                toolbox.register("mate", tools.cxSimulatedBinaryBounded, low=BOUND_LOW, up=BOUND_UP, eta=2)
                toolbox.register("mutate", tools.mutPolynomialBounded, low=BOUND_LOW, up=BOUND_UP, eta=2, indpb=1.0)
                toolbox.register("selectSPEACOIN", speacoin.selSPEA2COIN, world=my_world)

            elif moea_alg=='S':
                toolbox.register("mate", tools.cxSimulatedBinaryBounded, low=BOUND_LOW, up=BOUND_UP, eta=20.0)
                toolbox.register("mutate", tools.mutPolynomialBounded, low=BOUND_LOW, up=BOUND_UP, eta=20.0, indpb=1.0/NDIM_DTLZ)
                toolbox.register("select", smscoin.selSMS)
                toolbox.register("selectSMSCOIN", smscoin.selSMSCOIN, world=my_world)

            else:

                toolbox.register("mate", tools.cxSimulatedBinaryBounded, low=BOUND_LOW, up=BOUND_UP, eta=20.0)
                toolbox.register("mutate", tools.mutPolynomialBounded, low=BOUND_LOW, up=BOUND_UP, eta=20.0, indpb=1.0/NDIM_DTLZ)
                toolbox.register("select", tools.selNSGA2)
                #toolbox.register("select", tools.selSPEA2)
                toolbox.register("selectNSGACOIN", nsgacoin.selNSGA2COIN, world=my_world)


        elif problem == 'H':


            def uniform(low, up, size=None):
                try:
                    return [random.uniform(a, b) for a, b in zip(low, up)]
                except TypeError:
                    return [random.uniform(a, b) for a, b in zip([low] * size, [up] * size)]

            BOUND_LOW, BOUND_UP = 0.0, 1.0

            creator.create("FitnessMin", base.Fitness, weights=(-1.0,-1.0,-1.0))
            creator.create("Individual", array.array, typecode='d', fitness=creator.FitnessMin)


            toolbox.register("attr_float", uniform, BOUND_LOW, BOUND_UP, NDIM_DTLZ)
            toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.attr_float)
            toolbox.register("population", tools.initRepeat, list, toolbox.individual)

            toolbox.register("evaluate", lambda ind: benchmarks.dtlz3(ind, 3))




            if moea_alg=='P':
                toolbox.register("select", tools.selSPEA2)
                toolbox.register("selectTournament", tools.selTournament, tournsize=2)
                # toolbox.register("mate", tools.cxSimulatedBinaryBounded, low=BOUND_LOW, up=BOUND_UP, eta=0.5)
                # toolbox.register("mutate", tools.mutPolynomialBounded, low=BOUND_LOW, up=BOUND_UP, eta=0.5, indpb=1.0)
                toolbox.register("mate", tools.cxSimulatedBinaryBounded, low=BOUND_LOW, up=BOUND_UP, eta=2)
                toolbox.register("mutate", tools.mutPolynomialBounded, low=BOUND_LOW, up=BOUND_UP, eta=2, indpb=1.0)
                toolbox.register("selectSPEACOIN", speacoin.selSPEA2COIN, world=my_world)

            elif moea_alg=='S':
                toolbox.register("mate", tools.cxSimulatedBinaryBounded, low=BOUND_LOW, up=BOUND_UP, eta=20.0)
                toolbox.register("mutate", tools.mutPolynomialBounded, low=BOUND_LOW, up=BOUND_UP, eta=20.0, indpb=1.0/NDIM_DTLZ)
                toolbox.register("select", smscoin.selSMS)
                toolbox.register("selectSMSCOIN", smscoin.selSMSCOIN, world=my_world)

            else:

                toolbox.register("mate", tools.cxSimulatedBinaryBounded, low=BOUND_LOW, up=BOUND_UP, eta=20.0)
                toolbox.register("mutate", tools.mutPolynomialBounded, low=BOUND_LOW, up=BOUND_UP, eta=20.0, indpb=1.0/NDIM_DTLZ)
                toolbox.register("select", tools.selNSGA2)
                #toolbox.register("select", tools.selSPEA2)
                toolbox.register("selectNSGACOIN", nsgacoin.selNSGA2COIN, world=my_world)

        elif problem == 'I':


            def uniform(low, up, size=None):
                try:
                    return [random.uniform(a, b) for a, b in zip(low, up)]
                except TypeError:
                    return [random.uniform(a, b) for a, b in zip([low] * size, [up] * size)]

            BOUND_LOW, BOUND_UP = 0.0, 1.0

            creator.create("FitnessMin", base.Fitness, weights=(-1.0,-1.0,-1.0))
            creator.create("Individual", array.array, typecode='d', fitness=creator.FitnessMin)


            toolbox.register("attr_float", uniform, BOUND_LOW, BOUND_UP, NDIM_DTLZ)
            toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.attr_float)
            toolbox.register("population", tools.initRepeat, list, toolbox.individual)

            toolbox.register("evaluate", lambda ind: benchmarks.dtlz4(ind, 3, 100))



            if moea_alg=='P':
                toolbox.register("select", tools.selSPEA2)
                toolbox.register("selectTournament", tools.selTournament, tournsize=2)
                # toolbox.register("mate", tools.cxSimulatedBinaryBounded, low=BOUND_LOW, up=BOUND_UP, eta=0.5)
                # toolbox.register("mutate", tools.mutPolynomialBounded, low=BOUND_LOW, up=BOUND_UP, eta=0.5, indpb=1.0)
                toolbox.register("mate", tools.cxSimulatedBinaryBounded, low=BOUND_LOW, up=BOUND_UP, eta=2)
                toolbox.register("mutate", tools.mutPolynomialBounded, low=BOUND_LOW, up=BOUND_UP, eta=2, indpb=1.0)
                toolbox.register("selectSPEACOIN", speacoin.selSPEA2COIN, world=my_world)

            elif moea_alg=='S':
                toolbox.register("mate", tools.cxSimulatedBinaryBounded, low=BOUND_LOW, up=BOUND_UP, eta=20.0)
                toolbox.register("mutate", tools.mutPolynomialBounded, low=BOUND_LOW, up=BOUND_UP, eta=20.0, indpb=1.0/NDIM_DTLZ)
                toolbox.register("select", smscoin.selSMS)
                toolbox.register("selectSMSCOIN", smscoin.selSMSCOIN, world=my_world)

            else:

                toolbox.register("mate", tools.cxSimulatedBinaryBounded, low=BOUND_LOW, up=BOUND_UP, eta=20.0)
                toolbox.register("mutate", tools.mutPolynomialBounded, low=BOUND_LOW, up=BOUND_UP, eta=20.0, indpb=1.0/NDIM_DTLZ)
                toolbox.register("select", tools.selNSGA2)
                #toolbox.register("select", tools.selSPEA2)
                toolbox.register("selectNSGACOIN", nsgacoin.selNSGA2COIN, world=my_world)

        elif problem == 'J':


            def uniform(low, up, size=None):
                try:
                    return [random.uniform(a, b) for a, b in zip(low, up)]
                except TypeError:
                    return [random.uniform(a, b) for a, b in zip([low] * size, [up] * size)]

            BOUND_LOW, BOUND_UP = 0.0, 1.0

            creator.create("FitnessMin", base.Fitness, weights=(-1.0,-1.0,-1.0))
            creator.create("Individual", array.array, typecode='d', fitness=creator.FitnessMin)


            toolbox.register("attr_float", uniform, BOUND_LOW, BOUND_UP, NDIM_DTLZ)
            toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.attr_float)
            toolbox.register("population", tools.initRepeat, list, toolbox.individual)

            toolbox.register("evaluate", lambda ind: dtlz5(ind, 3))



            if moea_alg=='P':
                toolbox.register("select", tools.selSPEA2)
                toolbox.register("selectTournament", tools.selTournament, tournsize=2)
                # toolbox.register("mate", tools.cxSimulatedBinaryBounded, low=BOUND_LOW, up=BOUND_UP, eta=0.5)
                # toolbox.register("mutate", tools.mutPolynomialBounded, low=BOUND_LOW, up=BOUND_UP, eta=0.5, indpb=1.0)
                toolbox.register("mate", tools.cxSimulatedBinaryBounded, low=BOUND_LOW, up=BOUND_UP, eta=2)
                toolbox.register("mutate", tools.mutPolynomialBounded, low=BOUND_LOW, up=BOUND_UP, eta=2, indpb=1.0)
                toolbox.register("selectSPEACOIN", speacoin.selSPEA2COIN, world=my_world)

            elif moea_alg=='S':
                toolbox.register("mate", tools.cxSimulatedBinaryBounded, low=BOUND_LOW, up=BOUND_UP, eta=20.0)
                toolbox.register("mutate", tools.mutPolynomialBounded, low=BOUND_LOW, up=BOUND_UP, eta=20.0, indpb=1.0/NDIM_DTLZ)
                toolbox.register("select", smscoin.selSMS)
                toolbox.register("selectSMSCOIN", smscoin.selSMSCOIN, world=my_world)

            else:
                toolbox.register("mate", tools.cxSimulatedBinaryBounded, low=BOUND_LOW, up=BOUND_UP, eta=20.0)
                toolbox.register("mutate", tools.mutPolynomialBounded, low=BOUND_LOW, up=BOUND_UP, eta=20.0, indpb=1.0/NDIM_DTLZ)
                toolbox.register("select", tools.selNSGA2)
                #toolbox.register("select", tools.selSPEA2)
                toolbox.register("selectNSGACOIN", nsgacoin.selNSGA2COIN, world=my_world)

        elif problem == 'L':


            def uniform(low, up, size=None):
                try:
                    return [random.uniform(a, b) for a, b in zip(low, up)]
                except TypeError:
                    return [random.uniform(a, b) for a, b in zip([low] * size, [up] * size)]

            BOUND_LOW, BOUND_UP = 0.0, 1.0

            creator.create("FitnessMin", base.Fitness, weights=(-1.0,-1.0,-1.0))
            creator.create("Individual", array.array, typecode='d', fitness=creator.FitnessMin)


            toolbox.register("attr_float", uniform, BOUND_LOW, BOUND_UP, NDIM_DTLZ)
            toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.attr_float)
            toolbox.register("population", tools.initRepeat, list, toolbox.individual)

            toolbox.register("evaluate", lambda ind: dtlz6(ind, 3))



            if moea_alg=='P':
                toolbox.register("select", tools.selSPEA2)
                toolbox.register("selectTournament", tools.selTournament, tournsize=2)
                # toolbox.register("mate", tools.cxSimulatedBinaryBounded, low=BOUND_LOW, up=BOUND_UP, eta=0.5)
                # toolbox.register("mutate", tools.mutPolynomialBounded, low=BOUND_LOW, up=BOUND_UP, eta=0.5, indpb=1.0)
                toolbox.register("mate", tools.cxSimulatedBinaryBounded, low=BOUND_LOW, up=BOUND_UP, eta=2)
                toolbox.register("mutate", tools.mutPolynomialBounded, low=BOUND_LOW, up=BOUND_UP, eta=2, indpb=1.0)
                toolbox.register("selectSPEACOIN", speacoin.selSPEA2COIN, world=my_world)

            elif moea_alg=='S':
                toolbox.register("mate", tools.cxSimulatedBinaryBounded, low=BOUND_LOW, up=BOUND_UP, eta=20.0)
                toolbox.register("mutate", tools.mutPolynomialBounded, low=BOUND_LOW, up=BOUND_UP, eta=20.0, indpb=1.0/NDIM_DTLZ)
                toolbox.register("select", smscoin.selSMS)
                toolbox.register("selectSMSCOIN", smscoin.selSMSCOIN, world=my_world)

            else:
                toolbox.register("mate", tools.cxSimulatedBinaryBounded, low=BOUND_LOW, up=BOUND_UP, eta=20.0)
                toolbox.register("mutate", tools.mutPolynomialBounded, low=BOUND_LOW, up=BOUND_UP, eta=20.0, indpb=1.0/NDIM_DTLZ)
                toolbox.register("select", tools.selNSGA2)
                #toolbox.register("select", tools.selSPEA2)
                toolbox.register("selectNSGACOIN", nsgacoin.selNSGA2COIN, world=my_world)

        elif problem == 'M':


            def uniform(low, up, size=None):
                try:
                    return [random.uniform(a, b) for a, b in zip(low, up)]
                except TypeError:
                    return [random.uniform(a, b) for a, b in zip([low] * size, [up] * size)]

            BOUND_LOW, BOUND_UP = 0.0, 1.0

            creator.create("FitnessMin", base.Fitness, weights=(-1.0,-1.0,-1.0))
            creator.create("Individual", array.array, typecode='d', fitness=creator.FitnessMin)


            toolbox.register("attr_float", uniform, BOUND_LOW, BOUND_UP, NDIM_DTLZ)
            toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.attr_float)
            toolbox.register("population", tools.initRepeat, list, toolbox.individual)

            toolbox.register("evaluate", lambda ind: dtlz7(ind, 3))




            if moea_alg=='P':
                toolbox.register("select", tools.selSPEA2)
                toolbox.register("selectTournament", tools.selTournament, tournsize=2)
                # toolbox.register("mate", tools.cxSimulatedBinaryBounded, low=BOUND_LOW, up=BOUND_UP, eta=0.5)
                # toolbox.register("mutate", tools.mutPolynomialBounded, low=BOUND_LOW, up=BOUND_UP, eta=0.5, indpb=1.0)
                toolbox.register("mate", tools.cxSimulatedBinaryBounded, low=BOUND_LOW, up=BOUND_UP, eta=2)
                toolbox.register("mutate", tools.mutPolynomialBounded, low=BOUND_LOW, up=BOUND_UP, eta=2, indpb=1.0)
                toolbox.register("selectSPEACOIN", speacoin.selSPEA2COIN, world=my_world)

            elif moea_alg=='S':
                toolbox.register("mate", tools.cxSimulatedBinaryBounded, low=BOUND_LOW, up=BOUND_UP, eta=20.0)
                toolbox.register("mutate", tools.mutPolynomialBounded, low=BOUND_LOW, up=BOUND_UP, eta=20.0, indpb=1.0/NDIM_DTLZ)
                toolbox.register("select", smscoin.selSMS)
                toolbox.register("selectSMSCOIN", smscoin.selSMSCOIN, world=my_world)

            else:
                toolbox.register("mate", tools.cxSimulatedBinaryBounded, low=BOUND_LOW, up=BOUND_UP, eta=20.0)
                toolbox.register("mutate", tools.mutPolynomialBounded, low=BOUND_LOW, up=BOUND_UP, eta=20.0, indpb=1.0/NDIM_DTLZ)
                toolbox.register("select", tools.selNSGA2)
                #toolbox.register("select", tools.selSPEA2)
                toolbox.register("selectNSGACOIN", nsgacoin.selNSGA2COIN, world=my_world)

        #parameters: mutation and number, generations, number of population
        self.CXPB  = CXPB
        self.MUTPB = MUTPB
        self.NGEN  = NGEN
        self.num_population = num_population

        self.stats = tools.Statistics(lambda ind: ind.fitness.values)
        self.logbook = tools.Logbook()
    # def __init__(self, my_world, CXPB=None, MUTPB=None, NGEN = None,  num_population=None, problem='A', moea_alg='N'):
    #
    #     if problem == 'A':
    #
    #         #define fitness of the individual
    #         creator.create("FitnessMin", base.Fitness, weights=(-1.0,-1.0))
    #         creator.create("Individual", list, fitness=creator.FitnessMin)
    #
    #         # Attribute generator
    #         toolbox.register("individual_gen", my_world.CreateFullindividual)
    #
    #         # Structure initializers
    #         toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.individual_gen, 1)
    #         toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    #
    #         # Operator registering
    #         toolbox.register("evaluate", f1Cost, world=my_world)
    #         toolbox.register("evaluatePAIR", f1CostPAIR, world=my_world)
    #
    #         #toolbox.register("mate", tools.cxTwoPoint) #changes between 2 points
    #         #toolbox.register("mate", tools.cxOnePoint) #changes before and after a randon point
    #         toolbox.register("mate", tools.cxUniform, indpb=CXPB) #changes vector values uniformly by indpb
    #         #cxPartialyMatched = index vector, it does not fit to my structure
    #         #cxUniformPartialyMatched = index vector, it does not fit to my stucture
    #         #cxOrdered = index vector, it does not fit to my stucture
    #         #cxBlend = expects floating points
    #         #cxSimulatedBinary =  expects floating points
    #         #cxSimulatedBinaryBounded =  expects floating points
    #         #cxMessyOnePoint = change the individuals size.
    #
    #         toolbox.register("mutate", tools.mutUniformInt,low=0, up=min(my_world.n-1, my_world.m-1), indpb=MUTPB) #gives an integer between 0 or M
    #         #toolbox.register("mutate", tools.mutShuffleIndexes, indpb=MUTPB) #swap index of individuals
    #         #mutGaussian= not applicable to my dna
    #         #mutPolynomialBounded ?
    #         #mutShuffleIndexes = vector of indexes
    #         #mutPolynomialBounded = floating points, not
    #
    #         #toolbox.register("select", tools.selTournament, tournsize=3)
    #         #toolbox.register("select", tools.selRoulette)
    #         toolbox.register("select", tools.selNSGA2)
    #         #toolbox.register("select", tools.selSPEA2)
    #         toolbox.register("selectNSGACOIN", nsgacoin.selNSGA2COIN, world=my_world)
    #
    #
    #         toolbox.register("selectTournament", tools.selTournament, tournsize=2)
    #
    #     elif problem == 'P':
    #
    #         NDIM = 30
    #
    #
    #         def uniform(low, up, size=None):
    #             try:
    #                 return [random.uniform(a, b) for a, b in zip(low, up)]
    #             except TypeError:
    #                 return [random.uniform(a, b) for a, b in zip([low] * size, [up] * size)]
    #
    #         BOUND_LOW, BOUND_UP = 0.0, 1.0
    #         creator.create("FitnessMin", base.Fitness, weights=(-1.0,-1.0))
    #         creator.create("Individual", array.array, typecode='d', fitness=creator.FitnessMin)
    #
    #
    #         toolbox.register("attr_float", uniform, BOUND_LOW, BOUND_UP, NDIM)
    #         toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.attr_float)
    #         toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    #
    #         #toolbox.register("mate", tools.cxSimulatedBinaryBounded, low=BOUND_LOW, up=BOUND_UP, eta=20.0)
    #         #toolbox.register("mutate", tools.mutPolynomialBounded, low=BOUND_LOW, up=BOUND_UP, eta=20.0, indpb=1.0/NDIM)
    #
    #         toolbox.register("evaluate", benchmarks.zdt1)
    #
    #         if moea_alg=='P':
    #             toolbox.register("select", tools.selSPEA2)
    #             #toolbox.register("select", speacoin.selSPEA2_PURE)
    #             toolbox.register("selectTournament", tools.selTournament, tournsize=2)
    #             # toolbox.register("mate", tools.cxSimulatedBinaryBounded, low=BOUND_LOW, up=BOUND_UP, eta=0.5)
    #             # toolbox.register("mutate", tools.mutPolynomialBounded, low=BOUND_LOW, up=BOUND_UP, eta=0.5, indpb=1.0)
    #             toolbox.register("mate", tools.cxSimulatedBinaryBounded, low=BOUND_LOW, up=BOUND_UP, eta=4.0)
    #             toolbox.register("mutate", tools.mutPolynomialBounded, low=BOUND_LOW, up=BOUND_UP, eta=4.0, indpb=1.0/NDIM)
    #             toolbox.register("selectSPEACOIN", speacoin.selSPEA2COIN, world=my_world)
    #
    #         elif moea_alg=='P':
    #             toolbox.register("mate", tools.cxSimulatedBinaryBounded, low=BOUND_LOW, up=BOUND_UP, eta=20.0)
    #             toolbox.register("mutate", tools.mutPolynomialBounded, low=BOUND_LOW, up=BOUND_UP, eta=20.0, indpb=1.0/NDIM)
    #             toolbox.register("select", tools.selNSGA2)
    #             toolbox.register("selectNSGACOIN", nsgacoin.selNSGA2COIN, world=my_world)
    #             toolbox.register("selectNSGAPURE", nsgacoin.selNSGA2PURE)
    #
    #         elif moea_alg=='S':
    #             toolbox.register("mate", tools.cxSimulatedBinaryBounded, low=BOUND_LOW, up=BOUND_UP, eta=20.0)
    #             toolbox.register("mutate", tools.mutPolynomialBounded, low=BOUND_LOW, up=BOUND_UP, eta=20.0, indpb=1.0/NDIM)
    #             toolbox.register("select", smscoin.selSMS)
    #             toolbox.register("selectSMSCOIN", smscoin.selSMSCOIN, world=my_world)
    #
    #
    #
    #
    #     elif problem == 'Q':
    #
    #         NDIM = 30
    #
    #         def uniform(low, up, size=None):
    #             try:
    #                 return [random.uniform(a, b) for a, b in zip(low, up)]
    #             except TypeError:
    #                 return [random.uniform(a, b) for a, b in zip([low] * size, [up] * size)]
    #
    #         BOUND_LOW, BOUND_UP = 0.0, 1.0
    #         creator.create("FitnessMin", base.Fitness, weights=(-1.0,-1.0))
    #         creator.create("Individual", array.array, typecode='d', fitness=creator.FitnessMin)
    #
    #
    #         toolbox.register("attr_float", uniform, BOUND_LOW, BOUND_UP, NDIM)
    #         toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.attr_float)
    #         toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    #
    #         toolbox.register("evaluate", benchmarks.zdt2)
    #
    #
    #         if moea_alg=='P':
    #             toolbox.register("select", tools.selSPEA2)
    #             toolbox.register("selectTournament", tools.selTournament, tournsize=2)
    #             # toolbox.register("mate", tools.cxSimulatedBinaryBounded, low=BOUND_LOW, up=BOUND_UP, eta=0.5)
    #             # toolbox.register("mutate", tools.mutPolynomialBounded, low=BOUND_LOW, up=BOUND_UP, eta=0.5, indpb=1.0)
    #             toolbox.register("mate", tools.cxSimulatedBinaryBounded, low=BOUND_LOW, up=BOUND_UP, eta=2)
    #             toolbox.register("mutate", tools.mutPolynomialBounded, low=BOUND_LOW, up=BOUND_UP, eta=2, indpb=1.0/NDIM)
    #             toolbox.register("selectSPEACOIN", speacoin.selSPEA2COIN, world=my_world)
    #
    #         elif moea_alg=='S':
    #             toolbox.register("mate", tools.cxSimulatedBinaryBounded, low=BOUND_LOW, up=BOUND_UP, eta=20.0)
    #             toolbox.register("mutate", tools.mutPolynomialBounded, low=BOUND_LOW, up=BOUND_UP, eta=20.0, indpb=1.0/NDIM)
    #             toolbox.register("select", smscoin.selSMS)
    #             toolbox.register("selectSMSCOIN", smscoin.selSMSCOIN, world=my_world)
    #
    #         else:
    #             toolbox.register("mate", tools.cxSimulatedBinaryBounded, low=BOUND_LOW, up=BOUND_UP, eta=20.0)
    #             toolbox.register("mutate", tools.mutPolynomialBounded, low=BOUND_LOW, up=BOUND_UP, eta=20.0, indpb=1.0/NDIM)
    #             toolbox.register("select", tools.selNSGA2)
    #             #toolbox.register("select", tools.selSPEA2)
    #             toolbox.register("selectNSGACOIN", nsgacoin.selNSGA2COIN, world=my_world)
    #
    #
    #
    #     elif problem == 'R':
    #         NDIM = 30
    #
    #
    #         def uniform(low, up, size=None):
    #             try:
    #                 return [random.uniform(a, b) for a, b in zip(low, up)]
    #             except TypeError:
    #                 return [random.uniform(a, b) for a, b in zip([low] * size, [up] * size)]
    #
    #         BOUND_LOW, BOUND_UP = 0.0, 1.0
    #         creator.create("FitnessMin", base.Fitness, weights=(-1.0,-1.0))
    #         creator.create("Individual", array.array, typecode='d', fitness=creator.FitnessMin)
    #
    #
    #         toolbox.register("attr_float", uniform, BOUND_LOW, BOUND_UP, NDIM)
    #         toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.attr_float)
    #         toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    #
    #         toolbox.register("evaluate", benchmarks.zdt3)
    #
    #
    #         if moea_alg=='P':
    #             toolbox.register("select", tools.selSPEA2)
    #             toolbox.register("selectTournament", tools.selTournament, tournsize=2)
    #             # toolbox.register("mate", tools.cxSimulatedBinaryBounded, low=BOUND_LOW, up=BOUND_UP, eta=0.5)
    #             # toolbox.register("mutate", tools.mutPolynomialBounded, low=BOUND_LOW, up=BOUND_UP, eta=0.5, indpb=1.0)
    #             toolbox.register("mate", tools.cxSimulatedBinaryBounded, low=BOUND_LOW, up=BOUND_UP, eta=2)
    #             toolbox.register("mutate", tools.mutPolynomialBounded, low=BOUND_LOW, up=BOUND_UP, eta=2, indpb=1.0/NDIM)
    #             toolbox.register("selectSPEACOIN", speacoin.selSPEA2COIN, world=my_world)
    #
    #         elif moea_alg=='S':
    #             toolbox.register("mate", tools.cxSimulatedBinaryBounded, low=BOUND_LOW, up=BOUND_UP, eta=20.0)
    #             toolbox.register("mutate", tools.mutPolynomialBounded, low=BOUND_LOW, up=BOUND_UP, eta=20.0, indpb=1.0/NDIM)
    #             toolbox.register("select", smscoin.selSMS)
    #             toolbox.register("selectSMSCOIN", smscoin.selSMSCOIN, world=my_world)
    #
    #         else:
    #             toolbox.register("mate", tools.cxSimulatedBinaryBounded, low=BOUND_LOW, up=BOUND_UP, eta=20.0)
    #             toolbox.register("mutate", tools.mutPolynomialBounded, low=BOUND_LOW, up=BOUND_UP, eta=20.0, indpb=1.0/NDIM)
    #             toolbox.register("select", tools.selNSGA2)
    #             #toolbox.register("select", tools.selSPEA2)
    #             toolbox.register("selectNSGACOIN", nsgacoin.selNSGA2COIN, world=my_world)
    #
    #         # toolbox.register("mate", tools.cxSimulatedBinaryBounded, low=BOUND_LOW, up=BOUND_UP, eta=20.0)
    #         # toolbox.register("mutate", tools.mutPolynomialBounded, low=BOUND_LOW, up=BOUND_UP, eta=20.0, indpb=1.0/NDIM)
    #         # toolbox.register("select", tools.selNSGA2)
    #         # #toolbox.register("select", tools.selSPEA2)
    #         # toolbox.register("selectNSGACOIN", nsgacoin.selNSGA2COIN, world=my_world)
    #
    #     elif problem == 'S':
    #
    #         NDIM = 10
    #
    #         def uniform(low, up, size=None):
    #             try:
    #                 return [random.uniform(a, b) for a, b in zip(low, up)]
    #             except TypeError:
    #                 return [random.uniform(a, b) for a, b in zip([low] * size, [up] * size)]
    #
    #         #BOUND_LOW, BOUND_UP = 0.0 + (-5.0*9), 1.0 + (5.0*9)
    #         BOUND_LOW, BOUND_UP = [0.0] + [-5.0]*9, [1.0] + [5.0]*9
    #         creator.create("FitnessMin", base.Fitness, weights=(-1.0,-1.0))
    #         creator.create("Individual", array.array, typecode='d', fitness=creator.FitnessMin)
    #
    #
    #         toolbox.register("attr_float", uniform, BOUND_LOW, BOUND_UP, 10)
    #         toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.attr_float)
    #         toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    #
    #         toolbox.register("evaluate", benchmarks.zdt4)
    #
    #         if moea_alg=='P':
    #             toolbox.register("select", tools.selSPEA2)
    #             toolbox.register("selectTournament", tools.selTournament, tournsize=2)
    #             # toolbox.register("mate", tools.cxSimulatedBinaryBounded, low=BOUND_LOW, up=BOUND_UP, eta=0.5)
    #             # toolbox.register("mutate", tools.mutPolynomialBounded, low=BOUND_LOW, up=BOUND_UP, eta=0.5, indpb=1.0)
    #             toolbox.register("mate", tools.cxSimulatedBinaryBounded, low=BOUND_LOW, up=BOUND_UP, eta=2)
    #             toolbox.register("mutate", tools.mutPolynomialBounded, low=BOUND_LOW, up=BOUND_UP, eta=2, indpb=1.0/NDIM)
    #             toolbox.register("selectSPEACOIN", speacoin.selSPEA2COIN, world=my_world)
    #
    #         elif moea_alg=='S':
    #             toolbox.register("mate", tools.cxSimulatedBinaryBounded, low=BOUND_LOW, up=BOUND_UP, eta=20.0)
    #             toolbox.register("mutate", tools.mutPolynomialBounded, low=BOUND_LOW, up=BOUND_UP, eta=20.0, indpb=1.0/NDIM)
    #             toolbox.register("select", smscoin.selSMS)
    #             toolbox.register("selectSMSCOIN", smscoin.selSMSCOIN, world=my_world)
    #
    #         else:
    #
    #             toolbox.register("mate", tools.cxSimulatedBinaryBounded, low=BOUND_LOW, up=BOUND_UP, eta=20.0)
    #             toolbox.register("mutate", tools.mutPolynomialBounded, low=BOUND_LOW, up=BOUND_UP, eta=20.0, indpb=1.0/NDIM)
    #             toolbox.register("select", tools.selNSGA2)
    #             #toolbox.register("select", tools.selSPEA2)
    #             toolbox.register("selectNSGACOIN", nsgacoin.selNSGA2COIN, world=my_world)
    #
    #     elif problem == 'T':
    #
    #         NDIM = 30
    #         print "ssssssss"
    #
    #         def uniform(low, up, size=None):
    #             try:
    #                 return [random.uniform(a, b) for a, b in zip(low, up)]
    #             except TypeError:
    #                 return [random.uniform(a, b) for a, b in zip([low] * size, [up] * size)]
    #
    #         BOUND_LOW, BOUND_UP = 0.0, 1.0
    #         creator.create("FitnessMin", base.Fitness, weights=(-1.0,-1.0))
    #         creator.create("Individual", array.array, typecode='d', fitness=creator.FitnessMin)
    #
    #
    #         toolbox.register("attr_float", uniform, BOUND_LOW, BOUND_UP, NDIM)
    #         toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.attr_float)
    #         toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    #
    #         toolbox.register("evaluate", benchmarks.zdt6)
    #
    #
    #         if moea_alg=='P':
    #             toolbox.register("select", tools.selSPEA2)
    #             toolbox.register("selectTournament", tools.selTournament, tournsize=2)
    #             # toolbox.register("mate", tools.cxSimulatedBinaryBounded, low=BOUND_LOW, up=BOUND_UP, eta=0.5)
    #             # toolbox.register("mutate", tools.mutPolynomialBounded, low=BOUND_LOW, up=BOUND_UP, eta=0.5, indpb=1.0)
    #             toolbox.register("mate", tools.cxSimulatedBinaryBounded, low=BOUND_LOW, up=BOUND_UP, eta=2)
    #             toolbox.register("mutate", tools.mutPolynomialBounded, low=BOUND_LOW, up=BOUND_UP, eta=2, indpb=1.0/NDIM)
    #             toolbox.register("selectSPEACOIN", speacoin.selSPEA2COIN, world=my_world)
    #
    #         elif moea_alg=='S':
    #             toolbox.register("mate", tools.cxSimulatedBinaryBounded, low=BOUND_LOW, up=BOUND_UP, eta=20.0)
    #             toolbox.register("mutate", tools.mutPolynomialBounded, low=BOUND_LOW, up=BOUND_UP, eta=20.0, indpb=1.0/NDIM)
    #             toolbox.register("select", smscoin.selSMS)
    #             toolbox.register("selectSMSCOIN", smscoin.selSMSCOIN, world=my_world)
    #
    #         else:
    #             toolbox.register("mate", tools.cxSimulatedBinaryBounded, low=BOUND_LOW, up=BOUND_UP, eta=20.0)
    #             toolbox.register("mutate", tools.mutPolynomialBounded, low=BOUND_LOW, up=BOUND_UP, eta=20.0, indpb=1.0/NDIM)
    #             toolbox.register("select", tools.selNSGA2)
    #             #toolbox.register("select", tools.selSPEA2)
    #             toolbox.register("selectNSGACOIN", nsgacoin.selNSGA2COIN, world=my_world)
    #
    #     elif problem == 'F':
    #
    #
    #         def uniform(low, up, size=None):
    #             try:
    #                 return [random.uniform(a, b) for a, b in zip(low, up)]
    #             except TypeError:
    #                 return [random.uniform(a, b) for a, b in zip([low] * size, [up] * size)]
    #
    #         BOUND_LOW, BOUND_UP = 0.0, 1.0
    #
    #         creator.create("FitnessMin", base.Fitness, weights=(-1.0,-1.0))
    #         creator.create("Individual", array.array, typecode='d', fitness=creator.FitnessMin)
    #
    #
    #         toolbox.register("attr_float", uniform, BOUND_LOW, BOUND_UP, NDIM_DTLZ)
    #         toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.attr_float)
    #         toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    #
    #         toolbox.register("evaluate", lambda ind: benchmarks.dtlz1(ind, 2))
    #
    #
    #         if moea_alg=='P':
    #             toolbox.register("select", tools.selSPEA2)
    #             toolbox.register("selectTournament", tools.selTournament, tournsize=2)
    #             # toolbox.register("mate", tools.cxSimulatedBinaryBounded, low=BOUND_LOW, up=BOUND_UP, eta=0.5)
    #             # toolbox.register("mutate", tools.mutPolynomialBounded, low=BOUND_LOW, up=BOUND_UP, eta=0.5, indpb=1.0)
    #             toolbox.register("mate", tools.cxSimulatedBinaryBounded, low=BOUND_LOW, up=BOUND_UP, eta=2)
    #             toolbox.register("mutate", tools.mutPolynomialBounded, low=BOUND_LOW, up=BOUND_UP, eta=2, indpb=1.0)
    #             toolbox.register("selectSPEACOIN", speacoin.selSPEA2COIN, world=my_world)
    #
    #         elif moea_alg=='S':
    #             toolbox.register("mate", tools.cxSimulatedBinaryBounded, low=BOUND_LOW, up=BOUND_UP, eta=20.0)
    #             toolbox.register("mutate", tools.mutPolynomialBounded, low=BOUND_LOW, up=BOUND_UP, eta=20.0, indpb=1.0)
    #             toolbox.register("select", smscoin.selSMS)
    #             toolbox.register("selectSMSCOIN", smscoin.selSMSCOIN, world=my_world)
    #
    #
    #         else:
    #
    #             # toolbox.register("mate", tools.cxSimulatedBinaryBounded, low=BOUND_LOW, up=BOUND_UP, eta=20.0)
    #             # toolbox.register("mutate", tools.mutPolynomialBounded, low=BOUND_LOW, up=BOUND_UP, eta=20.0, indpb=1.0/NDIM_DTLZ)
    #             toolbox.register("mate", tools.cxSimulatedBinaryBounded, low=BOUND_LOW, up=BOUND_UP, eta=0.5)
    #             toolbox.register("mutate", tools.mutPolynomialBounded, low=BOUND_LOW, up=BOUND_UP, eta=0.5, indpb=1)
    #             toolbox.register("select", tools.selNSGA2)
    #             #toolbox.register("select", tools.selSPEA2)
    #             toolbox.register("selectNSGACOIN", nsgacoin.selNSGA2COIN, world=my_world)
    #
    #
    #     elif problem == 'G':
    #
    #
    #         def uniform(low, up, size=None):
    #             try:
    #                 return [random.uniform(a, b) for a, b in zip(low, up)]
    #             except TypeError:
    #                 return [random.uniform(a, b) for a, b in zip([low] * size, [up] * size)]
    #
    #         BOUND_LOW, BOUND_UP = 0.0, 1.0
    #
    #         creator.create("FitnessMin", base.Fitness, weights=(-1.0,-1.0,-1.0))
    #         creator.create("Individual", array.array, typecode='d', fitness=creator.FitnessMin)
    #
    #
    #         toolbox.register("attr_float", uniform, BOUND_LOW, BOUND_UP, NDIM_DTLZ)
    #         toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.attr_float)
    #         toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    #
    #         toolbox.register("evaluate", lambda ind: benchmarks.dtlz2(ind, 3))
    #
    #
    #
    #         if moea_alg=='P':
    #             toolbox.register("select", tools.selSPEA2)
    #             toolbox.register("selectTournament", tools.selTournament, tournsize=2)
    #             # toolbox.register("mate", tools.cxSimulatedBinaryBounded, low=BOUND_LOW, up=BOUND_UP, eta=0.5)
    #             # toolbox.register("mutate", tools.mutPolynomialBounded, low=BOUND_LOW, up=BOUND_UP, eta=0.5, indpb=1.0)
    #             toolbox.register("mate", tools.cxSimulatedBinaryBounded, low=BOUND_LOW, up=BOUND_UP, eta=2)
    #             toolbox.register("mutate", tools.mutPolynomialBounded, low=BOUND_LOW, up=BOUND_UP, eta=2, indpb=1.0)
    #             toolbox.register("selectSPEACOIN", speacoin.selSPEA2COIN, world=my_world)
    #
    #         elif moea_alg=='S':
    #             toolbox.register("mate", tools.cxSimulatedBinaryBounded, low=BOUND_LOW, up=BOUND_UP, eta=20.0)
    #             toolbox.register("mutate", tools.mutPolynomialBounded, low=BOUND_LOW, up=BOUND_UP, eta=20.0, indpb=1.0/NDIM_DTLZ)
    #             toolbox.register("select", smscoin.selSMS)
    #             toolbox.register("selectSMSCOIN", smscoin.selSMSCOIN, world=my_world)
    #
    #         else:
    #
    #             toolbox.register("mate", tools.cxSimulatedBinaryBounded, low=BOUND_LOW, up=BOUND_UP, eta=20.0)
    #             toolbox.register("mutate", tools.mutPolynomialBounded, low=BOUND_LOW, up=BOUND_UP, eta=20.0, indpb=1.0/NDIM_DTLZ)
    #             toolbox.register("select", tools.selNSGA2)
    #             #toolbox.register("select", tools.selSPEA2)
    #             toolbox.register("selectNSGACOIN", nsgacoin.selNSGA2COIN, world=my_world)
    #
    #
    #     elif problem == 'H':
    #
    #
    #         def uniform(low, up, size=None):
    #             try:
    #                 return [random.uniform(a, b) for a, b in zip(low, up)]
    #             except TypeError:
    #                 return [random.uniform(a, b) for a, b in zip([low] * size, [up] * size)]
    #
    #         BOUND_LOW, BOUND_UP = 0.0, 1.0
    #
    #         creator.create("FitnessMin", base.Fitness, weights=(-1.0,-1.0,-1.0))
    #         creator.create("Individual", array.array, typecode='d', fitness=creator.FitnessMin)
    #
    #
    #         toolbox.register("attr_float", uniform, BOUND_LOW, BOUND_UP, NDIM_DTLZ)
    #         toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.attr_float)
    #         toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    #
    #         toolbox.register("evaluate", lambda ind: benchmarks.dtlz3(ind, 3))
    #
    #
    #
    #
    #         if moea_alg=='P':
    #             toolbox.register("select", tools.selSPEA2)
    #             toolbox.register("selectTournament", tools.selTournament, tournsize=2)
    #             # toolbox.register("mate", tools.cxSimulatedBinaryBounded, low=BOUND_LOW, up=BOUND_UP, eta=0.5)
    #             # toolbox.register("mutate", tools.mutPolynomialBounded, low=BOUND_LOW, up=BOUND_UP, eta=0.5, indpb=1.0)
    #             toolbox.register("mate", tools.cxSimulatedBinaryBounded, low=BOUND_LOW, up=BOUND_UP, eta=2)
    #             toolbox.register("mutate", tools.mutPolynomialBounded, low=BOUND_LOW, up=BOUND_UP, eta=2, indpb=1.0)
    #             toolbox.register("selectSPEACOIN", speacoin.selSPEA2COIN, world=my_world)
    #
    #         elif moea_alg=='S':
    #             toolbox.register("mate", tools.cxSimulatedBinaryBounded, low=BOUND_LOW, up=BOUND_UP, eta=20.0)
    #             toolbox.register("mutate", tools.mutPolynomialBounded, low=BOUND_LOW, up=BOUND_UP, eta=20.0, indpb=1.0/NDIM_DTLZ)
    #             toolbox.register("select", smscoin.selSMS)
    #             toolbox.register("selectSMSCOIN", smscoin.selSMSCOIN, world=my_world)
    #
    #         else:
    #
    #             toolbox.register("mate", tools.cxSimulatedBinaryBounded, low=BOUND_LOW, up=BOUND_UP, eta=20.0)
    #             toolbox.register("mutate", tools.mutPolynomialBounded, low=BOUND_LOW, up=BOUND_UP, eta=20.0, indpb=1.0/NDIM_DTLZ)
    #             toolbox.register("select", tools.selNSGA2)
    #             #toolbox.register("select", tools.selSPEA2)
    #             toolbox.register("selectNSGACOIN", nsgacoin.selNSGA2COIN, world=my_world)
    #
    #     elif problem == 'I':
    #
    #
    #         def uniform(low, up, size=None):
    #             try:
    #                 return [random.uniform(a, b) for a, b in zip(low, up)]
    #             except TypeError:
    #                 return [random.uniform(a, b) for a, b in zip([low] * size, [up] * size)]
    #
    #         BOUND_LOW, BOUND_UP = 0.0, 1.0
    #
    #         creator.create("FitnessMin", base.Fitness, weights=(-1.0,-1.0,-1.0))
    #         creator.create("Individual", array.array, typecode='d', fitness=creator.FitnessMin)
    #
    #
    #         toolbox.register("attr_float", uniform, BOUND_LOW, BOUND_UP, NDIM_DTLZ)
    #         toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.attr_float)
    #         toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    #
    #         toolbox.register("evaluate", lambda ind: benchmarks.dtlz4(ind, 3, 100))
    #
    #
    #
    #         if moea_alg=='P':
    #             toolbox.register("select", tools.selSPEA2)
    #             toolbox.register("selectTournament", tools.selTournament, tournsize=2)
    #             # toolbox.register("mate", tools.cxSimulatedBinaryBounded, low=BOUND_LOW, up=BOUND_UP, eta=0.5)
    #             # toolbox.register("mutate", tools.mutPolynomialBounded, low=BOUND_LOW, up=BOUND_UP, eta=0.5, indpb=1.0)
    #             toolbox.register("mate", tools.cxSimulatedBinaryBounded, low=BOUND_LOW, up=BOUND_UP, eta=2)
    #             toolbox.register("mutate", tools.mutPolynomialBounded, low=BOUND_LOW, up=BOUND_UP, eta=2, indpb=1.0)
    #             toolbox.register("selectSPEACOIN", speacoin.selSPEA2COIN, world=my_world)
    #
    #         elif moea_alg=='S':
    #             toolbox.register("mate", tools.cxSimulatedBinaryBounded, low=BOUND_LOW, up=BOUND_UP, eta=20.0)
    #             toolbox.register("mutate", tools.mutPolynomialBounded, low=BOUND_LOW, up=BOUND_UP, eta=20.0, indpb=1.0/NDIM_DTLZ)
    #             toolbox.register("select", smscoin.selSMS)
    #             toolbox.register("selectSMSCOIN", smscoin.selSMSCOIN, world=my_world)
    #
    #         else:
    #
    #             toolbox.register("mate", tools.cxSimulatedBinaryBounded, low=BOUND_LOW, up=BOUND_UP, eta=20.0)
    #             toolbox.register("mutate", tools.mutPolynomialBounded, low=BOUND_LOW, up=BOUND_UP, eta=20.0, indpb=1.0/NDIM_DTLZ)
    #             toolbox.register("select", tools.selNSGA2)
    #             #toolbox.register("select", tools.selSPEA2)
    #             toolbox.register("selectNSGACOIN", nsgacoin.selNSGA2COIN, world=my_world)
    #
    #     elif problem == 'J':
    #
    #
    #         def uniform(low, up, size=None):
    #             try:
    #                 return [random.uniform(a, b) for a, b in zip(low, up)]
    #             except TypeError:
    #                 return [random.uniform(a, b) for a, b in zip([low] * size, [up] * size)]
    #
    #         BOUND_LOW, BOUND_UP = 0.0, 1.0
    #
    #         creator.create("FitnessMin", base.Fitness, weights=(-1.0,-1.0,-1.0))
    #         creator.create("Individual", array.array, typecode='d', fitness=creator.FitnessMin)
    #
    #
    #         toolbox.register("attr_float", uniform, BOUND_LOW, BOUND_UP, NDIM_DTLZ)
    #         toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.attr_float)
    #         toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    #
    #         toolbox.register("evaluate", lambda ind: dtlz5(ind, 3))
    #
    #
    #
    #         if moea_alg=='P':
    #             toolbox.register("select", tools.selSPEA2)
    #             toolbox.register("selectTournament", tools.selTournament, tournsize=2)
    #             # toolbox.register("mate", tools.cxSimulatedBinaryBounded, low=BOUND_LOW, up=BOUND_UP, eta=0.5)
    #             # toolbox.register("mutate", tools.mutPolynomialBounded, low=BOUND_LOW, up=BOUND_UP, eta=0.5, indpb=1.0)
    #             toolbox.register("mate", tools.cxSimulatedBinaryBounded, low=BOUND_LOW, up=BOUND_UP, eta=2)
    #             toolbox.register("mutate", tools.mutPolynomialBounded, low=BOUND_LOW, up=BOUND_UP, eta=2, indpb=1.0)
    #             toolbox.register("selectSPEACOIN", speacoin.selSPEA2COIN, world=my_world)
    #
    #         elif moea_alg=='S':
    #             toolbox.register("mate", tools.cxSimulatedBinaryBounded, low=BOUND_LOW, up=BOUND_UP, eta=20.0)
    #             toolbox.register("mutate", tools.mutPolynomialBounded, low=BOUND_LOW, up=BOUND_UP, eta=20.0, indpb=1.0/NDIM_DTLZ)
    #             toolbox.register("select", smscoin.selSMS)
    #             toolbox.register("selectSMSCOIN", smscoin.selSMSCOIN, world=my_world)
    #
    #         else:
    #             toolbox.register("mate", tools.cxSimulatedBinaryBounded, low=BOUND_LOW, up=BOUND_UP, eta=20.0)
    #             toolbox.register("mutate", tools.mutPolynomialBounded, low=BOUND_LOW, up=BOUND_UP, eta=20.0, indpb=1.0/NDIM_DTLZ)
    #             toolbox.register("select", tools.selNSGA2)
    #             #toolbox.register("select", tools.selSPEA2)
    #             toolbox.register("selectNSGACOIN", nsgacoin.selNSGA2COIN, world=my_world)
    #
    #     elif problem == 'L':
    #
    #
    #         def uniform(low, up, size=None):
    #             try:
    #                 return [random.uniform(a, b) for a, b in zip(low, up)]
    #             except TypeError:
    #                 return [random.uniform(a, b) for a, b in zip([low] * size, [up] * size)]
    #
    #         BOUND_LOW, BOUND_UP = 0.0, 1.0
    #
    #         creator.create("FitnessMin", base.Fitness, weights=(-1.0,-1.0,-1.0))
    #         creator.create("Individual", array.array, typecode='d', fitness=creator.FitnessMin)
    #
    #
    #         toolbox.register("attr_float", uniform, BOUND_LOW, BOUND_UP, NDIM_DTLZ)
    #         toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.attr_float)
    #         toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    #
    #         toolbox.register("evaluate", lambda ind: dtlz6(ind, 3))
    #
    #
    #
    #         if moea_alg=='P':
    #             toolbox.register("select", tools.selSPEA2)
    #             toolbox.register("selectTournament", tools.selTournament, tournsize=2)
    #             # toolbox.register("mate", tools.cxSimulatedBinaryBounded, low=BOUND_LOW, up=BOUND_UP, eta=0.5)
    #             # toolbox.register("mutate", tools.mutPolynomialBounded, low=BOUND_LOW, up=BOUND_UP, eta=0.5, indpb=1.0)
    #             toolbox.register("mate", tools.cxSimulatedBinaryBounded, low=BOUND_LOW, up=BOUND_UP, eta=2)
    #             toolbox.register("mutate", tools.mutPolynomialBounded, low=BOUND_LOW, up=BOUND_UP, eta=2, indpb=1.0)
    #             toolbox.register("selectSPEACOIN", speacoin.selSPEA2COIN, world=my_world)
    #
    #         elif moea_alg=='S':
    #             toolbox.register("mate", tools.cxSimulatedBinaryBounded, low=BOUND_LOW, up=BOUND_UP, eta=20.0)
    #             toolbox.register("mutate", tools.mutPolynomialBounded, low=BOUND_LOW, up=BOUND_UP, eta=20.0, indpb=1.0/NDIM_DTLZ)
    #             toolbox.register("select", smscoin.selSMS)
    #             toolbox.register("selectSMSCOIN", smscoin.selSMSCOIN, world=my_world)
    #
    #         else:
    #             toolbox.register("mate", tools.cxSimulatedBinaryBounded, low=BOUND_LOW, up=BOUND_UP, eta=20.0)
    #             toolbox.register("mutate", tools.mutPolynomialBounded, low=BOUND_LOW, up=BOUND_UP, eta=20.0, indpb=1.0/NDIM_DTLZ)
    #             toolbox.register("select", tools.selNSGA2)
    #             #toolbox.register("select", tools.selSPEA2)
    #             toolbox.register("selectNSGACOIN", nsgacoin.selNSGA2COIN, world=my_world)
    #
    #     elif problem == 'M':
    #
    #
    #         def uniform(low, up, size=None):
    #             try:
    #                 return [random.uniform(a, b) for a, b in zip(low, up)]
    #             except TypeError:
    #                 return [random.uniform(a, b) for a, b in zip([low] * size, [up] * size)]
    #
    #         BOUND_LOW, BOUND_UP = 0.0, 1.0
    #
    #         creator.create("FitnessMin", base.Fitness, weights=(-1.0,-1.0,-1.0))
    #         creator.create("Individual", array.array, typecode='d', fitness=creator.FitnessMin)
    #
    #
    #         toolbox.register("attr_float", uniform, BOUND_LOW, BOUND_UP, NDIM_DTLZ)
    #         toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.attr_float)
    #         toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    #
    #         toolbox.register("evaluate", lambda ind: dtlz7(ind, 3))
    #
    #
    #
    #
    #         if moea_alg=='P':
    #             toolbox.register("select", tools.selSPEA2)
    #             toolbox.register("selectTournament", tools.selTournament, tournsize=2)
    #             # toolbox.register("mate", tools.cxSimulatedBinaryBounded, low=BOUND_LOW, up=BOUND_UP, eta=0.5)
    #             # toolbox.register("mutate", tools.mutPolynomialBounded, low=BOUND_LOW, up=BOUND_UP, eta=0.5, indpb=1.0)
    #             toolbox.register("mate", tools.cxSimulatedBinaryBounded, low=BOUND_LOW, up=BOUND_UP, eta=2)
    #             toolbox.register("mutate", tools.mutPolynomialBounded, low=BOUND_LOW, up=BOUND_UP, eta=2, indpb=1.0)
    #             toolbox.register("selectSPEACOIN", speacoin.selSPEA2COIN, world=my_world)
    #
    #         elif moea_alg=='S':
    #             toolbox.register("mate", tools.cxSimulatedBinaryBounded, low=BOUND_LOW, up=BOUND_UP, eta=20.0)
    #             toolbox.register("mutate", tools.mutPolynomialBounded, low=BOUND_LOW, up=BOUND_UP, eta=20.0, indpb=1.0/NDIM_DTLZ)
    #             toolbox.register("select", smscoin.selSMS)
    #             toolbox.register("selectSMSCOIN", smscoin.selSMSCOIN, world=my_world)
    #
    #         else:
    #             toolbox.register("mate", tools.cxSimulatedBinaryBounded, low=BOUND_LOW, up=BOUND_UP, eta=20.0)
    #             toolbox.register("mutate", tools.mutPolynomialBounded, low=BOUND_LOW, up=BOUND_UP, eta=20.0, indpb=1.0/NDIM_DTLZ)
    #             toolbox.register("select", tools.selNSGA2)
    #             #toolbox.register("select", tools.selSPEA2)
    #             toolbox.register("selectNSGACOIN", nsgacoin.selNSGA2COIN, world=my_world)
    #
    #     #parameters: mutation and number, generations, number of population
    #     self.CXPB  = CXPB
    #     self.MUTPB = MUTPB
    #     self.NGEN  = NGEN
    #     self.num_population = num_population
    #
    #
    #
    #
    #
    #
    #
    #
    #
    #
    #     self.stats = tools.Statistics(lambda ind: ind.fitness.values)
    #     self.logbook = tools.Logbook()


    #set individual
    def SetIndividual(self, gateways, units, max_units):

        man = []

        #create list of individual
        #gateways
        for i in gateways:
            man.append(i[0])
            man.append(i[1])
            man.append(i[2])
        #units
        aux = 0
        for i in units:
            man.append(i[0])
            man.append(i[1])
            man.append(i[2])
            aux=aux+1
        for i in range(aux,max_units):
            man.append(-1)
            man.append(-1)
            man.append(-1)

        return man


    #create population
    def SetPopulation(self):
        pop = toolbox.population(n=self.num_population)
        #pop.fitness.values = 3
        return pop

    #create population
    def SetPopulationFake(self, my_world):

        # Attribute generator
        toolbox.register("individual_gen_fake", my_world.CreateFullindividualFake)
        #toolbox.register("individual_gen", random.randint, 0, 1)


        # Structure initializers
        toolbox.register("individualfake", tools.initRepeat, creator.Individual, toolbox.individual_gen_fake, 1)
        toolbox.register("populationfake", tools.initRepeat, list, toolbox.individualfake)


        pop = toolbox.populationfake(n=self.num_population)
        #pop.fitness.values = 3
        return pop

    #create population
    def SetPopulationFakeBench(self, my_world, problem='P'):

        def uniform(size=None):
                return [1]*size


        # if problem != 'S':
        #     toolbox.register("attr_float_fake", uniform, NDIM)
        # elif problem == 'G':
        #     toolbox.register("attr_float_fake", uniform, NDIM_DTLZ)
        # else:
        #     toolbox.register("attr_float_fake", uniform, 10)

        if problem == 'S':
            toolbox.register("attr_float_fake", uniform, 10)
        elif problem == 'G' or problem == 'H' or problem == 'I'  or problem=='J' or problem=='L' or problem=='M' or problem == 'F':
            toolbox.register("attr_float_fake", uniform, NDIM_DTLZ)
        else:
            toolbox.register("attr_float_fake", uniform, NDIM)


        # Structure initializers
        toolbox.register("individualfake", tools.initIterate, creator.Individual, toolbox.attr_float_fake)
        toolbox.register("populationfake", tools.initRepeat, list, toolbox.individualfake)


        pop = toolbox.populationfake(n=self.num_population)
        #pop.fitness.values = 3

        return pop

    #get fitness of all individuals
    def GetFitness(self, pop):
        fitnesses = list(map(toolbox.evaluate, pop))
        #fitnesses = list(map(toolbox.evaluate, [[8, 8, 1, 19, 15, 1, 16, 11, 1, 14, 2, 1, 4, 15, 1, 9, 14, 1, 0, 8, 1, -1, -1, -1, -1, -1, -1, -1, -1, -1], [8, 8, 1, 19, 15, 1, 16, 11, 1, 14, 2, 1, 4, 15, 1, 9, 14, 1, 0, 8, 1, -1, -1, -1, -1, -1, -1, -1, -1, -1], [8, 8, 1, 19, 15, 1, 16, 11, 1, 14, 2, 1, 4, 15, 1, 9, 14, 1, 0, 8, 1, -1, -1, -1, -1, -1, -1, -1, -1, -1]]))
        return fitnesses

    def GetFitnessPAIR(self, pop):
        fitnesses = list(map(toolbox.evaluatePAIR, pop))

        return fitnesses


    #attach fitness to the individual
    def AttachFitness(self, pop, fitnesses):

        for ind, fit in zip(pop, fitnesses):
            ind.fitness.values = fit

    #cross and mutation
    def Selection(self, pop, my_world,rate=None):

        if my_world.FIX_strategy != 0:
            # Select the next generation individuals
            offspring = toolbox.selectTournament(pop, len(pop))
            # Clone the selected individuals
            offspring = list(map(toolbox.clone, offspring))

            # Apply crossover and mutation on the offspring
            for child1, child2 in zip(offspring[::2], offspring[1::2]): #get every single pair
                if random.random() < self.CXPB:
                    toolbox.mate(child1[0], child2[0])
                    del child1.fitness.values
                    del child2.fitness.values
            for mutant in offspring:
                if random.random() < self.MUTPB:
                    #print mutant
                    toolbox.mutate(mutant[0])
                    #print mutant
                    del mutant.fitness.values

            # Evaluate the individuals with an invalid fitness
            invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
            #fix individual
            # print "cross + mut: ", len(invalid_ind)
            # #check after cross and/or mutation
            # h=0
            # for k in range(0,len(invalid_ind)):
            #
            #     gates=my_world.GetGates(invalid_ind[k][0])
            #     units=my_world.GetUnits(invalid_ind[k][0])
            #     if my_world.CheckGateways(gates) == 0 or my_world.CheckUnits(units) == 0 or my_world.CheckLinks(gates, units) == 0:
            #         h += 1
            # print "error: ", h


            if my_world.FIX_on_off and invalid_ind:
                my_world.FIX_ind(invalid_ind)
            #print invalid_ind

            # print("  Cross or Mutation: %s" % len(invalid_ind))
            fitnesses = map(toolbox.evaluate, invalid_ind)

            for ind, fit in zip(invalid_ind, fitnesses):
                ind.fitness.values = fit


            #print("  Evaluated %i individuals" % len(invalid_ind))

            # The population is entirely replaced by the offspring
            #off = toolbox.select(pop + offspring, self.num_population)
            #pop[:] = off
            # Select the next generation population
            off = toolbox.select(pop + offspring, self.num_population)
            pop[:] = off

        #Well, I will keep the IF block. I know that many lines will be repeated bellow.
        #But the first block is ready and working. I will have to change these blocks in the
        #future ... so I will repeat it into the ELSE block. I am aware about the repetition of code
        else:
            still_inv = len(pop)
            spare_pop=[]
            CXPB= self.CXPB
            MUTPB=self.MUTPB
            while still_inv:
                # Select the next generation individuals
                offspring = toolbox.selectTournament(pop, still_inv)
                still_inv=0


                # Clone the selected individuals
                offspring = list(map(toolbox.clone, offspring))

                # Apply crossover and mutation on the offspring
                for child1, child2 in zip(offspring[::2], offspring[1::2]): #get every single pair
                    if random.random() < CXPB:
                        toolbox.mate(child1[0], child2[0])
                        del child1.fitness.values
                        del child2.fitness.values
                for mutant in offspring:
                    if random.random() < MUTPB:
                        #print mutant
                        toolbox.mutate(mutant[0])
                        #print mutant
                        del mutant.fitness.values

                #separate invalid individuals from invalid fitness
                invalid_ind = []
                valid_ind =[]
                for ind in offspring:
                    if not ind.fitness.valid:
                        #check for errors
                        gates=my_world.GetGates(ind[0])
                        units=my_world.GetUnits(ind[0])
                        if my_world.CheckGateways(gates) == 0 or my_world.CheckUnits(units) == 0 or my_world.CheckLinks(gates, units) == 0:
                            #error in individual ... let the selection get more
                            still_inv += 1

                        else:
                            #separate to get the fitness
                            invalid_ind.append(ind)
                    else:
                        valid_ind.append(ind)

                #fitness for the individuals without fitness and with RIGHT config
                fitnesses = map(toolbox.evaluate, invalid_ind)

                for ind, fit in zip(invalid_ind, fitnesses):
                    ind.fitness.values = fit

                # The population is entirely replaced by the offspring
                for ind in valid_ind:
                    spare_pop.append(ind)
                for ind in invalid_ind:
                    spare_pop.append(ind)

                #change the probability of cross and mutation
                CXPB=1
                MUTPB=0.5
                offspring=[]

            off = toolbox.select(pop + spare_pop, self.num_population)
            pop[:] = off



    #cross and mutation
    def NSGASelection(self, pop, my_world):

        if my_world.FIX_strategy != 0:
            # Vary the population
            offspring = tools.selTournamentDCD(pop, len(pop))
            offspring = [toolbox.clone(ind) for ind in offspring]


            # Apply crossover and mutation on the offspring
            for child1, child2 in zip(offspring[::2], offspring[1::2]): #get every single pair
                if random.random() < self.CXPB:
                    toolbox.mate(child1[0], child2[0])
                    del child1.fitness.values
                    del child2.fitness.values
            for mutant in offspring:
                if random.random() < self.MUTPB:
                    #print mutant
                    toolbox.mutate(mutant[0])
                    #print mutant
                    del mutant.fitness.values

            # Evaluate the individuals with an invalid fitness
            invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
            #fix individual
            # print "cross + mut: ", len(invalid_ind)
            # #check after cross and/or mutation
            # h=0
            # for k in range(0,len(invalid_ind)):
            #
            #     gates=my_world.GetGates(invalid_ind[k][0])
            #     units=my_world.GetUnits(invalid_ind[k][0])
            #     if my_world.CheckGateways(gates) == 0 or my_world.CheckUnits(units) == 0 or my_world.CheckLinks(gates, units) == 0:
            #         h += 1
            # print "error: ", h


            if my_world.FIX_on_off and invalid_ind:
                my_world.FIX_ind(invalid_ind)
            #print invalid_ind

            # print("  Cross or Mutation: %s" % len(invalid_ind))
            fitnesses = map(toolbox.evaluate, invalid_ind)

            for ind, fit in zip(invalid_ind, fitnesses):
                ind.fitness.values = fit


            #print("  Evaluated %i individuals" % len(invalid_ind))

            # The population is entirely replaced by the offspring
            off = toolbox.select(pop + offspring, self.num_population)
            pop[:] = off


        #Well, I will keep the IF block. I know that many lines will be repeated bellow.
        #But the first block is ready and working. I will have to change these blocks in the
        #future ... so I will repeat it into the ELSE block. I am aware about the repetition of code
        else:
            still_inv = len(pop)
            spare_pop=[]
            CXPB= self.CXPB
            MUTPB=self.MUTPB
            while still_inv:
                # Select the next generation individuals
                offspring = tools.selTournamentDCD(pop, still_inv)
                still_inv=0


                # Clone the selected individuals
                offspring = list(map(toolbox.clone, offspring))

                # Apply crossover and mutation on the offspring
                for child1, child2 in zip(offspring[::2], offspring[1::2]): #get every single pair
                    if random.random() < CXPB:
                        toolbox.mate(child1[0], child2[0])
                        del child1.fitness.values
                        del child2.fitness.values
                for mutant in offspring:
                    if random.random() < MUTPB:
                        #print mutant
                        toolbox.mutate(mutant[0])
                        #print mutant
                        del mutant.fitness.values

                #separate invalid individuals from invalid fitness
                invalid_ind = []
                valid_ind =[]
                for ind in offspring:
                    if not ind.fitness.valid:
                        #check for errors
                        gates=my_world.GetGates(ind[0])
                        units=my_world.GetUnits(ind[0])
                        if my_world.CheckGateways(gates) == 0 or my_world.CheckUnits(units) == 0 or my_world.CheckLinks(gates, units) == 0:
                            #error in individual ... let the selection get more
                            still_inv += 1

                        else:
                            #separate to get the fitness
                            invalid_ind.append(ind)
                    else:
                        valid_ind.append(ind)

                #fitness for the individuals without fitness and with RIGHT config
                fitnesses = map(toolbox.evaluate, invalid_ind)

                for ind, fit in zip(invalid_ind, fitnesses):
                    ind.fitness.values = fit

                # The population is entirely replaced by the offspring
                for ind in valid_ind:
                    spare_pop.append(ind)
                for ind in invalid_ind:
                    spare_pop.append(ind)

                #change the probability of cross and mutation
                CXPB=1
                MUTPB=0.5
                offspring=[]

            off = toolbox.select(pop + spare_pop, self.num_population)
            pop[:] = off



    #cross and mutation with COIN selection in NSGA
    def NSGASelectionCOIN(self, pop, my_world, ga,  type_selection, exp_type, best_gmm=None, kmm=None, partial=None, partial_rejected=None, problem='P'):

        if my_world.FIX_strategy != 0:

            # fitnesses_kk2 = self.GetFitness(pop)
            # fitnesses_kk2.sort(key=itemgetter(1), reverse=True)

            # Vary the population
            #here I get the individuals by non-domination AND COIN distance value (that is inside the same attribute Crowding Distance)
            offspring = nsgacoin.selTournamentCOINd(pop, len(pop))
            ###offspring = tools.selTournamentDCD(pop, len(pop))

            offspring = [toolbox.clone(ind) for ind in offspring]
            #
            # fitnesses_kk2 = self.GetFitness(offspring)
            # fitnesses_kk2.sort(key=itemgetter(1), reverse=True)


            # #TEMP only for my check
            # individuo = []
            # herd = []
            # for i in pop:
            #     if i not in individuo:
            #         individuo.append(i)
            # for i in offspring:
            #     if i not in herd:
            #         herd.append(i)
            #
            #

            if problem=='A':
                # Apply crossover and mutation on the offspring
                for child1, child2 in zip(offspring[::2], offspring[1::2]): #get every single pair
                    if random.random() < self.CXPB:
                        toolbox.mate(child1[0], child2[0])
                        del child1.fitness.values
                        del child2.fitness.values
                for mutant in offspring:
                    if random.random() < self.MUTPB:
                        #print mutant
                        toolbox.mutate(mutant[0])
                        #print mutant
                        del mutant.fitness.values


            else:
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

            if problem=='A':
                if my_world.FIX_on_off and invalid_ind:
                    my_world.FIX_ind(invalid_ind)

            # print("  Cross or Mutation: %s" % len(invalid_ind))
            fitnesses = map(toolbox.evaluate, invalid_ind)

            for ind, fit in zip(invalid_ind, fitnesses):
                ind.fitness.values = fit


            #print("  Evaluated %i individuals" % len(invalid_ind))

            # fitnesses_kk2 = self.GetFitness(offspring)
            # fitnesses_kk2.sort(key=itemgetter(1), reverse=True)


            # The population is entirely replaced by the offspring
            off, par, rej = toolbox.selectNSGACOIN(pop + offspring , self.num_population, type_selection=type_selection,
                                         exp_type=exp_type, best_gmm=best_gmm, kmm=kmm, ga=ga)
            pop[:] = off
            partial[:] = par
            partial_rejected[:] = rej

        else:

            print "not implemented!"
            quit()

    #cross and mutation with COIN selection in NSGA
    def SPEA2SelectionCOIN(self, pop, archive, my_world, ga,  type_selection, exp_type, best_gmm=None, kmm=None, partial=None, partial_rejected=None, problem='P', firstloop_continue=7):

        if my_world.FIX_strategy != 0:



            # Step 1 Environmental selection -- define external archive

            if firstloop_continue==0: #if first iteration in the continue bench procedure....
                archive_b  = toolbox.selectSPEACOIN(pop + archive, k=len(pop), type_selection=type_selection,
                                                exp_type=exp_type, best_gmm=best_gmm, kmm=kmm, ga=ga, nadir=NADIR_SPEA)

            else:
                archive_b  = toolbox.selectSPEACOIN(pop + archive, k=(len(pop)//2), type_selection=type_selection,
                                                exp_type=exp_type, best_gmm=best_gmm, kmm=kmm, ga=ga, nadir=NADIR_SPEA)



            #fitnesses = self.GetFitness(archive_b)
            #self.AttachFitness(archive_b,fitnesses)
            fitnesses_kk2 = self.GetFitness(archive_b)
            fitnesses_kk2.sort(key=itemgetter(1), reverse=True)

            #$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
            # Vary the population
            #step 2
            if exp_type == 'C' or exp_type == 'E':
                means = my_world.centroids
            elif exp_type == 'B' or exp_type == 'D':
                means = my_world.means
            else:
                print "Error!! wrong exp.type."
                quit()
            if firstloop_continue==0: #if first iteration in the continue bench procedure....
                #offspring = toolbox.selectTournament(archive_b, (len(pop)*2))
                offspring = speacoin.selTournamentCOINd(archive_b, (len(pop)*2),means)

            else:
                #offspring = toolbox.selectTournamentCOIN(archive_b, len(pop))
                offspring = speacoin.selTournamentCOINd(archive_b, len(pop),means)

            fitnesses_kk2 = self.GetFitness(archive_b)
            fitnesses_kk2.sort(key=itemgetter(1), reverse=True)


            #offspring = toolbox.selectTournament(pop, len(pop))
            offspring = [toolbox.clone(ind) for ind in offspring]

            # #TEMP only for my check
            # individuo = []
            # herd = []
            # for i in pop:
            #     if i not in individuo:
            #         individuo.append(i)
            # for i in offspring:
            #     if i not in herd:
            #         herd.append(i)
            #
            #

            if problem=='A':
                # Apply crossover and mutation on the offspring
                for child1, child2 in zip(offspring[::2], offspring[1::2]): #get every single pair
                    if random.random() < self.CXPB:
                        toolbox.mate(child1[0], child2[0])
                        del child1.fitness.values
                        del child2.fitness.values
                for mutant in offspring:
                    if random.random() < self.MUTPB:
                        #print mutant
                        toolbox.mutate(mutant[0])
                        #print mutant
                        del mutant.fitness.values


            else:
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

            if problem=='A':
                if my_world.FIX_on_off and invalid_ind:
                    my_world.FIX_ind(invalid_ind)

            # print("  Cross or Mutation: %s" % len(invalid_ind))
            fitnesses = map(toolbox.evaluate, invalid_ind)

            for ind, fit in zip(invalid_ind, fitnesses):
                ind.fitness.values = fit


            #print("  Evaluated %i individuals" % len(invalid_ind))


            #step 4
            pop[:] = offspring
            archive[:] = archive_b
            fitnesses_kk2 = self.GetFitness(offspring)
            fitnesses_kk2.sort(key=itemgetter(1), reverse=True)
            fitnesses_kk = self.GetFitness(archive_b)
            fitnesses_kk.sort(key=itemgetter(1), reverse=True)
            s=1



        else:

            print "not implemented!"
            quit()






    #cross and mutation with COIN selection in SMS-EMOA
    def SMSSelection_COIN(self, pop, my_world, ga,  type_selection, exp_type, best_gmm=None, kmm=None, partial=None, partial_rejected=None, problem='P'):

        if my_world.FIX_strategy != 0:

            # fitnesses_kk2 = self.GetFitness(pop)
            # fitnesses_kk2.sort(key=itemgetter(1), reverse=True)

            # Vary the population
            #here I get the individuals by non-domination AND COIN distance value (that is inside the same attribute Crowding Distance)
            offspring = smscoin.selTournamentHYPER(pop, len(pop))
            ###offspring = tools.selTournamentDCD(pop, len(pop))

            offspring = [toolbox.clone(ind) for ind in offspring]
            #
            # fitnesses_kk2 = self.GetFitness(offspring)
            # fitnesses_kk2.sort(key=itemgetter(1), reverse=True)


            # #TEMP only for my check
            # individuo = []
            # herd = []
            # for i in pop:
            #     if i not in individuo:
            #         individuo.append(i)
            # for i in offspring:
            #     if i not in herd:
            #         herd.append(i)
            #
            #

            if problem=='A':
                # Apply crossover and mutation on the offspring
                for child1, child2 in zip(offspring[::2], offspring[1::2]): #get every single pair
                    if random.random() < self.CXPB:
                        toolbox.mate(child1[0], child2[0])
                        del child1.fitness.values
                        del child2.fitness.values
                for mutant in offspring:
                    if random.random() < self.MUTPB:
                        #print mutant
                        toolbox.mutate(mutant[0])
                        #print mutant
                        del mutant.fitness.values


            else:
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

            if problem=='A':
                if my_world.FIX_on_off and invalid_ind:
                    my_world.FIX_ind(invalid_ind)

            # print("  Cross or Mutation: %s" % len(invalid_ind))
            fitnesses = map(toolbox.evaluate, invalid_ind)

            for ind, fit in zip(invalid_ind, fitnesses):
                ind.fitness.values = fit


            #print("  Evaluated %i individuals" % len(invalid_ind))

            # fitnesses_kk2 = self.GetFitness(offspring)
            # fitnesses_kk2.sort(key=itemgetter(1), reverse=True)


            # The population is entirely replaced by the offspring
            off, par, rej = toolbox.selectSMSCOIN(pop + offspring , self.num_population, type_selection=type_selection, exp_type=exp_type, best_gmm=best_gmm, kmm=kmm, ga=ga)
            pop[:] = off
            partial[:] = par
            partial_rejected[:] = rej

        else:

            print "not implemented!"
            quit()

    #cross and mutation
    def NSGASelectionPAIR(self, pop,  my_world):


        #################################################
        # METHOD 1: use INC only at the final SELECTION #
        #################################################

        #receives the real population and the real fitness
        #do everything (cross, mut) with the real pop
        if my_world.FIX_strategy != 0:
            # Vary the population
            offspring = tools.selTournamentDCD(pop, len(pop))
            offspring = [toolbox.clone(ind) for ind in offspring]


            # Apply crossover and mutation on the offspring
            for child1, child2 in zip(offspring[::2], offspring[1::2]): #get every single pair
                if random.random() < self.CXPB:
                    toolbox.mate(child1[0], child2[0])
                    del child1.fitness.values
                    del child2.fitness.values
            for mutant in offspring:
                if random.random() < self.MUTPB:
                    #print mutant
                    toolbox.mutate(mutant[0])
                    #print mutant
                    del mutant.fitness.values

            # Evaluate the individuals with an invalid fitness
            invalid_ind = [ind for ind in offspring if not ind.fitness.valid]


            if my_world.FIX_on_off and invalid_ind:
                my_world.FIX_ind(invalid_ind)
            #print invalid_ind

            #ok! HERE I have the offspring almost read (except the invalid individuals)

            #now I calculate the NEW fitness to pop and offspring, so I will select based on this incremental fitness - which is close to the mean
            last_pop=[]

            INC_ind = list([ind for ind in offspring])
            last_pop = list([ind for ind in pop])

            fitnessesINC = map(toolbox.evaluatePAIR, INC_ind)
            fitnessesINCpop = map(toolbox.evaluatePAIR, last_pop)

            for ind, fit in zip(INC_ind, fitnessesINC):
                ind.fitness.values = fit
            for ind, fit in zip(last_pop, fitnessesINCpop):
                ind.fitness.values = fit


            print("  Evaluated %i individuals" % len(invalid_ind))

            # The population is entirely replaced by the offspring
            off = toolbox.select(last_pop + INC_ind, self.num_population)


            #before return pop with individuals from off, I calculate the right fitness back
            fitnesses = map(toolbox.evaluate, off)

            for ind, fit in zip(off, fitnesses):
                ind.fitness.values = fit


            pop[:] = off



            # # print("  Cross or Mutation: %s" % len(invalid_ind))
            # fitnesses = map(toolbox.evaluatePAIR, invalid_ind)
            #
            # for ind, fit in zip(invalid_ind, fitnesses):
            #     ind.fitness.values = fit
            #
            # off = toolbox.select(pop + offspring, self.num_population)
            # pop[:] = off

        ####################################################
        # METHOD 2: use INC before Cross and Mut SELECTION #
        ####################################################
        #
        # #receives the real population and the real fitness
        # #do everything (cross, mut) with the real pop
        # if my_world.FIX_strategy != 0:
        #     # Vary the population
        #     offspring = tools.selTournamentDCD(pop, len(pop))
        #     offspring = [toolbox.clone(ind) for ind in offspring]
        #
        #
        #     # Apply crossover and mutation on the offspring
        #     for child1, child2 in zip(offspring[::2], offspring[1::2]): #get every single pair
        #         if random.random() < self.CXPB:
        #             toolbox.mate(child1[0], child2[0])
        #             del child1.fitness.values
        #             del child2.fitness.values
        #     for mutant in offspring:
        #         if random.random() < self.MUTPB:
        #             #print mutant
        #             toolbox.mutate(mutant[0])
        #             #print mutant
        #             del mutant.fitness.values
        #
        #     # Evaluate the individuals with an invalid fitness
        #     invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        #
        #
        #     if my_world.FIX_on_off and invalid_ind:
        #         my_world.FIX_ind(invalid_ind)
        #     #print invalid_ind
        #
        #     #ok! HERE I have the offspring almost read (except the invalid individuals)
        #
        #     #now I calculate the NEW fitness to pop and offspring, so I will select based on this incremental fitness - which is close to the mean
        #     #last_pop=[]
        #
        #     INC_ind = list([ind for ind in offspring])
        #     #last_pop = list([ind for ind in pop])
        #
        #     fitnessesINC = map(toolbox.evaluatePAIR, INC_ind)
        #     #fitnessesINCpop = map(toolbox.evaluatePAIR, last_pop)
        #
        #     for ind, fit in zip(INC_ind, fitnessesINC):
        #         ind.fitness.values = fit
        #     #for ind, fit in zip(last_pop, fitnessesINCpop):
        #     #    ind.fitness.values = fit
        #
        #
        #     #print("  Evaluated %i individuals" % len(invalid_ind))
        #
        #     # The population is entirely replaced by the offspring
        #     off = toolbox.select(pop + offspring, self.num_population)
        #
        #
        #     #before return pop with individuals from off, I calculate the right fitness back
        #     fitnesses = map(toolbox.evaluate, off)
        #
        #     for ind, fit in zip(off, fitnesses):
        #         ind.fitness.values = fit
        #
        #
        #     pop[:] = off
        #
        #

        #Well, I will keep the IF block. I know that many lines will be repeated bellow.
        #But the first block is ready and working. I will have to change these blocks in the
        #future ... so I will repeat it into the ELSE block. I am aware about the repetition of code
        else:

            print "not implemented!"
            quit()


    #cross and mutation
    def NSGASelection_Pure(self, pop, partialn=None, partial_rejn=None):

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

        #TIRAR
        #off, par, rej = toolbox.selectNSGAPURE(pop + offspring, self.num_population)
        #pop[:] = off
        #partialn[:] = par
        #partial_rejn[:] = rej




    #SMS-EMOA
    def SMSSelection_Pure(self, pop, partialn=None, partial_rejn=None):

        #$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
        # Vary the population
        offspring = smscoin.selTournamentHYPER(pop, len(pop))


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

        #TIRAR
        #off, par, rej = toolbox.selectNSGAPURE(pop + offspring, self.num_population)
        #pop[:] = off
        #partialn[:] = par
        #partial_rejn[:] = rej


    #cross and mutation
    def SPEA2Selection_Pure(self, pop, archive, ga, partialn=None, partial_rejn=None, firstloop_continue=7):

        #$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
        # Step 1 Environmental selection -- define external archive
        if firstloop_continue==0: #if first iteration in the continue bench procedure....
           archive_b  = toolbox.select(pop + archive, k=len(pop))
        else:
           archive_b  = toolbox.select(pop + archive, k=(len(pop)//2))

        fitnesses = self.GetFitness(archive_b)
        self.AttachFitness(archive_b,fitnesses)

        #$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
        # Vary the population
        #step 2
        if firstloop_continue==0: #if first iteration in the continue bench procedure....
            offspring = toolbox.selectTournament(archive_b, len(pop)*2)
        else:
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


        #TIRAR
        #off, par, rej = toolbox.selectNSGAPURE(pop + offspring, self.num_population)
        #pop[:] = off
        #partialn[:] = par
        #partial_rejn[:] = rej


    #sort non-dominated
    def MysortFastND(individuals, n, first_front_only=False):
        """Sort the first *n* *individuals* according the the fast non-dominated
        sorting algorithm.
        """
        N = len(individuals)
        pareto_fronts = []

        if n == 0:
            return pareto_fronts

        pareto_fronts.append([])
        pareto_sorted = 0
        dominating_inds = [0] * N
        dominated_inds = [list() for i in xrange(N)]

        # Rank first Pareto front
        for i in xrange(N):
            for j in xrange(i+1, N):
                if individuals[j].fitness.isDominated(individuals[i].fitness):
                    dominating_inds[j] += 1
                    dominated_inds[i].append(j)
                elif individuals[i].fitness.isDominated(individuals[j].fitness):
                    dominating_inds[i] += 1
                    dominated_inds[j].append(i)
            if dominating_inds[i] == 0:
                pareto_fronts[-1].append(i)
                pareto_sorted += 1

        if not first_front_only:
        # Rank the next front until all individuals are sorted or the given
        # number of individual are sorted
            N = min(N, n)
            while pareto_sorted < N:
                pareto_fronts.append([])
                for indice_p in pareto_fronts[-2]:
                    for indice_d in dominated_inds[indice_p]:
                        dominating_inds[indice_d] -= 1
                        if dominating_inds[indice_d] == 0:
                            pareto_fronts[-1].append(indice_d)
                            pareto_sorted += 1

        return [[individuals[index] for index in front] for front in pareto_fronts]

    #get best Pareto front (hypervolume)
    def UpdateBestParetoFront_hypervolume(self, best, local, refpoint):

        #transform best and local fitness to a list of fitness
        best_fit=[]
        local_fit=[]
        for i in best:
            best_fit.append(i.fitness.values)
        for i in local:
            local_fit.append(i.fitness.values)


        #evaluate the hypervolume
        hyper=HyperVolume(refpoint)
        A = hyper.compute(best_fit)
        B = hyper.compute(local_fit)

        if B > A:
            return local

        return best

    #get hypervolume
    def Hypervolume(self, front, refpoint):

        #transform front fitness to a list of fitness
        local_fit=[]
        for i in front:
            local_fit.append((i.fitness.values[0],i.fitness.values[1]+refpoint[1]))


        #evaluate the hypervolume
        hyper=HyperVolume(refpoint)
        aux = hyper.compute(local_fit)
        return aux/(refpoint[0]*refpoint[1])

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

    #get best Pareto front (cover)
    def UpdateBestParetoFront_cover(self, best, local):

        swp_coverpareto=[]
        swp_coverpareto=tools.ParetoFront()

        #put everything into Paretofront
        swp_coverpareto.update(best)
        swp_coverpareto.update(local)

        #loop
        n_best=0
        n_local=0
        for i in swp_coverpareto:

            #if not in best AND local at the same time
            if not (i in best and i in local):

                if i in best:
                    n_best += 1
                else:
                    n_local += 1

        #cover
        if (float(n_local)/(n_local+n_best))> .5:
            return local
        else:
            return best

    #get cover
    def Cover(self, optimalpareto, best_pareto_hypervolume, best_pareto_cover):

        #loop
        n_hyper=0
        n_cover=0
        for i in optimalpareto:

            if i in best_pareto_hypervolume:
                n_hyper += 1
            if i in best_pareto_cover:
                n_cover += 1


        return -1, (float(n_hyper)/len(optimalpareto)), (float(n_cover)/len(optimalpareto))

    def Cover2sets(self, optimalpareto, best_pareto_hypervolume):

        #loop
        n_hyper=0

        for i in optimalpareto:

            if i in best_pareto_hypervolume:
                n_hyper += 1



        return (float(n_hyper)/len(optimalpareto))

    #start stats
    def Stats(self):

        self.stats.register("avg", np.mean, axis=0)
        self.stats.register("std", np.std, axis=0)
        self.stats.register("min", np.min, axis=0)
        self.stats.register("max", np.max, axis=0)

        self.logbook.header = "gen", "std", "min", "avg", "max"

    #execute stat
    def ExeStats(self, pop, gen):
        record = self.stats.compile(pop)
        self.logbook.record(gen=gen, **record)


    #assign Crowding Distance
    def SetCrowdingDistance(self, pop):

        #$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
        # Select the next generation population
        off = toolbox.select(pop, self.num_population)

        # The population is entirely replaced by the offspring
        pop[:] = off


    #set hypervolume values for SMS-EMOA
    def SetHyperValue(self, pop):

        #$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
        # Select the next generation population
        off = toolbox.select(pop, self.num_population)

        # The population is entirely replaced by the offspring
        pop[:] = off

    #set hypervolume values for CI-SMS-EMOA
    def SetHyperValueCOIN(self, pop):

        #$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
        # Select the next generation population
        off = toolbox.selectSMSCOIN(pop, self.num_population)

        # The population is entirely replaced by the offspring
        pop[:] = off[0]

    #assign Crowding Distance
    def SetCOINDistance(self, pop, partial, partial_rej):

        #$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
        # Select the next generation population
        off, partial_ind, rej = toolbox.selectNSGACOIN(pop, self.num_population)

        # The population is entirely replaced by the offspring
        pop[:] = off
        partial[:] = partial_ind
        partial_rej[:] = rej

