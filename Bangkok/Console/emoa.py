'''
Created on 21/08/2014

@author: quatrosem
'''
from __future__ import division
import random
from deap import base
from deap import creator
from deap import tools
import collections
import numpy as np
from hv import HyperVolume

toolbox = base.Toolbox()

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


class GA:
    #class variables
    description = "This is the Genetic Algorithm class of the problem"
    author = "Daniel Cinalli"

    def __init__(self, my_world, CXPB=None, MUTPB=None, NGEN = None,  num_population=None):

        #define fitness of the individual
        creator.create("FitnessMin", base.Fitness, weights=(-1.0,-1.0))
        creator.create("Individual", list, fitness=creator.FitnessMin)


        # Attribute generator
        toolbox.register("individual_gen", my_world.CreateFullindividual)
        #toolbox.register("individual_gen", random.randint, 0, 1)


        # Structure initializers
        toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.individual_gen, 1)
        toolbox.register("population", tools.initRepeat, list, toolbox.individual)

        # Operator registering
        toolbox.register("evaluate", f1Cost, world=my_world)

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

        #toolbox.register("mutate", tools.mutFlipBit, indpb=MUTPB) #gives 0 or 1 (not good)
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

        toolbox.register("selectTournament", tools.selTournament, tournsize=2)

        #parameters: mutation and number, generations, number of population
        self.CXPB  = CXPB
        self.MUTPB = MUTPB
        self.NGEN  = NGEN
        self.num_population = num_population


        self.stats = tools.Statistics(lambda ind: ind.fitness.values)
        self.logbook = tools.Logbook()


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
