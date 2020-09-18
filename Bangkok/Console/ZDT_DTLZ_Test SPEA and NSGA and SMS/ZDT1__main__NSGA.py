'''
Created on 17/08/2014

@author: quatrosem
'''
from emoa import GA
from deap import base
from deap import creator
from deap import tools
import array
import random
import json
import numpy
from deap.benchmarks.tools import diversity, convergence







if __name__=='__main__':

    optimal_front = json.load(open("zdt1_front.json"))
    # Use 500 of the 1000 points in the json file
    #optimal_front = sorted(optimal_front[i] for i in range(0, len(optimal_front), 2))
    #optimal_front = json.load(open("zdt4_front.json"))
    #optimal_front = json.load(open("zdt6_front.json"))

    NGEN = 100
    MU = 0.1
    CXPB = 0.3

    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", numpy.mean, axis=0)
    stats.register("std", numpy.std, axis=0)
    stats.register("min", numpy.min, axis=0)
    stats.register("max", numpy.max, axis=0)

    logbook = tools.Logbook()
    logbook.header = "gen", "evals", "std", "min", "avg", "max"

    #declare Genetic Algorithm for the problem
    ga = GA(CXPB=CXPB, MUTPB=MU, NGEN = NGEN,  num_population=100)
    ga.ZDT1_init()




    ####################
    # first generation #
    ####################
    #set Population
    pareto=tools.ParetoFront()
    pop=ga.SetPopulation(ga.num_population)





    # Evaluate the entire population
    fitnesses = ga.GetFitness(pop)
    ga.AttachFitness(pop,fitnesses)
    pareto.update(pop)
    print "  generation: ",format(1, '03d')

    # This is just to assign the crowding distance to the individuals
    # no actual selection is done
    #$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
    ga.ACD(pop)

    record = stats.compile(pop)
    #$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
    #logbook.record(gen=0, evals=len(invalid_ind), **record)
    logbook.record(gen=0, evals=len(fitnesses), **record)
    print(logbook.stream)


    for i in range(1,ga.NGEN):

        # Select the next generation individuals
        ga.NSGA_Selection(pop)
        pareto.update(pop)

        record = stats.compile(pop)
        logbook.record(gen=i+1, evals=len(pop), **record)
        print(logbook.stream)



    pop.sort(key=lambda x: x.fitness.values)

    print(stats)
    print("Convergence: ", convergence(pop, optimal_front))
    print("Diversity: ", diversity(pop, optimal_front[0], optimal_front[-1]))

    import matplotlib.pyplot as plt
    import numpy

    front = numpy.array([ind.fitness.values for ind in pop])
    optimal_front = numpy.array(optimal_front)
    plt.scatter(optimal_front[:,0], optimal_front[:,1], c="r")
    plt.scatter(front[:,0], front[:,1], c="b")
    plt.axis("tight")
    plt.show()



