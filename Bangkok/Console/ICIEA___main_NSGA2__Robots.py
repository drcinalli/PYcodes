'''
Created on 17/08/2014

@author: quatrosem
'''
from world import World
from emoa import GA
import random
from world import World
from deap import base
from deap import creator
from deap import tools
import numpy as np
import matplotlib.pylab as plt
from deap.benchmarks.tools import diversity, convergence






#parameters: cross, mutation, number of generations, number of population
CXPB = 0.3
MUTPB= 0.1
#20x20
NGEN = 100
NPOP = 200
#50x50
#NGEN = 50
#NPOP = 600



if __name__=='__main__':


    #World domain

    #Cenario 1: number of Area must be greater than or equal the number of Units
    # my_world = World(50,50)
    # my_world.MaxArea(7)
    # my_world.MaxUnits(6)
    # my_world.Production(23,35)
    # my_world.Costs(5,13,17)
    # refpoint=my_world.RefPointHypervolume()
    # my_world.CreateArea(3,3,10)
    # my_world.CreateArea(18,15,5)
    # my_world.CreateArea(28,1,8)
    # my_world.CreateArea(22,6,3)
    # my_world.CreateArea(35,33,14)
    # my_world.CreateArea(8,27,5)
    # my_world.CreateArea(12,39,7)

    # #Cenario 2
    my_world = World(20,20)
    my_world.MaxArea(6)
    my_world.MaxUnits(6)
    my_world.Production(23,35)
    my_world.Costs(5,13,17)
    refpoint=my_world.RefPointHypervolume()
    my_world.CreateArea(3,3,7)
    my_world.CreateArea(18,15,2)
    my_world.CreateArea(12,8,5)
    my_world.CreateArea(13,1,3)
    my_world.CreateArea(2,15,4)
    my_world.CreateArea(7,12,3)

    #Cenario 3: big
    # my_world = World(200,200)
    # my_world.MaxArea(14)
    # my_world.MaxUnits(12)
    # my_world.Production(23,35)
    # my_world.Costs(5,13,17)
    # refpoint=my_world.RefPointHypervolume()
    # my_world.CreateArea(3,3,10)
    # my_world.CreateArea(18,15,5)
    # my_world.CreateArea(28,1,8)
    # my_world.CreateArea(22,6,3)
    # my_world.CreateArea(35,33,14)
    # my_world.CreateArea(8,27,5)
    # my_world.CreateArea(12,39,7)
    #
    # my_world.CreateArea(103,103,10)
    # my_world.CreateArea(118,115,5)
    # my_world.CreateArea(128,101,8)
    # my_world.CreateArea(122,106,3)
    # my_world.CreateArea(135,133,14)
    # my_world.CreateArea(108,127,5)
    # my_world.CreateArea(112,139,7)

    ########################
    # create WORLD details #
    ########################
    #create gateways
    #gateways = my_world.CreateGateways()
    #print gateways
    #create units
    #units = my_world.CreateUnits()
    #print units
    #create links
    #my_world.CreateLinks(gateways,units)
    #print gateways


    #declare Genetic Algorithm for the problem
    ga = GA(my_world,CXPB, MUTPB, NGEN, NPOP)

    #define statistics for the MOEA
    ga.Stats()

    ####################
    # first generation #
    ####################
    #set Population
    pareto=tools.ParetoFront()
    pop=ga.SetPopulation()

    # Evaluate the entire population
    fitnesses = ga.GetFitness(pop)
    ga.AttachFitness(pop,fitnesses)
    pareto.update(pop)
    print "  generation: ",format(1, '03d')


    #assign the crowding distance to the individuals
    ga.SetCrowdingDistance(pop)


    #STATS############################################################
    ga.ExeStats(pop, 0)
    #print(ga.logbook.stream)
    ##################################################################



    #################################################################
    #################
    #print EVOLUTION#
    #################
    pointsX =[]
    pointsY=[]
    for j in pop:
        pointsX.append(j.fitness.values[0])
        pointsY.append(j.fitness.values[1])
    plt.scatter(pointsX,pointsY, c='r')
    plt.pause(0.01)
    plt.plot()
    #################################################################


    for i in range(1,NGEN):

        # Select the next generation individuals
        #ga.Selection(pop, my_world)
        ga.NSGASelection(pop, my_world)
        pareto.update(pop)

        #if i==50:
        #    my_world.PrintPop(pop)

        ga.ExeStats(pop, i+1)
        #print(ga.logbook.stream)
        #################################################################
        #################
        #print EVOLUTION#
        #################
        pointsX =[]
        pointsY=[]
        plt.xlabel('$Cost')
        plt.ylabel('#Production')
        for j in pop:
            pointsX.append(j.fitness.values[0])
            pointsY.append(j.fitness.values[1])
        plt.scatter(pointsX,pointsY, c='r')
        paretoX =[]
        paretoY=[]
        for j in pareto:
            paretoX.append(j.fitness.values[0])
            paretoY.append(j.fitness.values[1])
        plt.scatter(paretoX,paretoY, c='b')
        plt.plot(paretoX,paretoY, c='b')
        plt.pause(0.01)
        plt.clf()
        #################################################################
        #print generation
        print "  generation: ", format(i+1, '03d')




    #print all non-dominated solutions WORLD
    print "\nnumber of points: ",len(pareto)
    print "\n"
    print(ga.logbook.stream)
    t=[]
    for i in pareto:
        t.append(i.fitness.values)
    print t
    my_world.PrintPop(pareto)
    print pareto
    my_world.PrintSolutions(pareto)





    print my_world.author




