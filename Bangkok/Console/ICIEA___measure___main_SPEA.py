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


#parameters: cross, mutation, number of generations, number of population
CXPB = 0.3
MUTPB= 0.1
#20x20
# NGEN = 300
# NPOP = 200
#50x50
# NGEN = 901
# NPOP = 700

lst_hyper=[]

NGEN = 300
NPOP = 200
NTEST = 40


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



    #declare Genetic Algorithm for the problem
    ga = GA(my_world,CXPB, MUTPB, NGEN, NPOP)


    #loop for measure
    #optimalpareto=[]
    #optimalpareto=tools.ParetoFront()
    #best_pareto_cover = []
    #best_pareto_hypervolume = []
    sum_hypervolume = 0
    #sumHYPER_bestcover = 0
    #sumCOVER_besthypervolume = 0
    #sumCOVER_bestcover = 0

    ga.Stats()


    for loop in range (0, NTEST):

        print "  loop: ", format(loop+1, '03d')
        print "  "


        ####################
        # first generation #
        ####################
        #set Population
        pareto=[]
        pareto=tools.ParetoFront()
        pop=ga.SetPopulation()

        # Evaluate the entire population
        fitnesses = ga.GetFitness(pop)
        ga.AttachFitness(pop,fitnesses)
        pareto.update(pop)
        print "  generation: ",format(1, '03d')

        #assign the crowding distance to the individuals
        #!!!!!!!!
        #ga.SetCrowdingDistance(pop)


        #STATS############################################################
        ga.ExeStats(pop, 0)
        #print(ga.logbook.stream)
        ##################################################################

        for i in range(1,NGEN):

            # Select the next generation individuals
            #!!!!!!!!!!!!!!!!!!!!!!!!!
            ga.Selection(pop, my_world)
            #ga.NSGASelection(pop, my_world)
            pareto.update(pop)

            #run stats
            ga.ExeStats(pop, i)
            #print(ga.logbook.stream)

            #print generation
            print "  generation: ", format(i+1, '03d')

        #local Pareto optimal front
        #local_Paretofront = list(pareto)
        #optimal pareto front
        #optimalpareto.update(local_Paretofront)
        #update best pareto front with hypervolume
        #best_pareto_hypervolume = ga.UpdateBestParetoFront_hypervolume(best_pareto_hypervolume, local_Paretofront, list(refpoint))
        #update best pareto front with cover
        #best_pareto_cover = ga.UpdateBestParetoFront_cover(best_pareto_cover, local_Paretofront)

        #get average of the bests
        #get average of the bests
        a=ga.Hypervolume(pareto, refpoint)
        lst_hyper.append(a)
        sum_hypervolume += a
        #sumHYPER_bestcover += ga.Hypervolume(best_pareto_cover, refpoint)
        # sum_aux=ga.Cover(optimalpareto, best_pareto_hypervolume, best_pareto_cover)
        # sumCOVER_besthypervolume += sum_aux[1]
        # sumCOVER_bestcover += sum_aux[2]


        print(ga.logbook.stream)

        print sum_hypervolume/(loop+1)

        #clean stats
    #end loop


    print "\n*** Final ***\n"
    #print "optima Pareto front   : ", ga.Hypervolume(optimalpareto, refpoint)
    #print "best hypervolume front: ", ga.Hypervolume(best_pareto_hypervolume, refpoint)
    #print "best cover front      : ", ga.Hypervolume(best_pareto_cover, refpoint)
    print "average hypervolume: ", sum_hypervolume /NTEST
    #print "average best cover front       : ", sumHYPER_bestcover/NTEST

    #print "\ncover"
    #aux=ga.Cover(optimalpareto, best_pareto_hypervolume, best_pareto_cover)
    #print "optima Pareto front   : X"
    #print "best hypervolume front: ", aux[1]
    #print "best cover front      : ", aux[2]
    # print "average best hypervolume front : ", sumCOVER_besthypervolume/NTEST
    # print "average best cover front       : ", sumCOVER_bestcover/NTEST
    print "\n*************\n\n\n"
    print(lst_hyper)

    #save solution to file
    #my_world.WriteFileSolution(optimalpareto, 0) #0=Pareto; 1=Best Hypercube; 2=Best Cover
    #solution=my_world.GetFileSolution(0) #0=Pareto; 1=Best Hypercube; 2=Best Cover

    #my_world.WriteFileSolution(best_pareto_hypervolume, 1) #0=Pareto; 1=Best Hypercube; 2=Best Cover
    #solution=my_world.GetFileSolution(1) #0=Pareto; 1=Best Hypercube; 2=Best Cover

    #my_world.WriteFileSolution(best_pareto_cover, 2) #0=Pareto; 1=Best Hypercube; 2=Best Cover
    #solution=my_world.GetFileSolution(2) #0=Pareto; 1=Best Hypercube; 2=Best Cover


    #print all non-dominated solutions WORLD
    #my_world.PrintSolutions(optimalpareto)

    print my_world.author




