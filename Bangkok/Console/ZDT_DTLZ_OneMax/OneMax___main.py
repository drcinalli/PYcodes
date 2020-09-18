'''
Created on 17/08/2014

@author: quatrosem
'''
from emoa import GA
from deap import base
from deap import creator
from deap import tools







if __name__=='__main__':


    #declare Genetic Algorithm for the problem
    ga = GA(0.3, MUTPB=0.05, NGEN = 70,  num_population=300)
    ga.OneMax_init()




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


    for i in range(1,ga.NGEN):

        # Select the next generation individuals
        ga.Selection(pop)
        pareto.update(pop)


        #################################################################
        #print generation
        print "  generation: ", format(i+1, '03d')





    print pareto




