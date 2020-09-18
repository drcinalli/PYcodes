main########

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





if __name__=='__main__':


    #World domain

    #Cenario 1: number of Area must be greater than or equal the number of Units
    #my_world = World(50,50)
    #my_world.MaxArea(7)
    #my_world.CreateArea(3,3,10)
    #my_world.CreateArea(18,15,5)
    #my_world.CreateArea(28,1,8)
    #my_world.CreateArea(22,6,3)
    #my_world.CreateArea(35,33,14)
    #my_world.CreateArea(8,27,5)
    #my_world.CreateArea(12,39,7)
    #my_world.PrintGeoAreas()

    #Cenario 2
    my_world = World(20,20)
    my_world.MaxArea(6)
    my_world.CreateArea(3,3,7)
    my_world.CreateArea(18,15,2)
    my_world.CreateArea(12,8,5)
    my_world.CreateArea(13,1,3)
    my_world.CreateArea(2,15,4)
    my_world.CreateArea(7,12,3)

    ################
    #for individual#
    ################
    for i in range(0,1):
        #create gateways
        gateways = my_world.CreateGateways()
        print gateways
        #check gateways
        y_or_n = my_world.CheckGateways(gateways)
        print y_or_n
        #create units
        units = my_world.CreateUnits()
        print units
        #GET num units for individuals
        #check units
        y_or_n = my_world.CheckUnits(units)
        print y_or_n
        #create links
        my_world.CreateLinks(gateways,units)
        print gateways
        #check units
        y_or_n = my_world.CheckLinks(gateways, units)
        print y_or_n

        #declare Genetic Algorithm for the problem
        ga = GA(my_world,0.5, 0.2, 40,50)
        #man = ga.SetIndividual(gateways, units, my_world.all_units)
        #print man
        pop=ga.SetPopulation()
        # pop=[]
        # for i in pop_aux:
        #     pop.append(i[0])
        # print pop

        # Evaluate the entire population
        fitnesses = ga.GetFitness(pop)
        ga.AttachFitness(pop,fitnesses)
        print fitnesses

        print("  Evaluated %i individuals" % len(pop))


        # Select the next generation individuals
        offspring = ga.Selection(pop)

        # Gather all the fitnesses in one list and print the stats
        fits = [ind.fitness.values[0] for ind in offspring]

        length = len(offspring)
        mean = sum(fits) / length
        sum2 = sum(x*x for x in fits)
        std = abs(sum2 / length - mean**2)**0.5

        print("  Min %s" % min(fits))
        print("  Max %s" % max(fits))
        print("  Avg %s" % mean)
        print("  Std %s" % std)


        #Plot the World
        #my_world.PlotWorldDetails(gateways,units)

        #Plot the World MANY times
        for i in pop:
            print i[0]
            gates=my_world.GetGates(i[0])
            units=my_world.GetUnits(i[0])
            #my_world.PlotWorldDetails(gates,units)


    print my_world.author




WORLD ###########
'''
Created on 18/08/2014

@author: quatrosem
'''
import numpy as np
import matplotlib.pyplot as plt
from random import randint
import matplotlib.path as mpath
import matplotlib.lines as mlines
import matplotlib.patches as mpatches
from time import sleep
import math


#Class of the Problem Domain
class World:
    #class variables
    description = "This is the domain of the problem"
    author = "Daniel Cinalli"

    #quantities
    all_areas = 5
    all_units = 4
    all_units_type = 2

    #costs
    BRL_gateway = 50
    BRL_unit = 200
    #BRL_allunits = all_units * BRL_unit
    BRL_allgates = all_areas * BRL_gateway

    #methods
    def __init__(self,m,n):
        self.m = m
        self.n = n
        self.geo      = [[1 for x in xrange(self.m)] for x in xrange(self.n)]
        #self.geo_dist = [[0]*(m*n) for x in xrange(m*n)]
        #self.GeoDist(self.geo_dist)
        self.areas = []

    #define number of areas in the World
    def MaxArea(self, i):
        self.all_areas = i

    #Plot the World in graphical mode
    def PlotWorld(self, gateways=None, units=None, ax=None):
        #matrix = [[1 for x in xrange(self.m)] for x in xrange(self.n)]
        ax = ax if ax is not None else plt.gca()

        #if not max_weight:
        #    max_weight = 2**np.ceil(np.log(np.abs(matrix).max())/np.log(2))

        #declare the background
        ax.patch.set_facecolor('gray')
        ax.set_aspect('equal', 'box')
        ax.xaxis.set_major_locator(plt.NullLocator())
        ax.yaxis.set_major_locator(plt.NullLocator())

        #loop on the World
        for (x,y),w in np.ndenumerate(self.geo):
            #regular square (rectangle)
            #color = 'white' if w > 0 else 'black'
            color = 'white'
            border = 'black'
            size = np.sqrt(np.abs(w))
            rect = plt.Rectangle([x - size / 2, y - size / 2], size, size,
                                 facecolor=color, edgecolor=border)
            ax.add_patch(rect)


        #draw production areas
        for w in self.areas:
            for i in range(w[0], w[0]+w[2]):
                for j in range(w[1], w[1]+w[2]):
                    #print i,j
                    rect = plt.Rectangle([i - size / 2, j - size / 2], size, size,
                                 facecolor='#A8A8A8', edgecolor=border)
                    ax.add_patch(rect)



        ax.autoscale_view()
        ax.invert_yaxis()

        #Font definition to the Plot Title
        # font = {'family' : 'serif',
        #         'color'  : 'black',
        #         'weight' : 'bold',
        #         'size'   : 16,
        #         }
        plt.title('Resource Placement and Assignment')

        #show World
        plt.show()

    #Plot the World in graphical mode with details
    def PlotWorldDetails(self, gateways=None, units=None, ax=None):
        #matrix = [[1 for x in xrange(self.m)] for x in xrange(self.n)]
        ax = ax if ax is not None else plt.gca()

        #if not max_weight:
        #    max_weight = 2**np.ceil(np.log(np.abs(matrix).max())/np.log(2))

        #declare the background
        ax.patch.set_facecolor('gray')
        ax.set_aspect('equal', 'box')
        ax.xaxis.set_major_locator(plt.NullLocator())
        ax.yaxis.set_major_locator(plt.NullLocator())

        #loop on the World
        for (x,y),w in np.ndenumerate(self.geo):
            #regular square (rectangle)
            #color = 'white' if w > 0 else 'black'
            color = 'white'
            border = 'black'
            size = np.sqrt(np.abs(w))
            rect = plt.Rectangle([x - size / 2, y - size / 2], size, size,
                                 facecolor=color, edgecolor=border)
            ax.add_patch(rect)

        #draw production areas
        for w in self.areas:
            for i in range(w[0], w[0]+w[2]):
                for j in range(w[1], w[1]+w[2]):
                    #print i,j
                    rect = plt.Rectangle([i - size / 2, j - size / 2], size, size,
                                 facecolor='#A8A8A8', edgecolor=border)
                    ax.add_patch(rect)

        #draw gateways
        for w in gateways:
            rect = plt.Rectangle([w[0] - size / 2, w[1] - size / 2], size, size,
                                 facecolor='black', edgecolor=border)
            ax.add_patch(rect)
            u= units[w[2]-1]
            #line 2 x in the first parameter ... them 2 y on the second
            line = mlines.Line2D((w[0],u[0]) , (w[1],u[1]),color='#AC0000', linestyle='--', lw=2., alpha=.6)
            ax.add_line(line)



        #draw production units
        for w in units:
            if w[2]==0:
                circle = plt.Circle((w[0], w[1]), radius=0.4, fc='b')
            else:
                circle = plt.Circle((w[0], w[1]), radius=0.4, fc='#1BA015')
            ax.add_patch(circle)



        ax.autoscale_view()
        ax.invert_yaxis()

        plt.title('Resource Placement and Assignment')

        #show World
        #plt.show()
        plt.show(block=False)
        plt.pause(3.0001)
        plt.close('all')


    #build distance matrix with costs = 1
    def GeoDist(self, geo_dist):

        for i in range(0,self.m*self.n):
            for j in range(0,self.m*self.n):
                self.geo_dist[i][j] = abs(j-i)

        return

    #create areas of production
    def CreateArea(self, x, y, len):
        aux=[x, y, len]

        #?????
        #before append, check overlap

        self.areas.append(aux)

    #print Geo elements
    def PrintGeoElement(self,x,y):
        print self.geo[x][y]

    #print GeoDist elements
    def PrintGeoDistElement(self,x,y):
        print self.geo_dist[x][y]

    #print Geo Areas of production
    def PrintGeoAreas(self):
        print self.areas

    #create all the World's gateways
    def CreateGateways(self):
        glist=[]
        for i in self.areas:
            glist.append(self.CreateGate(i))
        return glist

    #create a unique gateway
    def CreateGate(self, the_area):
        x= randint(the_area[0],the_area[0]+the_area[2]-1)
        y= randint(the_area[1],the_area[1]+the_area[2]-1)
        z=-1 #this is the link... it will be done later
        return (x,y,z)

    #check gateways
    def CheckGateways(self, glist):
        if len(glist) != len(self.areas):
            print 'ERROR: different number of gateways and areas.'
            return 0

        #copy of areas
        cp_area = list(self.areas)

        #check each area
        for i in glist:
            #if the gateway is not inside any area: error!
            if self.CheckGate(i,cp_area)==0:
                return 0
        return 1


    #check a specific gateway is inside an area
    def CheckGate(self, gateway, remaining_areas):
        for i in remaining_areas:
            if i[0]<= gateway[0] <i[0]+i[2] and i[1]<= gateway[1] <i[1]+i[2]:
                remaining_areas.remove(i)
                return 1

        #did not find the gateway inside the area
        return 0

    #check a specific gateway is inside an area
    def CheckGate_onepass(self, gateway, remaining_areas):
        for i in remaining_areas:
            if i[0]<= gateway[0] <i[0]+i[2] and i[1]<= gateway[1] <i[1]+i[2]:
                return 1

        #did not find the gateway inside the area
        return 0

    #create all the World's production units
    def CreateUnits(self):
        ulist=[]
        num=randint(1,self.all_units)
        for i in range(0,num):
            ulist.append(self.CreateProdUnit())
        return ulist

    #create a unique production unit
    def CreateProdUnit(self):
        while True:
            #choose point and type of unit
            x= randint(0,self.m-1)
            y= randint(0,self.n-1)
            z= randint(0,self.all_units_type-1)

            #check if it is out of the gateway area
            if self.CheckGate_onepass((x,y,z),self.areas)==0:
                break

        return (x,y,z)

    #check production units
    def CheckUnits(self, ulist):
        if len(ulist) > self.all_units:
            print 'ERROR: superior number of units in domain.'
            return 0

        #check each unit
        for i in ulist:
            #if the unit is inside any area: error!
            if self.CheckGate_onepass(i,self.areas)!=0:
                return 0
        return 1

    #create links
    def CreateLinks(self, glist, ulist):

        #copy of units and gateways
        cp_unit = list(ulist)
        cp_gate = list(glist)
        r=0

        #for each gateway, link to one unit
        for i in cp_gate:
            #link gateway to unit. The flag "r" indicates when all units were linked, so the free rand choose can happen
            new_gateway = self.CreateLink2Unit(i,cp_unit, ulist,r)
            if len(cp_unit)==0:
                r=1
                cp_unit = list(ulist)

            #rearrange the list
            glist.remove(i)
            glist.append(new_gateway)


    #create specific link
    def CreateLink2Unit(self,gateway,units,read_units,r):

        #choose one of the units
        i=randint(1,len(units))
        #discover its index in the original units list
        j= read_units.index(units[i-1])

        #if there are gateways and units to be linked, remove this from the list
        if r!=1:
            del units[i-1]
        #return tuple
        return (gateway[0], gateway[1],j+1)





    #check links
    def CheckLinks(self,glist,units):

        cp_unit = list(units)

        #check each link inside gateway
        for i in glist:
            #if the gateway is not linked to some unit: error!
            #if self.CheckLink2Unit(i,units,cp_unit)==0:
            if i[2]<=len(units):
                #remove the unit from aux list
                if units[i[2]-1] in cp_unit: cp_unit.remove(units[i[2]-1])
            else:
                return 0
        #if the aux list is not empty... ERROr
        if not cp_unit:
            return 1

        return 0

    #create all structure for a individual
    def CreateFull(self):
        #create gateways
        gateways = self.CreateGateways()

        #create units
        units = self.CreateUnits()

        #create links
        self.CreateLinks(gateways,units)

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
        for i in range(aux,self.all_units):
            man.append(-1)
            man.append(-1)
            man.append(-1)

        return man

    #get gateways from DNA
    def GetGates(self, dna):
        #gateways
        gates = []
        start = 0
        for i in range(0,self.all_areas):
            stop = start + 3
            #x=dna[start:stop]
            gates.append(tuple(dna[start:stop]))
            start = stop

        return gates

    #get units from DNA
    def GetUnits(self, dna):
        #units
        units = []
        start = self.all_areas*3
        for i in range(self.all_areas,self.all_areas+self.all_units):
            stop = start + 3
            #x=dna[start:stop]
            aux=dna[start:stop]
            if aux[0]>=0 and aux[1]>=0 and aux[2]>=0: #just in case I will force this 3 checks
                units.append(tuple(dna[start:stop]))
            start = stop

        return units

            # x= dna[i]
            # y= dna[i+1]
            # l= dna[i+2]
            # i=i+2
            # gates.append(x,y,l)
        #for i in range(0,self.all_areas-1):
        #    gates.append(dna[i:i+2])

    def CostLinks(self,glist,units):

        #get each link inside gateway
        total_cost=0
        for i in glist:

            #get right position in the world for Matrix Distance
            #dim = self.m * self.n
            #x= (self.n *(i[1]-1)) + (i[0]+1)
            #y= (self.n *(units[i[2]-1][1])) + (units[i[2]-1][0]+1)
            #total_cost = total_cost + self.geo_dist[x][y]
            c=math.sqrt( (units[i[2]-1][0] - i[0])**2 + (units[i[2]-1][1] - i[1])**2 )
            total_cost = total_cost + c


        return int(total_cost)


    #check specific link

    #balance Gateways again

    #balance Units again

    #balance Links again


EMOA ########################

'''
Created on 21/08/2014

@author: quatrosem
'''

import random
from world import World
from deap import base
from deap import creator
from deap import tools

toolbox = base.Toolbox()

#Fitness of OBJECTIVE #1
def f1Cost(individual, world):

    gates=world.GetGates(individual[0])
    units=world.GetUnits(individual[0])


    #check after cross and/or mutation
    if world.CheckGateways(gates) == 0 or world.CheckUnits(units) == 0 or world.CheckLinks(gates, units) == 0:
        return 1000000,

    cost_units= len(units) * world.BRL_unit
    cost_gates= world.BRL_allgates#world.all_areas * world.BRL_gateway
    cost_links= world.CostLinks(gates,units)

    return cost_gates+cost_units+cost_links,
    #return sum(individual)


class GA:
    #class variables
    description = "This is the Genetic Algorithm class of the problem"
    author = "Daniel Cinalli"

    #parameters: mutation and number, generations, number of population
    CXPB, MUTPB, NGEN = 0.5, 0.2, 40
    num_population    = 50



    def __init__(self, my_world, CXPB=None, MUTPB=None, NGEN = None,  num_population=None):

        #define fitness of the individual
        creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
        creator.create("Individual", list, fitness=creator.FitnessMin)


        # Attribute generator
        toolbox.register("individual_gen", my_world.CreateFull)
        #toolbox.register("individual_gen", random.randint, 0, 1)


        # Structure initializers
        toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.individual_gen, 1)
        toolbox.register("population", tools.initRepeat, list, toolbox.individual)

        # Operator registering
        toolbox.register("evaluate", f1Cost, world=my_world)
        toolbox.register("mate", tools.cxTwoPoint)
        toolbox.register("mutate", tools.mutFlipBit, indpb=0.05)
        toolbox.register("select", tools.selTournament, tournsize=3)

        #parameters: mutation and number, generations, number of population
        self.CXPB  = CXPB
        self.MUTPB = MUTPB
        self.NGEN  = NGEN
        self.num_population = num_population


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

    #attach fitness to the individual
    def Selection(self, pop):

        # Select the next generation individuals
        offspring = toolbox.select(pop, len(pop))
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
                print mutant
                toolbox.mutate(mutant[0])
                print mutant
                del mutant.fitness.values

        # Evaluate the individuals with an invalid fitness
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit

        print("  Evaluated %i individuals" % len(invalid_ind))

        # The population is entirely replaced by the offspring
        pop[:] = offspring


        return offspring

