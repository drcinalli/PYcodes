__author__ = 'quatrosem'
from world import World
from emoa import GA
from ast import literal_eval
from random import randint
import random
from robot import Robot
from copy import deepcopy



#Get Front
def fakeGetFront():

    fake=[[9, 8, 1, 18, 15, 1, 12, 11, 1, 13, 3, 1, 5, 16, 1, 9, 12, 1, 11, 11, 0, -1000, -1000, -1000, -1000, -1000, -1000, -1000, -1000, -1000, -1000, -1000, -1000, -1000, -1000, -1000],
          [9, 7, 1, 18, 15, 1, 12, 8, 1, 13, 3, 1, 5, 16, 2, 7, 14, 2, 11, 7, 0, 7, 15, 0, -1000, -1000, -1000, -1000, -1000, -1000, -1000, -1000, -1000, -1000, -1000, -1000],
          [9, 7, 1, 18, 15, 3, 12, 8, 1, 13, 3, 1, 5, 16, 2, 7, 14, 2, 11, 7, 0, 7, 15, 0, 18, 14, 0, -1000, -1000, -1000, -1000, -1000, -1000, -1000, -1000, -1000],
          [9, 7, 1, 18, 15, 3, 12, 8, 1, 13, 3, 1, 5, 16, 2, 7, 14, 2, 11, 7, 1, 7, 15, 0, 18, 14, 0, -1000, -1000, -1000, -1000, -1000, -1000, -1000, -1000, -1000],
          [9, 7, 1, 18, 15, 3, 12, 8, 1, 13, 3, 1, 5, 16, 2, 7, 14, 2, 11, 7, 1, 7, 15, 0, 18, 14, 1, -1000, -1000, -1000, -1000, -1000, -1000, -1000, -1000, -1000],
          [9, 7, 1, 18, 16, 3, 12, 8, 1, 13, 3, 1, 5, 16, 2, 7, 14, 2, 11, 7, 1, 7, 15, 0, 18, 17, 1, -1000, -1000, -1000, -1000, -1000, -1000, -1000, -1000, -1000],
          [9, 7, 1, 18, 16, 3, 12, 8, 1, 13, 3, 1, 5, 16, 2, 7, 14, 2, 11, 7, 1, 7, 15, 1, 18, 17, 1, -1000, -1000, -1000, -1000, -1000, -1000, -1000, -1000, -1000],
          [9, 7, 1, 18, 15, 3, 12, 8, 1, 13, 3, 1, 5, 16, 2, 7, 14, 2, 11, 7, 1, 7, 15, 1, 18, 14, 1, -1000, -1000, -1000, -1000, -1000, -1000, -1000, -1000, -1000],
          [9, 3, 2, 18, 16, 1, 16, 8, 3, 13, 2, 2, 5, 15, 4, 7, 14, 4, 17, 16, 0, 11, 3, 1, 17, 8, 1, 5, 14, 0, -1000, -1000, -1000, -1000, -1000, -1000],
          [9, 3, 2, 18, 16, 1, 16, 8, 3, 13, 2, 2, 5, 15, 4, 7, 14, 4, 17, 16, 1, 11, 3, 0, 17, 8, 1, 5, 14, 0, -1000, -1000, -1000, -1000, -1000, -1000],
          [9, 3, 2, 18, 16, 1, 16, 8, 3, 13, 2, 2, 5, 15, 4, 7, 14, 4, 17, 16, 1, 11, 3, 1, 17, 8, 0, 5, 14, 0, -1000, -1000, -1000, -1000, -1000, -1000],
          [9, 3, 2, 18, 16, 1, 16, 8, 3, 13, 2, 2, 5, 15, 4, 7, 14, 4, 17, 16, 1, 11, 3, 1, 17, 8, 0, 5, 14, 1, -1000, -1000, -1000, -1000, -1000, -1000],
          [9, 3, 2, 18, 16, 1, 16, 8, 3, 13, 2, 2, 5, 15, 4, 7, 14, 4, 17, 16, 1, 11, 3, 1, 17, 8, 1, 5, 14, 0, -1000, -1000, -1000, -1000, -1000, -1000],
          [9, 3, 2, 18, 16, 1, 16, 8, 3, 13, 2, 2, 5, 15, 4, 7, 14, 4, 17, 16, 1, 11, 3, 1, 17, 8, 1, 5, 14, 1, -1000, -1000, -1000, -1000, -1000, -1000],
          [3, 3, 2, 18, 16, 1, 16, 8, 3, 13, 2, 5, 5, 15, 4, 7, 14, 4, 17, 16, 0, 2, 3, 1, 17, 8, 1, 5, 14, 1, 12, 2, 0, -1000, -1000, -1000],
          [3, 3, 2, 18, 16, 1, 16, 8, 3, 13, 2, 5, 5, 15, 4, 7, 14, 4, 17, 16, 1, 2, 3, 1, 17, 8, 1, 5, 14, 0, 12, 2, 0, -1000, -1000, -1000],
          [3, 3, 2, 18, 16, 1, 16, 8, 3, 13, 2, 5, 5, 15, 4, 7, 14, 4, 17, 16, 1, 2, 3, 1, 17, 8, 1, 5, 14, 0, 12, 2, 1, -1000, -1000, -1000],
          [3, 3, 2, 18, 16, 1, 16, 8, 3, 13, 2, 5, 5, 15, 4, 7, 14, 4, 17, 16, 1, 2, 3, 1, 17, 8, 1, 5, 14, 1, 12, 2, 0, -1000, -1000, -1000],
          [3, 3, 2, 18, 16, 1, 16, 8, 3, 13, 2, 5, 5, 15, 4, 7, 14, 4, 17, 16, 1, 2, 3, 1, 17, 8, 1, 5, 14, 1, 12, 2, 1, -1000, -1000, -1000],
          [3, 3, 2, 18, 16, 1, 16, 8, 3, 13, 2, 5, 5, 15, 4, 7, 14, 6, 17, 16, 1, 2, 3, 1, 17, 8, 1, 5, 14, 1, 12, 2, 0, 6, 14, 0],
          [3, 3, 2, 18, 16, 1, 16, 8, 3, 13, 2, 5, 5, 15, 4, 7, 14, 6, 17, 16, 1, 2, 3, 1, 17, 8, 1, 5, 14, 0, 12, 2, 1, 6, 14, 0],
          [3, 3, 2, 18, 16, 1, 16, 12, 3, 13, 2, 5, 5, 15, 4, 7, 14, 6, 17, 16, 1, 2, 3, 1, 17, 12, 1, 5, 14, 1, 12, 2, 1, 6, 14, 0],
          [3, 3, 2, 18, 16, 1, 16, 8, 3, 13, 2, 5, 5, 15, 4, 7, 14, 6, 17, 16, 1, 2, 3, 1, 17, 8, 1, 5, 14, 1, 12, 2, 0, 6, 14, 1],
          [3, 3, 2, 18, 16, 1, 16, 8, 3, 13, 2, 5, 5, 15, 4, 7, 14, 6, 17, 16, 1, 2, 3, 1, 17, 8, 1, 5, 14, 0, 12, 2, 1, 6, 14, 1],
          [3, 3, 2, 18, 16, 1, 16, 8, 3, 13, 2, 5, 5, 15, 4, 7, 14, 6, 17, 16, 1, 2, 3, 1, 17, 8, 1, 5, 14, 1, 12, 2, 1, 6, 14, 0],
          [3, 3, 2, 18, 16, 1, 16, 8, 3, 13, 2, 5, 5, 15, 4, 7, 14, 6, 17, 16, 1, 2, 3, 1, 17, 8, 1, 5, 14, 1, 12, 2, 1, 6, 14, 1]
    ]

    return fake

def fakeGetFrontFit():

    fakefit= [(73.868159279972645, -23.0), (77.431658819361502, -35.0), (78.923887369973485, -46.0), (78.923887369973485, -46.0),
              (82.901600471735435, -58.0), (83.053405777305727, -69.0), (85.9976776873049, -70.0), (87.053405777305727, -81.0),
              (91.053405777305727, -93.0), (95.053405777305727, -105.0), (99.064495102245985, -116.0), (103.06449510224598, -128.0),
              (107.06449510224598, -140.0), (114.41421356237309, -151.0), (114.41421356237309, -151.0), (118.41421356237309, -163.0),
              (118.41421356237309, -163.0), (118.41421356237309, -163.0), (118.41421356237309, -163.0), (122.41421356237309, -175.0),
              (130.82842712474618, -186.0), (130.82842712474618, -186.0), (134.82842712474618, -198.0), (134.82842712474618, -198.0),
              (138.82842712474618, -210.0), (138.82842712474618, -210.0)]

    return fakefit

#take decision on what objective to look at: 0=first=cost=f1; 1=second=production=f2
def TakeDecision( value, ind1, ind2):

    coin = round(random.uniform(0, 1.0), 2)
    if coin <= value:
        return GetBetterCost(ind1,ind2)
    else:
        return GetBetterProduction(ind1, ind2)

#get the better individual on cost
def GetBetterCost(ind1, ind2):
    if ind1[0]<=ind2[0]:
        return 0
    else:
        return 1

#get the better individual on production
def GetBetterProduction(ind1, ind2):
    if ind1[1]<=ind2[1]:
        return 0
    else:
        return 1





#parameters: cross, mutation, number of generations, number of population
CXPB = 0.3
MUTPB= 0.1
#20x20
NGEN = 400
NPOP = 200
#50x50
#NGEN = 50
#NPOP = 600



if __name__=='__main__':

    print fakeGetFront()



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


    #declare Genetic Algorithm for the problem
    ga = GA(my_world,CXPB, MUTPB, NGEN, NPOP)

    bot = Robot()

    pop=[]
    #get Pareto Front
    pf=fakeGetFront()

    #fake population
    pop=ga.SetPopulationFake(my_world)
    del pop[len(pf):]

    #replace by front
    count=0
    for ind in pf:
       pop[count][0] = ind
       count +=1

    # set FITNESS to the entire front
    fitnesses = ga.GetFitness(pop)
    ga.AttachFitness(pop,fitnesses)

    print pop
    #create list of fitness
    front_fit=[]
    for i in pop:
        front_fit.append([i.fitness.values[0],i.fitness.values[1]])

    print front_fit


    clean_individual_answer=[0,0]
    clean_answer=[]
    front_size = len(pop)

    all_comparisons = []
    robot_comparison = []

    #create clean structure
    for i in xrange(front_size):
        clean_answer.append(deepcopy(clean_individual_answer))
    #clean answer done!


    for i in xrange(200):

        #so i is the robot index

        #copy clean answer
        robot_comparison= deepcopy(clean_answer)

        #loop from 1 to 5 random
        for j in xrange(6):

            #random elements for ind1 and ind2 FROM front
            a = randint(0,(front_size-1))
            b = randint(0,(front_size-1))
            while a == b:
                b = randint(0,front_size-1)

            #keep their indexes
            #ask for comparisons to robo_lst[i]
            ans = bot.TakeDecision(.45, front_fit[a], front_fit[b])

            #use index to adjust the votes (pro and against)
            if ans == 0:
                #sum robot [a]
                robot_comparison[a][0] += 1
                robot_comparison[b][1] += 1

            else:
                #sum robot [b]
                robot_comparison[b][0] += 1
                robot_comparison[a][1] += 1

            #save robot comparison
            chromosomeOne= pop[a][0]
            chromosomeOneIndex=a
            chromosomeTwo= pop[b][0]
            chromosomeTwoIndex=b

            #play.save()

        all_comparisons.append(deepcopy(robot_comparison))

    final_computation=[]
    #create clean structure
    for i in xrange(front_size):
        final_computation.append(deepcopy(clean_individual_answer))
    #clean answer done!

    for i in all_comparisons:
        for j in xrange(len(i)):
            final_computation[j][0] +=  i[j][0]
            final_computation[j][1] +=  i[j][1]


    print final_computation