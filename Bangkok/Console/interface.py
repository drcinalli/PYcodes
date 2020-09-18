from world import World
from emoa import GA
from robot import Robot
from service.models import GameWorld, Area, Experiment, Generation, PFront, Population, Player, Play
from deap import tools
from django.db import transaction
import datetime
from ast import literal_eval
from random import randint
import random
from copy import deepcopy


class InterfaceGA:
    #class variables
    description = "This is the Genetic Algorithm INTERFACE class of the problem"
    author = "Daniel Cinalli"


    #Start Evolution
    def fakeStartEvolution(self, name,date,type ,description, flag, num_population, num_robots, num_levels, num_gen,
                           block_size, gen_threshold, actual_gen, player, mutation, cross):
        x=1

    def StartEvolution(self, name,date,type ,description, flag, num_population, num_robots, num_levels, num_gen,
                       block_size, gen_threshold, actual_gen, player, mutation, cross, wrld):



        #ATENCAO!! Deve existir pelo menos 1 GameWorld
        #depois arrumo essa verificacao no script do banco, mas agora PRECISA ser assim

        #get World config
        gameworld = GameWorld.objects.get(id=wrld)
        my_world = World(gameworld.m,gameworld.n)
        my_world.MaxArea(gameworld.max_areas)
        my_world.MaxUnits(gameworld.max_units)
        my_world.Production(gameworld.prod_unit0,gameworld.prod_unit1)
        my_world.Costs(gameworld.cost_gateway,gameworld.cost_unit0,gameworld.cost_unit1)

        for area in Area.objects.all().filter(world=gameworld.id):
            my_world.CreateArea(area.x,area.y,area.length)

        #declare Genetic Algorithm for the problem
        ga = GA(my_world,cross, mutation, num_gen, num_population)
        pareto=tools.ParetoFront()
        pop=[]

        #create a new exp
        now=datetime.datetime.now()
        exp = Experiment(world=gameworld, name=name, date=date, block_size=block_size, start=now, type=type, description=description,
                         flag=flag, actual_gen=actual_gen,  CXPB=cross, MUTPB=mutation, NGEN=num_gen, NPOP=num_population,
                         num_robots=num_robots, numLevels=num_levels, gen_threshold=gen_threshold, numMinPlayers=player, time_elapsed_end=0)
        exp.save()



        #set NEW Population
        pop=ga.SetPopulation()


        #set block for the generation block table
        num_block=1

        # set FITNESS to the entire population
        fitnesses = ga.GetFitness(pop)
        ga.AttachFitness(pop,fitnesses)
        #assign the crowding distance to the individuals
        ga.SetCrowdingDistance(pop)
        #discover again the LAST Pareto
        pareto.update(pop)



        #loop for BLOCK generations
        for i in range(1,exp.block_size):

                # Select the next generation individuals
                #ga.Selection(pop, my_world)
                ga.NSGASelection(pop, my_world)
                pareto.update(pop)



        #update actual generations with block_size
        exp.actual_gen += exp.block_size

        #save Generations Block
        gen = Generation(experiment=exp, block=num_block, comparisons="to be collected")
        gen.save()

        #save population and Front into database
        with transaction.atomic ():
            count=1
            for ind in pop:
                population = Population(generation=gen, chromosome=str(ind[0]), index=count)
                population.save()
                count +=1
            count=1
            for ind in pareto:
                pfront = PFront(generation=gen, chromosome=str(ind[0]), index=count)
                pfront.save()
                count +=1


        #set the next status according the number of generations
        if exp.actual_gen >= exp.NGEN:
            exp.flag = 'F'
        else:
            exp.flag = 'R'

        exp.save()

        #run robots
        self.RunRobots(exp, num_block, ga, my_world)

        return exp



    #Start Evolution
    def fakeContinueEvolution(self):
        x=1

    def ContinueEvolution(self):

        #get last ID of the Experiment
        exp = Experiment.objects.latest('id')

        #ATENCAO!! Deve existir pelo menos 1 GameWorld
        #depois arrumo essa verificacao no script do banco, mas agora PRECISA ser assim
        #inclusive fixei HARDCODE que sera esse mundo apenas

        #get World config
        gameworld = GameWorld.objects.get(id=1) ###HARDCODE
        my_world = World(gameworld.m,gameworld.n)
        my_world.MaxArea(gameworld.max_areas)
        my_world.MaxUnits(gameworld.max_units)
        my_world.Production(gameworld.prod_unit0,gameworld.prod_unit1)
        my_world.Costs(gameworld.cost_gateway,gameworld.cost_unit0,gameworld.cost_unit1)

        for area in Area.objects.all().filter(world=gameworld.id):
            my_world.CreateArea(area.x,area.y,area.length)

        #declare Genetic Algorithm for the problem
        ga = GA(my_world,exp.CXPB, exp.MUTPB, exp.NGEN, exp.NPOP)
        pareto=tools.ParetoFront()
        pop=[]

        #check if it is START or CONTINUE evolution - must know if I will create a new experiment or use the one received
        #if F then start NEW
        # if exp.flag=='F':
        #
        #     #create a new exp
        #     now=datetime.datetime.now()
        #     exp = Experiment(world=gameworld, name='Bangkok experimento ' + now.strftime("%Y-%m-%d %H:%M:%S"), date=now, block_size=10,
        #                      flag='W', actual_gen=0, keep_interaction=True, CXPB=0.3, MUTPB=0.1, NGEN=200, NPOP=300)
        #     exp.save()
        #
        #
        #     #set NEW Population
        #     pop=ga.SetPopulation()
        #
        #
        #     #set block for the generation block table
        #     num_block=1
        #
        # else:


        #number of the next block of generations
        gen = Generation.objects.latest('id')
        num_block=gen.block + 1

        #fake population
        pop=ga.SetPopulationFake(my_world)
        #get the LAST Population
        count=0
        for ind in Population.objects.all().filter(generation=gen.id):
           pop[count][0] = literal_eval(ind.chromosome)
           count +=1

        ###### EM ########


        # set FITNESS to the entire population
        fitnesses = ga.GetFitness(pop)
        ga.AttachFitness(pop,fitnesses)
        #assign the crowding distance to the individuals
        ga.SetCrowdingDistance(pop)
        #discover again the LAST Pareto
        pareto.update(pop)


        #
        if exp.actual_gen + exp.block_size <= exp.gen_threshold and num_block <= exp.numLevels:
            loop= exp.block_size
        else:
            loop= exp.NGEN - exp.actual_gen

        #loop for BLOCK generations
        for i in range(1,loop):

            # Select the next generation individuals
            #ga.Selection(pop, my_world)
            ga.NSGASelection(pop, my_world)
            pareto.update(pop)



        #update actual generations with block_size
        exp.actual_gen += loop

        #save Generations Block
        gen = Generation(experiment=exp, block=num_block, comparisons="to be collected")
        gen.save()

        #save population and Front into database
        with transaction.atomic ():
            count=1
            for ind in pop:
                population = Population(generation=gen, chromosome=str(ind[0]), index=count)
                population.save()
                count +=1
            count=1
            for ind in pareto:
                pfront = PFront(generation=gen, chromosome=str(ind[0]), index=count)
                pfront.save()
                count +=1


        #set the next status according the number of generations
        if exp.actual_gen >= exp.NGEN:
            exp.flag = 'F'
        else:
            exp.flag = 'R'

        exp.save()

        #run robots
        self.RunRobots(exp, num_block, ga, my_world)




    #Get Front
    def fakeGetFront(self):

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


    #Get Front
    def GetFront(self):

        #get last generation inserted
        gen = Generation.objects.latest('id')
        pop=[]

        #get the LAST Front
        for ind in PFront.objects.all().filter(generation=gen.id):
           pop.append(literal_eval(ind.chromosome))


        return pop


    #Get Popuation
    def fakeGetPopulation(self):


        fake=[[6, 6, 2, 18, 15, 1, 14, 11, 1, 13, 1, 2, 4, 15, 1, 7, 13, 3, 13, 13, 0, 13, 7, 1, 10, 19, 0, -1000, -1000, -1000, -1000, -1000, -1000, -1000, -1000, -1000],
              [5, 9, 3, 19, 15, 6, 14, 10, 1, 13, 3, 2, 5, 18, 4, 8, 12, 5, 15, 18, 1, 19, 4, 1, 12, 18, 1, 8, 17, 0, 7, 18, 1, 10, 7, 1],
              [9, 3, 3, 19, 15, 1, 13, 12, 2, 13, 1, 1, 4, 16, 1, 9, 13, 3, 14, 15, 1, 15, 17, 1, 7, 1, 1, -1000, -1000, -1000, -1000, -1000, -1000, -1000, -1000, -1000],
              [5, 8, 2, 18, 15, 1, 14, 12, 4, 14, 3, 3, 2, 15, 3, 9, 12, 2, 16, 16, 1, 19, 7, 1, 5, 12, 1, 17, 10, 1, -1000, -1000, -1000, -1000, -1000, -1000],
              [9, 8, 3, 19, 16, 2, 12, 9, 1, 15, 2, 3, 5, 16, 4, 7, 14, 5, 12, 7, 0, 12, 7, 1, 16, 1, 0, 17, 13, 0, 10, 12, 1, -1000, -1000, -1000],
              [6, 6, 1, 18, 16, 2, 14, 10, 2, 14, 3, 2, 2, 17, 1, 8, 14, 2, 3, 11, 1, 11, 3, 1, -1000, -1000, -1000, -1000, -1000, -1000, -1000, -1000, -1000, -1000, -1000, -1000],
              [9, 4, 1, 2, 15, 2, 12, 9, 1, 13, 3, 1, 18, 15, 3, 7, 13, 3, 11, 6, 0, 2, 13, 0, 10, 15, 0, -1000, -1000, -1000, -1000, -1000, -1000, -1000, -1000, -1000],
              [9, 4, 1, 2, 15, 2, 12, 9, 1, 13, 3, 1, 18, 15, 3, 7, 13, 3, 11, 6, 0, 2, 13, 0, 10, 15, 0, -1000, -1000, -1000, -1000, -1000, -1000, -1000, -1000, -1000],
              [4, 4, 5, 18, 16, 1, 13, 9, 3, 14, 3, 3, 4, 16, 4, 7, 13, 2, 15, 18, 1, 6, 12, 0, 15, 7, 1, 5, 14, 1, 0, 3, 1, -1000, -1000, -1000],
              [4, 4, 5, 18, 16, 1, 13, 9, 3, 14, 3, 3, 4, 16, 4, 7, 13, 2, 15, 18, 1, 6, 12, 0, 15, 7, 1, 5, 14, 1, 0, 3, 1, -1000, -1000, -1000],
              [4, 6, 4, 19, 16, 3, 16, 9, 5, 14, 3, 1, 3, 18, 6, 7, 13, 2, 13, 0, 1, 11, 12, 1, 18, 17, 1, 4, 10, 1, 18, 6, 1, 0, 12, 1],
              [4, 4, 5, 18, 16, 1, 16, 9, 3, 14, 3, 3, 5, 16, 4, 7, 13, 2, 15, 18, 1, 6, 12, 0, 15, 7, 0, 5, 14, 1, 0, 3, 1, -1000, -1000, -1000],
              [4, 6, 4, 19, 16, 3, 16, 9, 5, 14, 3, 1, 3, 18, 6, 7, 13, 2, 13, 0, 1, 11, 12, 1, 18, 17, 1, 4, 10, 1, 18, 6, 1, 0, 12, 1],
              [4, 6, 4, 19, 16, 3, 16, 9, 5, 14, 3, 1, 3, 18, 6, 7, 13, 2, 13, 0, 1, 11, 12, 1, 18, 17, 1, 4, 10, 1, 18, 6, 1, 0, 12, 1],
              [9, 4, 1, 18, 15, 1, 12, 8, 1, 13, 3, 1, 4, 16, 1, 7, 12, 1, 11, 6, 0, -1000, -1000, -1000, -1000, -1000, -1000, -1000, -1000, -1000, -1000, -1000, -1000, -1000, -1000, -1000],
              [6, 9, 5, 18, 16, 1, 15, 8, 3, 14, 3, 6, 2, 15, 4, 7, 12, 2, 17, 16, 1, 6, 12, 1, 15, 7, 0, 1, 15, 1, 6, 10, 0, 14, 4, 1],
              [9, 6, 1, 5, 15, 2, 12, 8, 1, 13, 3, 1, 18, 15, 3, 7, 14, 2, 11, 6, 0, 6, 14, 0, 17, 15, 0, -1000, -1000, -1000, -1000, -1000, -1000, -1000, -1000, -1000],
              [6, 9, 2, 18, 16, 1, 15, 8, 3, 14, 3, 5, 2, 15, 4, 7, 12, 2, 17, 16, 1, 6, 10, 1, 15, 7, 1, 1, 15, 1, 14, 4, 1, -1000, -1000, -1000],
              [6, 9, 5, 18, 16, 1, 15, 8, 3, 14, 3, 6, 2, 15, 4, 7, 12, 2, 17, 16, 1, 6, 12, 1, 15, 7, 1, 1, 15, 1, 6, 10, 1, 14, 4, 1]
        ]

        return fake


    #Get Front
    def GetPopulation(self):

        #get last generation inserted
        gen = Generation.objects.latest('id')
        pop=[]

        #get the LAST Front
        for ind in Population.objects.all().filter(generation=gen.id):
           pop.append(literal_eval(ind.chromosome))


        return pop



    #Set results from COIN
    def fakeSetComparisonsResult(self, fake):
        #modelo de entrada para que eu possa pegar as votacoes
        x= [[8,9],[0,4], [5,6]]
        print x

    #Set results from COIN
    def SetComparisonsResult(self, results):
        #save results into database

        #get last generation inserted
        gen = Generation.objects.latest('id')

        gen.comparisons = results
        gen.save()


    def fakeGetArea(self):
        return [[3,3,7],[18,15,2],[12,8,5],[13,1,3],[2,15,4],[7,12,3]]
        # my_world.CreateArea(3,3,7)
        # my_world.CreateArea(18,15,2)
        # my_world.CreateArea(12,8,5)
        # my_world.CreateArea(13,1,3)
        # my_world.CreateArea(2,15,4)
        # my_world.CreateArea(7,12,3)


    def GetArea(self):
        #get World config
        gameworld = GameWorld.objects.get(id=1) ###HARDCODE

        area_lst=[]
        for area in Area.objects.all().filter(world=gameworld.id):
            area_lst.append([area.x, area.y, area.length])

        return area_lst


    def fakeGetStatus(self):
        return 'R'
        #status R diz que o GA pode CONTINUAR
        #status F diz que o GA pode INICIAR um NOVO EXPERIMENTO

        #veja no models
        # FLAGS = (('W', 'Waiting'),
        #          ('R', 'Ready'),
        #          ('F', 'Finished'),
        #          ('I', 'Idle'))


    def GetStatus(self):
        #get last ID of the Experiment
        exp = Experiment.objects.latest('id')

        return exp.flag
        #status R diz que o GA pode CONTINUAR
        #status F diz que o GA pode INICIAR um NOVO EXPERIMENTO

    #run robots
    def RunRobots(self, exp, num_block, ga, my_world):

        pop=[]
        out=[]
        robot_lst=[]
        #get Pareto Front
        pf=self.GetFront()


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


        #create list of fitness
        front_fit=[]
        for i in pop:
            front_fit.append([i.fitness.values[0],i.fitness.values[1]])



        #GetRobotComparisons
        if num_block>1:
            #get plays from first round
            k=Play.objects.all().filter(play_experiment=exp.id, level=1).values('play_player').distinct()

            for i in k:
                robot_lst.append(Player.objects.get(id=i['play_player']))


        else:
            #robot creation
            #create Robots
            robot_lst = self.CreateRobots(exp.num_robots)


        out = self.GetRobotComparisons(pop, front_fit, robot_lst, exp, num_block)

        #return robot_lst[6].PrintStyle()

        #return out

    #create NUM robots
    def CreateRobots(self,num_robots):
        #list of robots
        robot_lst=[]

        #minimum is 3 robots
        if num_robots < 3:
            num_robots = 3

        ###############
        #create robots#
        ###############

        #change this .... not now ... i am in a rush
        robot = Player.objects.filter(objective1_pref=1, type='C')
        #create the 3 basic styles: Pro; Anti; Random
        if not robot:
            #cria
            robot = Player(username='Robot1.00_0', email='robot@noemail.com', password='robot', schooling='robot', gender='robot', age=2001, name='Robot 1.00/0', type='C', objective1_pref=1)
            robot.save()
        robot = Player.objects.get(objective1_pref=1, type='C')
        robot_lst.append(robot)

        robot = Player.objects.filter(objective1_pref=0, type='C')
        if not robot:
            #cria
            robot = Player(username='Robot0_1.00', email='robot@noemail.com', password='robot', schooling='robot', gender='robot', age=2001, name='Robot 0/1.00', type='C', objective1_pref=0)
            robot.save()
        robot = Player.objects.get(objective1_pref=0, type='C')
        robot_lst.append(robot)

        robot = Player.objects.filter(objective1_pref=.5, type='C')
        if not robot:
            #cria
            robot = Player(username='Robot0.50_0.50', email='robot@noemail.com', password='robot', schooling='robot', gender='robot', age=2001, name='Robot 0.50/0.50', type='C', objective1_pref=.5)
            robot.save()
        robot = Player.objects.get(objective1_pref=.5, type='C')
        robot_lst.append(robot)


        #create more random robots
        for i in xrange(num_robots-3):
            p = round(random.uniform(0, 1.0), 2)
            robot = Player.objects.filter(objective1_pref=p, type='C')
            if not robot:
                #cria
                robot = Player(username='Robot'+str(p)+'_'+str(1-p), email='robot@noemail.com', password='robot', schooling='robot', gender='robot', age=2001, name='Robot '+str(p)+'/'+str(1-p), type='C', objective1_pref=p)
                robot.save()
            robot = Player.objects.get(objective1_pref=p, type='C')
            robot_lst.append(robot)

        return robot_lst

    #create Robots comparisons
    def GetRobotComparisons(self, pop, front_fit, robot_lst, exp, num_block):
        #example of row
        # [first_robot] = [...] = [ [8,9],[0,4], ..., [5,6] ]
        #first_robot[0][0]= 8
        #first_robot[0][1]= 9

        bot = Robot()
        clean_individual_answer=[0,0]
        clean_answer=[]
        front_size = len(pop)
        robot_size = len(robot_lst)

        all_comparisons = []
        robot_comparison = []

        #create clean structure
        for i in xrange(front_size):
            clean_answer.append(deepcopy(clean_individual_answer))
        #clean answer done!

        #for each Robot
        #with transaction.atomic ():

        for i in xrange(robot_size):

            #so i is the robot index

            #copy clean answer
            robot_comparison=[]
            robot_comparison=deepcopy(clean_answer)

            #loop from 1 to 5 random
            for j in xrange(randint(1,5)):

                #random elements for ind1 and ind2 FROM front
                a = randint(0,front_size-1)
                b = randint(0,front_size-1)
                while a == b:
                    b = randint(0,front_size-1)

                #keep their indexes
                #ask for comparisons to robo_lst[i]
                #k=robot_lst[i].objective1_pref
                ans = bot.TakeDecision(.5, front_fit[a], front_fit[b])

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
                play = Play(play_experiment=exp, play_player=robot_lst[i], answer_time=0, points=0, answer=ans,
                            level=num_block, chromosomeOne= pop[a][0], chromosomeOneIndex=a, chromosomeTwo= pop[b][0], chromosomeTwoIndex=b)
                play.save()

            all_comparisons.append(deepcopy(robot_comparison))




        return all_comparisons
