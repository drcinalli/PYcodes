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

import django
# do this before importing pylab or pyplot
# import matplotlib
# matplotlib.use('Agg')
# from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
# from matplotlib.figure import Figure
# import matplotlib.pylab as plt
# import matplotlib.animation as animation




class InterfaceGA:
    #class variables
    description = "This is the Genetic Algorithm INTERFACE class of the problem"
    author = "Daniel Cinalli"


    #Start Evolution
    def fakeStartEvolution(self, name,date,type ,description, flag, num_population, num_robots, num_levels, num_gen,
                           block_size, gen_threshold, actual_gen, player, mutation, cross):
        x=1

    def StartEvolution(self, name,date,type ,description, flag, num_population, num_robots, num_levels, num_gen,
                       block_size, gen_threshold, actual_gen, player, mutation, cross, wrld, first_loop):



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
                         num_robots=num_robots, numLevels=num_levels, gen_threshold=gen_threshold, numMinPlayers=player, time_elapsed_end=0, first_loop=first_loop)
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

        #pareto from the generation 1 of this round
        #save front 1
        paretoX_gen1 =[]
        paretoY_gen1=[]
        for j in pareto:
            paretoX_gen1.append(j.fitness.values[0])
            paretoY_gen1.append(j.fitness.values[1])
        print "  generation: ",format(1, '03d')
        #save front
        #?????????


        #save points of pop 1
        pointsX =[]
        pointsY=[]
        all_x=[]
        all_y=[]
        # fig1 = plt.figure()
        for j in pop:
            pointsX.append(j.fitness.values[0])
            pointsY.append(j.fitness.values[1])
        # scat = plt.scatter(pointsX,pointsY, c='r')
        all_x.append(pointsX)
        all_y.append(pointsY)


        #
        if exp.actual_gen + exp.first_loop <= exp.gen_threshold and 1 <= exp.numLevels:
            loop= exp.first_loop
        else:
            loop= exp.NGEN - exp.actual_gen


        #loop for BLOCK generations
        for i in xrange(loop):

            # Select the next generation individuals
            #ga.Selection(pop, my_world)
            ga.NSGASelection(pop, my_world)
            pareto.update(pop)

            #save points of evolution
            pointsX =[]
            pointsY=[]
            for j in pop:
                pointsX.append(j.fitness.values[0])
                pointsY.append(j.fitness.values[1])
            all_x.append(pointsX)
            all_y.append(pointsY)

            print "  generation: ", format(i+1, '03d')

        #
        #
        # def update_plot(i, data, pata, paretoX, paretoY,  paretoX_gen1, paretoY_gen1,  last, scat):
        #
        #     #scat.set_array(data[i])
        #     #plt.pause(0.5)
        #     plt.clf()
        #     plt.xlabel('$Cost')
        #     plt.ylabel('#Production')
        #     plt.title('the evolution')
        #
        #     #draw first pop
        #     # if i>0:
        #     #     scat = plt.scatter(data[0],pata[0], c='r')
        #
        #     if i>=(last-1):
        #         scat = plt.plot(paretoX_gen1,paretoY_gen1, c='r', label='initial iteration front')
        #         scat = plt.plot(paretoX,paretoY, c='b', label='final iteration front')
        #         scat = plt.scatter(data[loop-1],pata[loop-1], c='b')
        #
        #     else:
        #         #generations
        #         scat = plt.scatter(data[0],pata[0], c='r', label='initial individuals')
        #         scat = plt.scatter(data[i],pata[i], c='b', label='evolved individuals')
        #
        #     plt.legend()
        #
        #     return scat,
        #
        #
        #
        #
        #
        #
        # paretoX =[]
        # paretoY=[]
        # for j in pareto:
        #     paretoX.append(j.fitness.values[0])
        #     paretoY.append(j.fitness.values[1])
        #
        # ani = animation.FuncAnimation(fig1, update_plot, frames=xrange((loop)+8), fargs=( all_x,all_y, paretoX, paretoY, paretoX_gen1, paretoY_gen1,loop-1, scat),blit=False)
        # ani.save('polls/static/videos/bang__' + str(exp.id) + "__" + str(1) + '.mp4', fps=3, writer="ffmpeg", extra_args=['-vcodec', 'libx264'])


        exp.paretoX_gen1=str(paretoX_gen1)
        exp.paretoY_gen1=str(paretoY_gen1)


        #update actual generations with loop
        exp.actual_gen += loop

        #save Generations Block
        gen = Generation(experiment=exp, block=num_block, comparisons="to be collected", all_x=str(all_x), all_y=str(all_y))
        gen.save()

        #save population and Front into database
        with transaction.atomic ():
            count=1
            for ind in pop:
                population = Population(generation=gen, chromosome=str(ind[0]), index=count, chromosome_original=str(ind[0]))
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

        import numpy as np

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
        pop=[]
        pop_cp=[]

        #number of the next block of generations
        gen = Generation.objects.latest('id')
        num_block=gen.block + 1

        #fake population
        pop=ga.SetPopulationFake(my_world)
        pop_cp=ga.SetPopulationFake(my_world)

        #get the LAST Population
        count=0
        for ind in Population.objects.all().filter(generation=gen.id):
           pop[count][0] = literal_eval(ind.chromosome_original)
           pop_cp[count][0] = literal_eval(ind.chromosome)
           count +=1
        #POP ready

        #########################
        bot = Robot()
        pfront = self.GetFront()

        #get answers from plays
        answers = [[ 0 for i in range(2) ] for j in range(len(pfront)) ]

        plays = Play.objects.filter(play_experiment = exp, level = gen.block)
        for j in range(len(plays)):
            if plays[j].answer != -1:
                if plays[j].answer == 0:
                    correct = plays[j].chromosomeTwoIndex #verifica o indice da resposta correta
                    incorrect = plays[j].chromosomeOneIndex
                else:
                    correct = plays[j].chromosomeOneIndex #verifica o indice da resposta correta
                    incorrect = plays[j].chromosomeTwoIndex
                answers[correct][0] += 1
                answers[incorrect][1] += 1

        fake = answers

        #fake population
        pop_front=ga.SetPopulationFake(my_world)
        del pop_front[len(pfront):]

        #replace by front
        count=0
        for ind in pfront:
           pop_front[count][0] = ind
           count +=1

        # set FITNESS to the entire front
        fitnesses = ga.GetFitness(pop_front)
        ga.AttachFitness(pop_front,fitnesses)
        #POP FRONT ready

        #print pop
        #create list of fitness
        front_fit=[]
        for i in pop_front:
            front_fit.append([i.fitness.values[0],i.fitness.values[1]])

        #print front_fit
        fakefit=front_fit

        fake_array = []
        count=0
        for i in fake:
            #how many votes on i
            for j in xrange(i[0]):
                fake_array.append([fakefit[count][0]])
            count += 1

        t = np.array(fake_array, np.int32)
        #t.reshape(len(t),2).shape
        result = bot.expectation_maximization(t, nbiter=1, epsilon=1)

        k1_mu = result['params'][0]['mu']
        k1_p = result['params'][0]['proba']
        k1_sigma = result['params'][0]['sigma']

        k2_mu = result['params'][1]['mu']
        k2_p = result['params'][1]['proba']
        k2_sigma = result['params'][1]['sigma']

        #first mean is always the PRO-COST
        if float(k1_mu) <= float(k2_mu):
            mean1 = float(k1_mu)
            variance = float(k1_sigma)
            sigma1 = np.sqrt(variance)
            p1= float(k1_p)

            mean2 = float(k2_mu)
            variance = float(k2_sigma)
            sigma2 = np.sqrt(variance)
            p2= float(k2_p)

        else:
            mean1 = float(k2_mu)
            variance = float(k2_sigma)
            sigma1 = np.sqrt(variance)
            p1= float(k2_p)

            mean2 = float(k1_mu)
            variance = float(k1_sigma)
            sigma2 = np.sqrt(variance)
            p2= float(k1_p)


        #POP ready
        #POP original READY
        #FRONT ready
        #MEANS ready


        #######################################
        #HERE: the fitness inc/dec
        # set FITNESS to the entire population
        my_world.mean1=mean1
        my_world.sigma1 = sigma1
        my_world.p1 = p1
        my_world.mean2=mean2
        my_world.sigma2=sigma2
        my_world.p2=p2


        #fork to regular approach and COIN approach


        #fitnesses = ga.GetFitnessPAIR(pop)

        ##########
        #ORIGINAL#
        ##########

        #start matplot
        # fig1 = plt.figure()

        #arrays
        pointsX =[]
        pointsY=[]
        all_x=[]
        all_y=[]
        pointsX_cp =[]
        pointsY_cp=[]
        all_x_cp=[]
        all_y_cp=[]


        #ok, the real population!!
        fitnesses = ga.GetFitness(pop)
        ga.AttachFitness(pop,fitnesses)
        #assign the crowding distance to the individuals
        ga.SetCrowdingDistance(pop)
        #discover again the LAST Pareto
        pareto=tools.ParetoFront()
        pareto.update(pop)

        #pareto from the generation 1 of this round
        #save front 1
        # paretoX_gen1 =[]
        # paretoY_gen1=[]
        # for j in pareto:
        #     paretoX_gen1.append(j.fitness.values[0])
        #     paretoY_gen1.append(j.fitness.values[1])
        # print "  generation: ",format(1, '03d')

        for j in pop:
            pointsX.append(j.fitness.values[0])
            pointsY.append(j.fitness.values[1])
        # scat = plt.scatter(pointsX,pointsY, c='r')
        all_x.append(pointsX)
        all_y.append(pointsY)

        #oto
        #pop_cp is the population to be used with INCREMENTAL fitness

        #ok, the INC population!!
        fitnesses = ga.GetFitness(pop_cp)
        ga.AttachFitness(pop_cp,fitnesses)


        #discover again the LAST Pareto
        pareto_cp=tools.ParetoFront()
        pareto_cp.update(pop_cp)

        #pareto from the generation 1 of this round
        #save front 1
        paretoX_gen1 =[]
        paretoY_gen1=[]
        for j in pareto_cp:
            paretoX_gen1.append(j.fitness.values[0])
            paretoY_gen1.append(j.fitness.values[1])
        print "  generation: ",format(1, '03d')

        for j in pop_cp:
            pointsX_cp.append(j.fitness.values[0])
            pointsY_cp.append(j.fitness.values[1])
        # scat = plt.scatter(pointsX_cp,pointsY_cp, c='r')
        all_x_cp.append(pointsX_cp)
        all_y_cp.append(pointsY_cp)

        #calculate the INC fitness back
        fitnesses = ga.GetFitnessPAIR(pop_cp)
        ga.AttachFitness(pop_cp,fitnesses)
        # #assign the crowding distance to the individuals
        ga.SetCrowdingDistance(pop_cp)

        #create a new front only to the actual COIN solution... so I can check the real final populaion front, without old individuals from previous front
        #pareto_coin=tools.ParetoFront()




        #
        if exp.actual_gen + exp.block_size <= exp.gen_threshold and num_block <= exp.numLevels:
            loop= exp.block_size
        else:
            loop= exp.NGEN - exp.actual_gen

        #loop for BLOCK generations
        for i in xrange(loop):

            # Select the next generation individuals
            ga.NSGASelection(pop, my_world)
            pareto.update(pop)


            #pop_cp receives the INC population of pop
            ga.NSGASelectionPAIR(pop_cp, my_world)

            #this is the front that contains the previous population
            pareto_cp.update(pop_cp)
            #this considers only the actual front from actual pop
            #pareto_coin.update(pop_cp)

            #save points of evolution
            pointsX =[]
            pointsY=[]
            pointsX_cp =[]
            pointsY_cp=[]
            for j in xrange(len(pop)):
                pointsX.append(pop[j].fitness.values[0])
                pointsY.append(pop[j].fitness.values[1])
                pointsX_cp.append(pop_cp[j].fitness.values[0])
                pointsY_cp.append(pop_cp[j].fitness.values[1])
            all_x.append(pointsX)
            all_y.append(pointsY)
            all_x_cp.append(pointsX_cp)
            all_y_cp.append(pointsY_cp)


            fitnesses = ga.GetFitnessPAIR(pop_cp)
            ga.AttachFitness(pop_cp,fitnesses)
            # #assign the crowding distance to the individuals
            ga.SetCrowdingDistance(pop_cp)

            print "  generation: ", format(i+1, '03d')






        #####
        #MP4#
        #####
        #first generation
        expX= literal_eval(exp.paretoX_gen1)
        expY= literal_eval(exp.paretoY_gen1)
        #def update_plot(i, data, pata, paretoX, paretoY,  paretoX_gen1, paretoY_gen1, paretoX_cp, paretoY_cp, data_cp, pata_cp,  last, scat):
        # def update_plot(i, data, pata,last):
        #
        #     #scat.set_array(data[i])
        #     #plt.pause(0.5)
        #     plt.clf()
        #     plt.xlabel('$Cost')
        #     plt.ylabel('#Production')
        #     plt.title('the evolution')
        #     #draw first pop
        #
        #     #plt.axvline(x=mean1,  linewidth=2, color='g',  ls='--')
        #     #plt.axhline(y=(mean2*-1),  linewidth=2, color='g',  ls='--')
        #     plt.text(mean1-15,-240-15,'1-D cost-mean')
        #     plt.plot(mean1, -240,  'g^', markersize=8)
        #
        #     # plt.text(60-15, (mean2+15)*-1,'1-D prod-mean')
        #     # plt.plot(60, mean2*-1,  'g^', markersize=8)
        #     plt.text(mean2-15,-240-15,'1-D cost-mean')
        #     plt.plot(mean2, -240,  'g^', markersize=8)
        #
        #     # plt.plot(10, 10,  'g^', markersize=8)
        #
        #     # if i>0 and i<(last-1):
        #     #     #original generation
        #     #     scat = plt.scatter(data[0],pata[0], c='r', label='initial individuals')
        #
        #     if i>=(last-1):
        #
        #         #final
        #
        #         #first generation
        #         scat = plt.plot(expX, expY, c='k', label='first generation front')
        #         scat = plt.scatter(expX, expY, c='k')
        #
        #         #fronts
        #         scat = plt.plot(paretoX_gen1,paretoY_gen1, c='r', label='initial iteration front')
        #         scat = plt.plot(paretoX,paretoY, c='b', label='final iteration front')
        #
        #         scat = plt.scatter(data[loop-1],pata[loop-1], c='b')
        #         scat = plt.scatter(data[0],pata[0], c='r')
        #
        #         #COIN
        #         scat = plt.plot(paretoX_cp,paretoY_cp, c='y', label='COIN final iteration front')
        #         scat = plt.scatter(all_x_cp[loop-1],all_y_cp[loop-1], c='y')
        #
        #
        #     else:
        #         #generations
        #         scat = plt.scatter(data[0],pata[0], c='r', label='initial individuals')
        #         scat = plt.scatter(data[i],pata[i], c='b', label='evolved individuals')
        #
        #         #COIN
        #         scat = plt.scatter(all_x_cp[i],all_y_cp[i], c='y', label='evolved individuals')
        #
        #     plt.legend()
        #
        #     return scat,
        #
        # paretoX =[]
        # paretoY=[]
        # for j in pareto:
        #     paretoX.append(j.fitness.values[0])
        #     paretoY.append(j.fitness.values[1])
        #
        # paretoX_cp =[]
        # paretoY_cp=[]
        # for j in pareto_cp:
        #     paretoX_cp.append(j.fitness.values[0])
        #     paretoY_cp.append(j.fitness.values[1])
        #
        # # paretoX_cp =[]
        # # paretoY_cp=[]
        # # for j in pareto_coin:
        # #     paretoX_cp.append(j.fitness.values[0])
        # #     paretoY_cp.append(j.fitness.values[1])
        #
        # #ani = animation.FuncAnimation(fig1, update_plot, frames=xrange((loop-1)+8), fargs=( all_x,all_y, paretoX, paretoY, paretoX_gen1, paretoY_gen1, paretoX_cp,paretoY_cp, all_x_cp,all_y_cp ,loop-1, scat),blit=False)
        # ani = animation.FuncAnimation(fig1, update_plot, frames=xrange((loop)+8), fargs=(all_x,all_y, loop-1),blit=False)
        # ani.save('polls/static/videos/bang__' + str(exp.id) + "__" + str(num_block) + '.mp4', fps=3, writer="ffmpeg", extra_args=['-vcodec', 'libx264'])
        #
        #


        #update actual generations with block_size
        exp.actual_gen += loop

        #save Generations Block
        gen = Generation(experiment=exp, block=num_block, comparisons="not necessary anymore",all_x=str(all_x),
                         all_y=str(all_y),all_x_cp=str(all_x_cp), all_y_cp=str(all_y_cp), mean_1=mean1, sigma_1=sigma1,p_1=p1,
                          mean_2=mean2, sigma_2=sigma2,p_2=p2)
        gen.save()



        #save population and Front into database
        with transaction.atomic ():
            count=1
            for ind in xrange(len(pop_cp)):
                population = Population(generation=gen, chromosome=str(pop_cp[ind][0]), index=count, chromosome_original=str(pop[ind][0]))
                population.save()
                count +=1
            count=1
            for ind in pareto_cp:
                pfront = PFront(generation=gen, chromosome=str(ind[0]), index=count)
                pfront.save()
                count +=1


        #set the next status according the number of generations
        if exp.actual_gen >= exp.NGEN:
            exp.flag = 'F'
        else:
            exp.flag = 'R'

        exp.save()

        print "video ready!"
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

    #get Fitness of the Front
    def GetFitnessFront(self):

        #get last front
        pfront=[]
        pfront = self.GetFront()

        #get last ID of the Experiment
        exp = Experiment.objects.latest('id')

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

        #fake population
        pop_front=ga.SetPopulationFake(my_world)
        del pop_front[len(pfront):]

        #replace population by front
        count=0
        for ind in pfront:
           pop_front[count][0] = ind
           count +=1

        #front ready (right structure) but with no fitness

        # set FITNESS to the entire front
        fitnesses = ga.GetFitness(pop_front)
        ga.AttachFitness(pop_front,fitnesses)
        #POP FRONT ready WITH fitness

        #print pop
        #create list of fitness
        front_fit=[]
        for i in pop_front:
            front_fit.append([i.fitness.values[0],i.fitness.values[1]])

        #print front_fit

        return front_fit


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
            k=Play.objects.all().filter(play_experiment=exp.id, level=1, play_player__type = 'C').values('play_player').distinct()

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
                k=robot_lst[i].objective1_pref
                ans = bot.TakeDecision(k, front_fit[a], front_fit[b])

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

    def seeRobotGaussians(self):

        import numpy as np
        import matplotlib.pyplot as plt
        import matplotlib.mlab as mlab

        fig=Figure()



        bot = Cluster()

        pop = self.GetFront()



        ##########################################

        #get last ID of the Experiment
        exp = Experiment.objects.latest('id')
        gen = Generation.objects.latest('id')

        #get all plays
        # plays = Play.objects.filter(play_experiment=exp, level=1)
        # #check users ... put in the list
        # game_usr=[]
        # for i in plays:
        #     if i.play_player not in game_usr:
        #         game_usr.append(i.play_player)
        #


        #get answers from plays
        answers = [[ 0 for i in range(2) ] for j in range(len(pop)) ]

        plays = Play.objects.filter(play_experiment = exp, level = gen.block)
        for j in range(len(plays)):
            if plays[j].answer != -1:
                if plays[j].answer == 0:
                    correct = plays[j].chromosomeTwoIndex #verifica o indice da resposta correta
                    incorrect = plays[j].chromosomeOneIndex
                else:
                    correct = plays[j].chromosomeOneIndex #verifica o indice da resposta correta
                    incorrect = plays[j].chromosomeTwoIndex
                answers[correct][0] += 1
                answers[incorrect][1] += 1

        fake = answers

        #get front fitness
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

        #fake population
        pop_front=ga.SetPopulationFake(my_world)
        del pop_front[len(pop):]

        #replace by front
        count=0
        for ind in pop:
           pop_front[count][0] = ind
           count +=1

        # set FITNESS to the entire front
        fitnesses = ga.GetFitness(pop_front)
        ga.AttachFitness(pop_front,fitnesses)

        #print pop
        #create list of fitness
        front_fit=[]
        for i in pop_front:
            front_fit.append([i.fitness.values[0],i.fitness.values[1]])

        #print front_fit
        fakefit=front_fit

        #fake fitness
        # fakefit= [(73.868159279972645, -23.0), (77.431658819361502, -35.0), (78.923887369973485, -46.0), (78.923887369973485, -46.0),
        #           (82.901600471735435, -58.0), (83.053405777305727, -69.0), (85.9976776873049, -70.0), (87.053405777305727, -81.0),
        #           (91.053405777305727, -93.0), (95.053405777305727, -105.0), (99.064495102245985, -116.0), (103.06449510224598, -128.0),
        #           (107.06449510224598, -140.0), (114.41421356237309, -151.0), (114.41421356237309, -151.0), (118.41421356237309, -163.0),
        #           (118.41421356237309, -163.0), (118.41421356237309, -163.0), (118.41421356237309, -163.0), (122.41421356237309, -175.0),
        #           (130.82842712474618, -186.0), (130.82842712474618, -186.0), (134.82842712474618, -198.0), (134.82842712474618, -198.0),
        #           (138.82842712474618, -210.0), (138.82842712474618, -210.0)]

        #get robots .85
        #fake = [[74, 15], [74, 11], [69, 16], [73, 21], [68, 28], [60, 27], [69, 33], [65, 27], [50, 40], [45, 35], [66, 46], [59, 46], [39, 41], [49, 55], [44, 52], [44, 39], [43, 64], [30, 45], [32, 63], [31, 63], [23, 69], [27, 69], [15, 76], [12, 77], [23, 61], [16, 81]]
        #########################################

        fake_array = []
        count=0
        for i in fake:
            #how many votes on i
            for j in xrange(i[0]):
                fake_array.append([fakefit[count][0]])
            count += 1

        t = np.array(fake_array, np.int32)
        #t.reshape(len(t),2).shape
        result = bot.expectation_maximization(t, nbiter=3, epsilon=1)

        k1_mu = result['params'][0]['mu']
        k1_p = result['params'][0]['proba']
        k1_sigma = result['params'][0]['sigma']

        k2_mu = result['params'][1]['mu']
        k2_p = result['params'][1]['proba']
        k2_sigma = result['params'][1]['sigma']


        mean1 = float(k1_mu)
        variance = float(k1_sigma)
        sigma1 = np.sqrt(variance)

        mean2 = float(k2_mu)
        variance = float(k2_sigma)
        sigma2 = np.sqrt(variance)

        print mean1
        print mean2
        if mean1<=mean2:
            x = np.linspace(50,180,100)
            #plt.subplot(2, 1, 1)
            plt=fig.add_subplot(111)
            plt.plot(x,mlab.normpdf(x,mean1,sigma1),'-o',color='b')

            x = np.linspace(50,180,100)
            plt.plot(x,mlab.normpdf(x,mean2,sigma2),'-o', color='g')

            #azul e verde

        else:

            x = np.linspace(50,180,100)
            #plt.subplot(2, 1, 1)
            plt=fig.add_subplot(111)
            plt.plot(x,mlab.normpdf(x,mean1,sigma1),'-o',color='g')

            x = np.linspace(50,180,100)
            plt.plot(x,mlab.normpdf(x,mean2,sigma2),'-o', color='b')

            #verde e azul

        plt.hist(t[:len(result['clusters'][0])],histtype='stepfilled', bins=20, normed=True,color='#325ADB', label='Uniform')
        plt.hist(t[len(result['clusters'][0]):],histtype='stepfilled', bins=20, normed=True,color='#2FBA4F', label='Uniform')



        #
        # plt.hist(t[:len(result['clusters'][0])],histtype='stepfilled', bins=20, normed=True,color='#325ADB', label='Uniform')
        # plt.hist(t[len(result['clusters'][0]):],histtype='stepfilled', bins=20, normed=True,color='#2FBA4F', label='Uniform')



        #plt.title("PeloCano - Gaussian Mixture")
        #plt.xlabel("Value")
        #plt.ylabel("Frequency")
        #plt.show()

        # mean_full=np.mean(t)
        # variance_full = np.std(t)
        # sigma_full = np.sqrt(variance_full)
        #
        # plt.subplot(2, 1, 2)
        # plt=fig.add_subplot(212)
        # plt.plot(x,mlab.normpdf(x,mean_full,sigma_full),'-o',color='r')
        # plt.plot(x,mlab.normpdf(x,mean2,sigma),'-o', color='g')
        # plt.plot(x,mlab.normpdf(x,mean1,sigma),'-o',color='b')
        #
        #

        #plt.title("PeloCano - Gaussian Mixture")
        #plt.xlabel("Value")
        #plt.ylabel("Frequency")
        # #plt.show()
        #
        # buffer = StringIO.StringIO()
        # canvas = pylab.get_current_fig_manager().canvas
        # canvas.draw()
        #
        # graphIMG=PIL.Image.fromstring("RGB", canvas.get_width_height(), canvas.tostring_RGB())
        # graphIMG.save(buffer, "PNG")
        # pylab.close()

        # fig=Figure()
        # ax=fig.add_subplot(111)
        # x=[]
        # y=[]
        # now=datetime.datetime.now()
        # delta=datetime.timedelta(days=1)
        # for i in range(10):
        #     x.append(now)
        #     now+=delta
        #     y.append(random.randint(0, 1000))
        # ax.plot_date(x, y, '-')
        # ax.xaxis.set_major_formatter(DateFormatter('%Y-%m-%d'))
        # fig.autofmt_xdate()
        # canvas=FigureCanvas(fig)
        # response=django.http.HttpResponse(content_type='image/png')
        # canvas.print_png(response)
        # return response

        fig.autofmt_xdate()
        canvas=FigureCanvas(fig)
        response=django.http.HttpResponse(content_type='image/png')
        canvas.print_png(response)
        return response

