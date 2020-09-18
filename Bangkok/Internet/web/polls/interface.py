from world import World
from emoa import GA
from robot import Robot
from cluster import Cluster
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
import matplotlib
matplotlib.use('Agg')
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pylab as plt
import matplotlib.animation as animation
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
from operator import itemgetter




class InterfaceGA:
    #class variables
    description = "This is the Genetic Algorithm INTERFACE class of the problem"
    author = "Daniel Cinalli"


    #Start Evolution
    def fakeStartEvolution(self, name,date,type ,description, flag, num_population, num_robots, num_levels, num_gen,
                           block_size, gen_threshold, actual_gen, player, mutation, cross):
        x=1

    def StartEvolution(self, name,date,type ,description, flag, num_population, num_robots, num_levels, num_gen,
                       block_size, gen_threshold, actual_gen, player, mutation, cross, wrld, first_loop, ropoints,
                       freeK, moea_alg, tour, vote,type_prob):



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
                         num_robots=num_robots, numLevels=num_levels, gen_threshold=gen_threshold, numMinPlayers=player,
                         time_elapsed_end=0, first_loop=first_loop, bots_points=ropoints, freeK=freeK, moea_alg=moea_alg,
                         tour=tour, vote=vote, type_prob=type_prob)
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
        #print "  generation: ",format(1, '03d')
        #save front
        #?????????


        #save points of pop 1
        pointsX =[]
        pointsY=[]
        all_x=[]
        all_y=[]
        fig1 = plt.figure()
        for j in pop:
            pointsX.append(j.fitness.values[0])
            pointsY.append(j.fitness.values[1])
        scat = plt.scatter(pointsX,pointsY, c='r')
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

            #save points of evolution
            pointsX =[]
            pointsY=[]
            for j in pop:
                pointsX.append(j.fitness.values[0])
                pointsY.append(j.fitness.values[1])
            all_x.append(pointsX)
            all_y.append(pointsY)

            print "  generation: ", format(i+1, '03d')

        pareto = tools.ParetoFront()
        pareto.update(pop)



        def update_plot(i, data, pata, paretoX, paretoY,  paretoX_gen1, paretoY_gen1,  last, scat):

            #scat.set_array(data[i])
            #plt.pause(0.5)
            plt.clf()
            plt.xlabel('$Cost')
            plt.ylabel('#Production')
            plt.title('the evolution')

            #draw first pop
            # if i>0:
            #     scat = plt.scatter(data[0],pata[0], c='r')

            if i>=(last-1):
                scat = plt.plot(paretoX_gen1,paretoY_gen1, c='r', label='initial iteration front')
                scat = plt.plot(paretoX,paretoY, c='b', label='final iteration front')
                scat = plt.scatter(data[loop-1],pata[loop-1], c='b')

            else:
                #generations
                scat = plt.scatter(data[0],pata[0], c='r', label='initial individuals')
                scat = plt.scatter(data[i],pata[i], c='b', label='evolved individuals')

            plt.legend()

            return scat,






        paretoX =[]
        paretoY=[]
        for j in pareto:
            paretoX.append(j.fitness.values[0])
            paretoY.append(j.fitness.values[1])

        ani = animation.FuncAnimation(fig1, update_plot, frames=xrange((loop)+8), fargs=( all_x,all_y, paretoX, paretoY, paretoX_gen1, paretoY_gen1,loop-1, scat),blit=False)
        ani.save('polls/static/videos/bang__' + str(exp.id) + "__" + str(1) + '.mp4', fps=3, writer="ffmpeg", extra_args=['-vcodec', 'libx264'])


        print "video ready!"

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
                #ind[0]=[4.5, 3.0]
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
        if exp.vote == 'P':
            self.RunRobots(exp, num_block, ga, my_world,pareto)
        else:
            self.RunRobotsAll(exp, num_block, ga, my_world,pareto, pop)



        return exp


    def StartEvolutionBench(self, name,date,type ,description, flag, num_population, num_robots, num_levels, num_gen,
                       block_size, gen_threshold, actual_gen, player, mutation, cross, first_loop, ropoints,
                       freeK, moea_alg, tour, vote,type_prob):


        gameworld = GameWorld.objects.latest('id')
        my_world = World(gameworld.m,gameworld.n)
        #not my fault ... I implemented the Benchmanrk tests after the Resource Dist.... so... this foreign key is blocked.
        #and I do not want to create a separate structure to the tests...

        #declare Genetic Algorithm for the problem
        ga = GA(my_world,cross, mutation, num_gen, num_population,type_prob, moea_alg)
        pareto=tools.ParetoFront()
        pop=[]

        #create a new exp
        now=datetime.datetime.now()
        exp = Experiment( world=gameworld, name=name, date=date, block_size=block_size, start=now, type=type, description=description,
                         flag=flag, actual_gen=actual_gen,  CXPB=cross, MUTPB=mutation, NGEN=num_gen, NPOP=num_population,
                         num_robots=num_robots, numLevels=num_levels, gen_threshold=gen_threshold, numMinPlayers=player,
                         time_elapsed_end=0, first_loop=first_loop, bots_points=ropoints, freeK=freeK, moea_alg=moea_alg,
                         tour=tour, vote=vote, type_prob=type_prob)
        exp.save()

        #set NEW Population
        pop=ga.SetPopulation()


        #set block for the generation block table
        num_block=1

        # set FITNESS to the entire population
        fitnesses = ga.GetFitness(pop)
        ga.AttachFitness(pop,fitnesses)

        #if NSGA-II
        if moea_alg=='N':
            #assign the crowding distance to the individuals
            ga.SetCrowdingDistance(pop)
        elif moea_alg=='P':
            archive = ga.SetPopulationFakeBench(my_world, type_prob)
            fitnesses = ga.GetFitness(archive)
            ga.AttachFitness(archive,fitnesses)
        elif moea_alg=='S':
            #assign the Hypervolume initial measure to SMS-EMOA
            ga.SetHyperValue(pop)


        #discover again the LAST Pareto
        pareto.update(pop)

        #pareto from the generation 1 of this round
        #save front 1
        paretoX_gen1 =[]
        paretoY_gen1=[]
        for j in pareto:
            paretoX_gen1.append(j.fitness.values[0])
            paretoY_gen1.append(j.fitness.values[1])
        #print "  generation: ",format(1, '03d')

        #save points of pop 1
        pointsX =[]
        pointsY=[]
        all_x=[]
        all_y=[]
        fig1 = plt.figure()
        for j in pop:
            pointsX.append(j.fitness.values[0])
            pointsY.append(j.fitness.values[1])
        #scat = plt.scatter(pointsX,pointsY, c='r')
        all_x.append(pointsX)
        all_y.append(pointsY)


        #if (first loop is smaller) than the threshold, then do the first loop
        if exp.actual_gen + exp.first_loop <= exp.gen_threshold and 1 <= exp.numLevels:
            loop= exp.first_loop
        else:
            loop= exp.NGEN - exp.actual_gen

        #Tirar
        partialn=[]
        partial_rejn=[]


        #loop for BLOCK generations
        for i in xrange(loop):

            # Select the next generation individuals
            #ga.Selection(pop, my_world)

            #if NSGA-II
            if moea_alg=='N':
                ga.NSGASelection_Pure(pop, partialn, partial_rejn)

                #save points of evolution
                pointsX =[]
                pointsY=[]
                for j in pop:
                    pointsX.append(j.fitness.values[0])
                    pointsY.append(j.fitness.values[1])
                all_x.append(pointsX)
                all_y.append(pointsY)

            #SPEA2
            elif moea_alg=='P':
                ga.SPEA2Selection_Pure(pop, archive, partialn, partial_rejn)

                #save points of evolution
                pointsX =[]
                pointsY=[]
                for j in archive:
                    pointsX.append(j.fitness.values[0])
                    pointsY.append(j.fitness.values[1])
                all_x.append(pointsX)
                all_y.append(pointsY)

            #if SMS-EMOA
            elif moea_alg=='S':
                ga.SMSSelection_Pure(pop, partialn, partial_rejn)

                #save points of evolution
                pointsX =[]
                pointsY=[]
                for j in pop:
                    pointsX.append(j.fitness.values[0])
                    pointsY.append(j.fitness.values[1])
                all_x.append(pointsX)
                all_y.append(pointsY)

            print "  generation: ", format(i+1, '03d')

        #if SPEA2, take the archive as the Pareto Front
        if moea_alg=='P':
            pop = archive

        pareto=tools.ParetoFront()
        pareto.update(pop)


        def update_plot(i, data, pata, paretoX, paretoY,  paretoX_gen1, paretoY_gen1,  last, scat):

            #scat.set_array(data[i])
            #plt.pause(0.5)
            plt.clf()
            plt.xlabel('f1')
            plt.ylabel('f2')
            plt.title('the evolution')

            #draw first pop
            # if i>0:
            #     scat = plt.scatter(data[0],pata[0], c='r')

            if i>=(last-1):
                #scat = plt.plot(paretoX_gen1,paretoY_gen1, c='r', label='initial iteration front')
                scat = plt.plot(paretoX,paretoY, c='b', label='final iteration front')
                scat = plt.scatter(data[loop-1],pata[loop-1], c='b')

            else:
                #generations
                #scat = plt.scatter(data[0],pata[0], c='r', label='initial individuals')
                scat = plt.scatter(data[i],pata[i], c='b', label='evolved individuals')

            plt.legend()

            return scat,






        paretoX =[]
        paretoY=[]
        for j in pareto:
            paretoX.append(j.fitness.values[0])
            paretoY.append(j.fitness.values[1])

        #ani = animation.FuncAnimation(fig1, update_plot, frames=xrange((loop)+8), fargs=( all_x,all_y, paretoX, paretoY, paretoX_gen1, paretoY_gen1,loop-1, scat),blit=False)
        #ani.save('polls/static/videos/bang__' + str(exp.id) + "__" + str(1) + '.mp4', fps=3, writer="ffmpeg", extra_args=['-vcodec', 'libx264'])
        scat = plt.scatter(paretoX,paretoY, c='r')

        pp = PdfPages('polls/static/fronts/front_' + str(exp.id) + '__' + str(i) + '.pdf')
        pp.savefig(fig1)
        pp.close()

        print "video ready!"


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
                #burrada, I do not know how to do it!!
                #map(float, literal_eval(str(map(str, ind))))
                #str(map(float, literal_eval(str(map(str, ind)))))
                # temp="["
                # for i in ind:
                #     temp = temp + str(i) + ", "
                # temp = temp + "]"
                temp=[]
                for i in ind:
                    temp.append(i)

                population = Population(generation=gen, chromosome=str(temp), index=count, chromosome_original=str(temp))
                population.save()
                count +=1
            count=1
            for ind in pareto:

                # temp="["
                # for i in ind:
                #     temp = temp + str(i) + ", "
                # temp = temp + "]"
                temp=[]
                for i in ind:
                    temp.append(i)

                pfront = PFront(generation=gen, chromosome=str(temp), index=count)
                pfront.save()
                count +=1


        #set the next status according the number of generations
        if exp.actual_gen >= exp.NGEN:
            exp.flag = 'F'
        else:
            exp.flag = 'R'

        exp.save()

        #run robots
        if exp.vote == 'P':
            self.RunRobots(exp, num_block, ga, my_world,pareto, type_prob)
        else:
            self.RunRobotsAll(exp, num_block, ga, my_world,pareto, pop, type_prob)



        return exp


    #Start Evolution
    def fakeContinueEvolution(self):
        x=1

    def ContinueEvolution(self):

        import numpy as np

        #get last ID of the Experiment
        exp = Experiment.objects.latest('id')

        #get World config
        wrl = exp.world
        gameworld = GameWorld.objects.get(id=wrl.id)
        my_world = World(gameworld.m,gameworld.n)
        my_world.MaxArea(gameworld.max_areas)
        my_world.MaxUnits(gameworld.max_units)
        my_world.Production(gameworld.prod_unit0,gameworld.prod_unit1)
        my_world.Costs(gameworld.cost_gateway,gameworld.cost_unit0,gameworld.cost_unit1)
        my_world.experimentTYPE = exp.type

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
        pfront = self.GetFront(problem=exp.type_prob)

        #check candidate option: front or front and population
        if exp.vote == 'P':
            #get answers from plays
            answers = [[ 0 for i in range(2) ] for j in range(len(pfront)) ]
        else:
            #get answers from plays
            answers = [[ 0 for i in range(2) ] for j in range(len(pop_cp)) ]

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

        #fake = answers

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


        #ok, the real population!!
        fitnesses = ga.GetFitness(pop)
        ga.AttachFitness(pop,fitnesses)
        #assign the crowding distance to the individuals
        ga.SetCrowdingDistance(pop)
        #discover again the LAST Pareto
        pareto=tools.ParetoFront()
        pareto.update(pop)

        #POP ready

        #ok, the  population to be tested!!
        fitnesses = ga.GetFitness(pop_cp)
        ga.AttachFitness(pop_cp,fitnesses)
        # #assign the coin distance to the individuals
        partial=[]
        partial_rej=[]
        ga.SetCOINDistance(pop_cp, partial, partial_rej)
        #discover again the LAST Pareto
        pareto_cp=tools.ParetoFront()
        pareto_cp.update(pop_cp)


        #check candidate option: front or front and population
        if exp.vote == 'P':


            #print pop
            #create list of fitness
            front_fit=[]
            for i in pop_front:
                front_fit.append([i.fitness.values[0],i.fitness.values[1]])

            #print front_fit
            fakefit=front_fit

            #here I create the list of choosen points (fitness)
            fake_array = []
            lst_points2D = []
            count=0
            for i in answers:
                #how many votes on i
                for j in xrange(i[0]):
                    fake_array.append([fakefit[count][0]])
                    lst_points2D.append([fakefit[count][0] , fakefit[count][1]])
                count += 1
        else:

            #print pop
            #create list of fitness
            pop_fit=[]
            for i in pop_cp:
                pop_fit.append([i.fitness.values[0],i.fitness.values[1]])

            #print front_fit
            fakefit=pop_fit

            #here I create the list of choosen points (fitness)
            fake_array = []
            lst_points2D = []
            count=0
            for i in answers:
                #how many votes on i
                for j in xrange(i[0]):
                    fake_array.append([fakefit[count][0]])
                    lst_points2D.append([fakefit[count][0] , fakefit[count][1]])
                count += 1




        ##########################################################################
        #POP, PARETO FRONT, FITNESS FRONT and List of POINTS (fitness)  is  ready#
        ##########################################################################
        clusteOBJ = Cluster()

        t = np.array(lst_points2D, np.int32)

        #check possible number of Components for fixed number of components types
        num_clusters = 2 #my default
        if exp.bots_points == 'A':
            num_clusters = 1
        elif exp.bots_points == 'B':
            num_clusters = 2
        elif exp.bots_points == 'C':
            num_clusters = 3


        #initialize
        kmm=None
        best_gmm=None

        #check Experiment type
        if exp.type == 'B':

            #2-D bivariate normal gaussian mixture

            #if fixed number of clusters
            if not exp.freeK and exp.bots_points != 'D':

                #run EM
                best_gmm = clusteOBJ.EM(t, num_clusters)

            else:
                #free k clusters or many reference points for robots

                #run EM free Components K
                best_gmm = clusteOBJ.EMfreeK(t)
                num_clusters = best_gmm.n_components


            #save results to local vars for database
            #save results to my_world object
            means = str(best_gmm.means_.tolist())
            covar = str(best_gmm.covars_.tolist())
            weights = str(best_gmm.weights_.tolist())
            my_world.means=best_gmm.means_
            my_world.covar=best_gmm.covars_
            my_world.weights=best_gmm.weights_



        elif exp.type == 'C':

            #kmeans
            #if fixed number of clusters
            if not exp.freeK and exp.bots_points != 'D':

                #run Kmeans
                kmm = clusteOBJ.Kmeans(t, num_clusters)
                #kmm.cluster_centers_
            else:
                #free k clusters or many reference points for robots

                #run Kmeans free Components K
                #kmm.cluster_centers_
                kmm = clusteOBJ.KmeansfreeK(t)
                num_clusters = kmm.n_clusters

            #save results to local vars for database
            #save results to my_world object
            centroids = str(kmm.cluster_centers_.tolist())
            my_world.centroids=kmm.cluster_centers_.tolist()

        else:

            #type: 1-D Gaussian over the objective 1 (cost) only
            #here I fixed only at one objective ... and 2 componentes (cost x anti-cost)
            t = np.array(fake_array, np.int32)
            #t.reshape(len(t),2).shape
            # result = clusteOBJ.expectation_maximization(t, nbclusters=2, nbiter=1, epsilon=1)
            # k1_mu = result['params'][0]['mu']
            # k1_p = result['params'][0]['proba']
            # k1_sigma = result['params'][0]['sigma']
            #
            # k2_mu = result['params'][1]['mu']
            # k2_p = result['params'][1]['proba']
            # k2_sigma = result['params'][1]['sigma']

            result =  clusteOBJ.EM(t, 2)

            k1_mu = result.means_[0]
            k1_p = result.weights_[0]
            k1_sigma = result.covars_[0]

            k2_mu = result.means_[1]
            k2_p = result.weights_[1]
            k2_sigma = result.covars_[1]

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

            # set the means to the entire population
            my_world.mean1=mean1
            my_world.sigma1 = sigma1
            my_world.p1 = p1
            my_world.mean2=mean2
            my_world.sigma2=sigma2
            my_world.p2=p2


        ##########################
        #POP ready               #
        #POP original READY      #
        #FRONT ready             #
        #MEANS ready             #
        #COVARS ready            #
        #List of POINTS (fitness)#
        ##########################

        ##########
        #ORIGINAL#
        ##########

        #start matplot
        fig1 = plt.figure()

        #arrays
        pointsX =[]
        pointsY=[]
        all_x=[]
        all_y=[]
        pointsX_cp =[]
        pointsY_cp=[]
        all_x_cp=[]
        all_y_cp=[]

        #partial points
        partial_x=[]
        partial_y=[]
        partialrej_x=[]
        partialrej_y=[]
        for j in partial:
            partial_x.append(j.fitness.values[0])
            partial_y.append(j.fitness.values[1])
        # scat = plt.scatter(pointsX,pointsY, c='r')
        allpartial_x=[]
        allpartial_y=[]
        allpartial_x.append(partial_x)
        allpartial_y.append(partial_y)
        for j in partial_rej:
            partialrej_x.append(j.fitness.values[0])
            partialrej_y.append(j.fitness.values[1])
        # scat = plt.scatter(pointsX,pointsY, c='r')
        allpartialrej_x=[]
        allpartialrej_y=[]
        allpartialrej_x.append(partial_x)
        allpartialrej_y.append(partial_y)



        #points for the ORIGINAL population
        for j in pop:
            pointsX.append(j.fitness.values[0])
            pointsY.append(j.fitness.values[1])
        # scat = plt.scatter(pointsX,pointsY, c='r')
        all_x.append(pointsX)
        all_y.append(pointsY)



        #ok, the TEST population!!
        #pop_cp is the population to be used with COIN or Robot means


        #pareto from the generation 1 of this round
        paretoX_gen1 =[]
        paretoY_gen1=[]
        for j in pareto_cp:
            paretoX_gen1.append(j.fitness.values[0])
            paretoY_gen1.append(j.fitness.values[1])
        print "  generation: ",format(1, '03d')

        #points for the TEST population
        for j in pop_cp:
            pointsX_cp.append(j.fitness.values[0])
            pointsY_cp.append(j.fitness.values[1])
        scat = plt.scatter(pointsX_cp,pointsY_cp, c='r')
        all_x_cp.append(pointsX_cp)
        all_y_cp.append(pointsY_cp)




        #control the loop
        if exp.actual_gen + exp.block_size <= exp.gen_threshold and num_block <= exp.numLevels:
            loop= exp.block_size
        else:
            loop= exp.NGEN - exp.actual_gen

        #loop for BLOCK generations
        for i in xrange(loop):

            #ORIGINAL
            # Select the next generation individuals
            ga.NSGASelection(pop, my_world)

            #partial
            partial=[]
            partial_rej=[]

            ga.NSGASelectionCOIN(pop_cp, my_world, ga, exp.tour, exp.type, best_gmm, kmm, partial, partial_rej, problem=exp.type_prob)

            #save points of evolution to ORIGINAL and TEST
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

            #partial points
            partial_x=[]
            partial_y=[]
            for j in partial:
                partial_x.append(j.fitness.values[0])
                partial_y.append(j.fitness.values[1])
            # scat = plt.scatter(pointsX,pointsY, c='r')
            allpartial_x.append(partial_x)
            allpartial_y.append(partial_y)

            partialrej_x=[]
            partialrej_y=[]
            for j in partial_rej:
                partialrej_x.append(j.fitness.values[0])
                partialrej_y.append(j.fitness.values[1])
            # scat = plt.scatter(pointsX,pointsY, c='r')
            allpartialrej_x.append(partial_x)
            allpartialrej_y.append(partial_y)


            #
            # fitnesses = ga.GetFitnessPAIR(pop_cp)
            # ga.AttachFitness(pop_cp,fitnesses)
            # # #assign the crowding distance to the individuals
            # ga.SetCrowdingDistance(pop_cp)

            print "  generation: ", format(i+1, '03d')



        #this is the front that contains the previous population
        pareto_cp=tools.ParetoFront()
        pareto=tools.ParetoFront()
        pareto_cp.update(pop_cp)
        pareto.update(pop)




        #####
        #MP4#
        #####
        #first generation
        expX= literal_eval(exp.paretoX_gen1)
        expY= literal_eval(exp.paretoY_gen1)
        #def update_plot(i, data, pata, paretoX, paretoY,  paretoX_gen1, paretoY_gen1, paretoX_cp, paretoY_cp, data_cp, pata_cp,  last, scat):
        def update_plot(i, data, pata,last):

            #scat.set_array(data[i])
            #plt.pause(0.5)
            plt.clf()
            plt.xlabel('$Cost')
            plt.ylabel('#Production')
            plt.title('the evolution')
            #draw first pop

            if exp.type == 'B':

                #2-D bivariate normal gaussian mixture
                for j in best_gmm.means_:
                    #plt.plot(j[0], j[1],  'gD', markersize=8)
                    cir1 = plt.Circle((j[0], j[1]),2,color='k',fill=False)
                    cir2 = plt.Circle((j[0], j[1]),4,color='g',fill=False)
                    cir3 = plt.Circle((j[0], j[1]),6,color='y',fill=False)
                    fig = plt.gcf()
                    fig.gca().add_artist(cir1)
                    fig.gca().add_artist(cir2)

            elif exp.type == 'C':

                #kmeans
                for j in kmm.cluster_centers_:
                    plt.plot(j[0], j[1],  'gD', markersize=8)

            else:
                plt.text(mean1-15,-240-15,'1-D cost-mean')
                plt.plot(mean1, -240,  'g^', markersize=8)
                plt.text(mean2-15,-240-15,'1-D cost-mean')
                plt.plot(mean2, -240,  'g^', markersize=8)

            if i>=(last-1):

                #final

                #first generation
                scat = plt.plot(expX, expY, c='k', label='first generation front')
                scat = plt.scatter(expX, expY, c='k')

                #fronts
                #scat = plt.plot(data[0],pata[0], c='c', label='initial regular front')
                scat = plt.plot(paretoX_gen1,paretoY_gen1, c='r', label='initial COIN front')
                scat = plt.plot(paretoX,paretoY, c='b', label='final regular front')

                scat = plt.scatter(data[loop-1],pata[loop-1], c='b')
                #scat = plt.scatter(data[0],pata[0], c='r')

                #COIN
                scat = plt.plot(paretoX_cp,paretoY_cp, c='y', label='final COIN front')
                scat = plt.scatter(all_x_cp[loop-1],all_y_cp[loop-1], c='y')

                #PARTIAL
                #scat = plt.scatter(allpartialrej_x[i], allpartialrej_y[i], c='r', label='Partial Rejected')
                #scat = plt.scatter(allpartial_x[i], allpartial_y[i], c='c', label='Partial Ordered individuals')

            else:
                #generations
                #scat = plt.scatter(data[0],pata[0], c='r', label='initial individuals')
                scat = plt.scatter(data[i],pata[i], c='b', label='regular individuals')

                #COIN
                scat = plt.scatter(all_x_cp[i],all_y_cp[i], c='y', label='COIN individuals')

            plt.legend()

            return scat,

        paretoX =[]
        paretoY=[]
        for j in pareto:
            paretoX.append(j.fitness.values[0])
            paretoY.append(j.fitness.values[1])

        paretoX_cp =[]
        paretoY_cp=[]
        for j in pareto_cp:
            paretoX_cp.append(j.fitness.values[0])
            paretoY_cp.append(j.fitness.values[1])

        # paretoX_cp =[]
        # paretoY_cp=[]
        # for j in pareto_coin:
        #     paretoX_cp.append(j.fitness.values[0])
        #     paretoY_cp.append(j.fitness.values[1])

        ani = animation.FuncAnimation(fig1, update_plot, frames=xrange((loop)+8), fargs=(all_x,all_y, loop-1),blit=False)
        ani.save('polls/static/videos/bang__' + str(exp.id) + "__" + str(num_block) + '.mp4', fps=3, writer="ffmpeg", extra_args=['-vcodec', 'libx264'])
        print "video ready!"



        #update actual generations with block_size
        exp.actual_gen += loop

        #save Generations Block
        #check Experiment type
        if exp.type == 'B':

            #2-D bivariate normal gaussian mixture
            gen = Generation(experiment=exp, block=num_block, comparisons="not necessary anymore",all_x=str(all_x),
                             all_y=str(all_y),all_x_cp=str(all_x_cp), all_y_cp=str(all_y_cp), means=means, covar=covar,
                             weights=weights, fitness_points2D=str(lst_points2D), num_k=num_clusters)
            gen.save()

        elif exp.type == 'C':

            #kmeans
            gen = Generation(experiment=exp, block=num_block, comparisons="not necessary anymore",all_x=str(all_x),
                             all_y=str(all_y),all_x_cp=str(all_x_cp), all_y_cp=str(all_y_cp), fitness_points2D=str(lst_points2D), num_k=num_clusters, centroids=centroids)
            gen.save()

        else:

            #type: 1-D Gaussian over the objective 1 (cost) only
            #here I fixed only at one objective ... and 2 componentes (cost x anti-cost)
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

        #run robots
        if exp.actual_gen < exp.NGEN:

            if exp.vote == 'P':
                self.RunRobots(exp, num_block, ga, my_world,pareto_cp)
            else:
                self.RunRobotsAll(exp, num_block, ga, my_world,pareto_cp, pop_cp)
        else:
            print "game over"


    def ContinueEvolutionBench(self):

        import numpy as np

        #get last ID of the Experiment
        exp = Experiment.objects.latest('id')

        #get World config
        wrl = exp.world
        gameworld = GameWorld.objects.get(id=wrl.id)
        my_world = World(gameworld.m,gameworld.n)
        my_world.experimentTYPE = exp.type

        #declare Genetic Algorithm for the problem
        ga = GA(my_world,exp.CXPB, exp.MUTPB, exp.NGEN, exp.NPOP,exp.type_prob, exp.moea_alg)
        pop=[]
        pop_cp=[]

        #number of the next block of generations
        gen = Generation.objects.latest('id')
        num_block=gen.block + 1

        #fake population
        pop=ga.SetPopulationFakeBench(my_world, problem=exp.type_prob)
        pop_cp=ga.SetPopulationFakeBench(my_world, problem=exp.type_prob)


        #get the LAST Population
        count=0
        for ind in Population.objects.all().filter(generation=gen.id):

           chromo_original = literal_eval(ind.chromosome_original)
           chromo = literal_eval(ind.chromosome)
           for i in xrange(len(chromo)):
               pop[count][i]= chromo_original[i]
               pop_cp[count][i] = chromo[i]
           count +=1
        #POP ready

        #adjut first size because de archive of SPEA has POP/2 individuals
        if exp.moea_alg=='P':
            del pop[exp.NPOP//2:]
            del pop_cp[exp.NPOP//2:]

        #########################
        pfront = self.GetFront()

        #check candidate option: front or front and population
        if exp.vote == 'P':
            #get answers from plays
            answers = [[ 0 for i in range(2) ] for j in range(len(pfront)) ]
        else:
            #get answers from plays
            answers = [[ 0 for i in range(2) ] for j in range(len(pop_cp)) ]

        plays = Play.objects.filter(play_experiment = exp, level = gen.block)
        for j in xrange(len(plays)):
            if plays[j].answer != -1:
                #print j
                if plays[j].answer == 0:
                    correct = plays[j].chromosomeOneIndex #verifica o indice da resposta correta
                    incorrect = plays[j].chromosomeTwoIndex
                else:
                    correct = plays[j].chromosomeTwoIndex #verifica o indice da resposta correta
                    incorrect = plays[j].chromosomeOneIndex
                answers[correct][0] += 1
                answers[incorrect][1] += 1

        #fake population
        pop_front=ga.SetPopulationFakeBench(my_world, problem=exp.type_prob)
        del pop_front[len(pfront):]

        #replace by front
        count=0
        for ind in pfront:

           chromo = literal_eval(ind)
           for i in xrange(len(chromo)):
               pop_front[count][i] = chromo[i]
           count +=1


        #print pop
        #print pop_cp
        #print pop_front
        # set FITNESS to the entire front
        fitnesses = ga.GetFitness(pop_front)
        ga.AttachFitness(pop_front,fitnesses)

        #POP FRONT ready


        #ok, the real population!!
        fitnesses = ga.GetFitness(pop)
        ga.AttachFitness(pop,fitnesses)
        #assign the crowding distance to the individuals

        #if NSGA-II
        if exp.moea_alg=='N':
            ga.SetCrowdingDistance(pop)
        #if SMS-EMOA
        elif exp.moea_alg=='S':
            ga.SetHyperValue(pop)


        #discover again the LAST Pareto
        pareto=tools.ParetoFront()
        pareto.update(pop)
        #POP ready

        #ok, the  population to be tested!!
        fitnesses = ga.GetFitness(pop_cp)
        ga.AttachFitness(pop_cp,fitnesses)


        #check candidate option: front or front and population
        if exp.vote == 'P':

            #Zorro
            if exp.type_prob=='G' or exp.type_prob=='H' or exp.type_prob=='I' or exp.type_prob=='J' or exp.type_prob=='L' \
                    or exp.type_prob=='M':
                #print pop
                #create list of fitness
                front_fit=[]
                for i in pop_front:
                    front_fit.append([i.fitness.values[0],i.fitness.values[1],i.fitness.values[2]])

                #print front_fit
                fakefit=front_fit

                #here I create the list of choosen points (fitness)
                fake_array = []
                lst_points2D = []
                lst_points3D = []
                count=0
                for i in answers:
                    #how many votes on i
                    for j in xrange(i[0]):
                        fake_array.append([fakefit[count][0]])
                        lst_points2D.append([fakefit[count][0] , fakefit[count][1]])
                        lst_points3D.append([fakefit[count][0] , fakefit[count][1], fakefit[count][2]])
                    count += 1

            else:
                #print pop
                #create list of fitness
                front_fit=[]
                for i in pop_front:
                    front_fit.append([i.fitness.values[0],i.fitness.values[1]])

                #print front_fit
                fakefit=front_fit

                #here I create the list of choosen points (fitness)
                fake_array = []
                lst_points2D = []
                count=0
                for i in answers:
                    #how many votes on i
                    for j in xrange(i[0]):
                        fake_array.append([fakefit[count][0]])
                        lst_points2D.append([fakefit[count][0] , fakefit[count][1]])
                    count += 1
        else:

            #print pop
            #create list of fitness
            pop_fit=[]
            for i in pop_cp:
                pop_fit.append([i.fitness.values[0],i.fitness.values[1]])

            #print front_fit
            fakefit=pop_fit

            #here I create the list of choosen points (fitness)
            fake_array = []
            lst_points2D = []
            count=0
            for i in answers:
                #how many votes on i
                for j in xrange(i[0]):
                    fake_array.append([fakefit[count][0]])
                    lst_points2D.append([fakefit[count][0] , fakefit[count][1]])
                count += 1




        ##########################################################################
        #POP, PARETO FRONT, FITNESS FRONT and List of POINTS (fitness)  is  ready#
        ##########################################################################
        clusteOBJ = Cluster()

        #Zorro
        if exp.type_prob=='G' or exp.type_prob=='H' or exp.type_prob=='I'  or exp.type_prob=='J' or exp.type_prob=='L' or exp.type_prob=='M':
            t = np.array(lst_points3D, np.float)
        else:
            t = np.array(lst_points2D, np.float)

        #check possible number of Components for fixed number of components types
        num_clusters = 2 #my default
        if exp.bots_points == 'A':
            num_clusters = 1
        elif exp.bots_points == 'B':
            num_clusters = 2
        elif exp.bots_points == 'C':
            num_clusters = 3


        #initialize
        kmm=None
        best_gmm=None

        #check Experiment type
        if exp.type == 'B':

            #2-D bivariate normal gaussian mixture

            #if fixed number of clusters
            if not exp.freeK and exp.bots_points != 'D':

                #run EM
                best_gmm = clusteOBJ.EM(t, num_clusters)

            else:
                #free k clusters or many reference points for robots

                #run EM free Components K
                best_gmm = clusteOBJ.EMfreeK(t)
                num_clusters = best_gmm.n_components


            #save results to local vars for database
            #save results to my_world object
            means = str(best_gmm.means_.tolist())
            covar = str(best_gmm.covars_.tolist())
            weights = str(best_gmm.weights_.tolist())
            my_world.means=best_gmm.means_
            my_world.covar=best_gmm.covars_
            my_world.weights=best_gmm.weights_

            print means



        elif exp.type == 'D':

            #3-D trivariate normal gaussian mixture

            #if fixed number of clusters
            if not exp.freeK and exp.bots_points != 'D':

                #run EM
                best_gmm = clusteOBJ.EM(t, num_clusters)
                ###AQUIIIIII Trocar depois


            else:
                #free k clusters or many reference points for robots

                #run EM free Components K
                best_gmm = clusteOBJ.EMfreeK(t)
                num_clusters = best_gmm.n_components


            #save results to local vars for database
            #save results to my_world object
            means = str(best_gmm.means_.tolist())
            covar = str(best_gmm.covars_.tolist())
            weights = str(best_gmm.weights_.tolist())
            my_world.means=best_gmm.means_
            my_world.covar=best_gmm.covars_
            my_world.weights=best_gmm.weights_

            print means


        elif exp.type == 'C' or exp.type == 'E':

            #kmeans
            #if fixed number of clusters
            if not exp.freeK and exp.bots_points != 'D':

                #run Kmeans
                kmm = clusteOBJ.Kmeans(t, num_clusters)
                #kmm.cluster_centers_
            else:
                #free k clusters or many reference points for robots

                #run Kmeans free Components K
                #kmm.cluster_centers_
                kmm = clusteOBJ.KmeansfreeK(t)
                num_clusters = kmm.n_clusters

            #save results to local vars for database
            #save results to my_world object
            centroids = str(kmm.cluster_centers_.tolist())
            my_world.centroids=kmm.cluster_centers_.tolist()

            print centroids
        else:

            #type: 1-D Gaussian over the objective 1 (cost) only
            #here I fixed only at one objective ... and 2 componentes (cost x anti-cost)
            t = np.array(fake_array, np.int32)
            #t.reshape(len(t),2).shape
            # result = clusteOBJ.expectation_maximization(t, nbclusters=2, nbiter=1, epsilon=1)
            # k1_mu = result['params'][0]['mu']
            # k1_p = result['params'][0]['proba']
            # k1_sigma = result['params'][0]['sigma']
            #
            # k2_mu = result['params'][1]['mu']
            # k2_p = result['params'][1]['proba']
            # k2_sigma = result['params'][1]['sigma']

            result =  clusteOBJ.EM(t, 2)

            k1_mu = result.means_[0]
            k1_p = result.weights_[0]
            k1_sigma = result.covars_[0]

            k2_mu = result.means_[1]
            k2_p = result.weights_[1]
            k2_sigma = result.covars_[1]

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

            # set the means to the entire population
            my_world.mean1=mean1
            my_world.sigma1 = sigma1
            my_world.p1 = p1
            my_world.mean2=mean2
            my_world.sigma2=sigma2
            my_world.p2=p2


        # #assign the coin distance to the individuals
        partial=[]
        partial_rej=[]

        #TIRAR
        partialn=[]
        partial_rejn=[]
        partial_xn=[]
        partial_yn=[]
        partialrej_xn=[]
        partialrej_yn=[]
        allpartial_xn=[]
        allpartial_yn=[]
        allpartialrej_xn=[]
        allpartialrej_yn=[]

        #if NSGA-II
        if exp.moea_alg=='N':
            ga.SetCOINDistance(pop_cp, partial, partial_rej)
        #if SMS-EMOA
        if exp.moea_alg=='S':
            ga.SetHyperValueCOIN(pop_cp)



        #discover again the LAST Pareto
        pareto_cp=tools.ParetoFront()
        pareto_cp.update(pop_cp)

        ##########################
        #POP ready               #
        #POP original READY      #
        #FRONT ready             #
        #MEANS ready             #
        #COVARS ready            #
        #List of POINTS (fitness)#
        ##########################

        ##########
        #ORIGINAL#
        ##########

        #start matplot
        fig1 = plt.figure()
        if exp.type == 'D' or exp.type == 'E':
            ax = fig1.add_subplot(111, projection='3d')

        #arrays
        pointsX =[]
        pointsY=[]
        pointsZ=[]
        all_x=[]
        all_y=[]
        all_z=[]
        pointsX_cp =[]
        pointsY_cp=[]
        pointsZ_cp=[]
        all_x_cp=[]
        all_y_cp=[]
        all_z_cp=[]

        #partial points
        partial_x=[]
        partial_y=[]
        partialrej_x=[]
        partialrej_y=[]
        for j in partial:
            partial_x.append(j.fitness.values[0])
            partial_y.append(j.fitness.values[1])
        # scat = plt.scatter(pointsX,pointsY, c='r')
        allpartial_x=[]
        allpartial_y=[]
        allpartial_x.append(partial_x)
        allpartial_y.append(partial_y)
        for j in partial_rej:
            partialrej_x.append(j.fitness.values[0])
            partialrej_y.append(j.fitness.values[1])
        # scat = plt.scatter(pointsX,pointsY, c='r')
        allpartialrej_x=[]
        allpartialrej_y=[]
        allpartialrej_x.append(partial_x)
        allpartialrej_y.append(partial_y)


        #points for the ORIGINAL population
        #Zorro
        if exp.type == 'D' or exp.type == 'E':
            for j in pop:
                pointsX.append(j.fitness.values[0])
                pointsY.append(j.fitness.values[1])
                pointsZ.append(j.fitness.values[2])
            # scat = plt.scatter(pointsX,pointsY, c='r')
            all_z.append(pointsZ)
        else:
            for j in pop:
                pointsX.append(j.fitness.values[0])
                pointsY.append(j.fitness.values[1])
            # scat = plt.scatter(pointsX,pointsY, c='r')
        all_x.append(pointsX)
        all_y.append(pointsY)



        #ok, the TEST population!!
        #pop_cp is the population to be used with COIN or Robot means


        #pareto from the generation 1 of this round
        paretoX_gen1 =[]
        paretoY_gen1=[]
        paretoZ_gen1=[]
        #Zorro
        if exp.type == 'D' or exp.type =='E':
            for j in pareto_cp:
                paretoX_gen1.append(j.fitness.values[0])
                paretoY_gen1.append(j.fitness.values[1])
                paretoZ_gen1.append(j.fitness.values[2])
        else:
            for j in pareto_cp:
                paretoX_gen1.append(j.fitness.values[0])
                paretoY_gen1.append(j.fitness.values[1])
        print "  generation: ",format(0, '03d')

        #points for the TEST population
        #Zorro
        if exp.type == 'D' or exp.type == 'E':
            for j in pop_cp:
                pointsX_cp.append(j.fitness.values[0])
                pointsY_cp.append(j.fitness.values[1])
                pointsZ_cp.append(j.fitness.values[2])
            all_z_cp.append(pointsZ_cp)
            #scat = ax.scatter(pointsX_cp,pointsY_cp,pointsZ_cp, c='r')
        else:
            for j in pop_cp:
                pointsX_cp.append(j.fitness.values[0])
                pointsY_cp.append(j.fitness.values[1])
            #scat = plt.scatter(pointsX_cp,pointsY_cp, c='r')
        all_x_cp.append(pointsX_cp)
        all_y_cp.append(pointsY_cp)

        if exp.moea_alg=='P':
            archive = deepcopy(pop)
            archive_cp = deepcopy(pop_cp)



        #control the loop
        if exp.actual_gen + exp.block_size <= exp.gen_threshold and num_block <= exp.numLevels:
            loop= exp.block_size
        else:
            loop= exp.NGEN - exp.actual_gen

        #loop for BLOCK generations
        for i in xrange(loop):

            #ORIGINAL

            partialn=[]
            partial_rejn=[]

            #if NSGA-II
            if exp.moea_alg=='N':

                # Select the next generation individuals
                ga.NSGASelection_Pure(pop, partialn, partial_rejn)
                pareto.update(pop)


                #partial
                partial=[]
                partial_rej=[]



                ga.NSGASelectionCOIN(pop_cp, my_world, ga, exp.tour, exp.type, best_gmm, kmm, partial, partial_rej)
                #ga.NSGASelection_Pure(pop_cp)
                #this is the front that contains the previous population
                # pareto_cp.update(pop_cp)
                # print len(pareto_cp)

                #save points of evolution to ORIGINAL and TEST
                pointsX =[]
                pointsY=[]
                pointsZ=[]
                pointsX_cp =[]
                pointsY_cp=[]
                pointsZ_cp=[]


                #Zorro
                if exp.type == 'D' or exp.type == 'E':
                    for j in xrange(len(pop)):
                        pointsX.append(pop[j].fitness.values[0])
                        pointsY.append(pop[j].fitness.values[1])
                        pointsZ.append(pop[j].fitness.values[2])
                        pointsX_cp.append(pop_cp[j].fitness.values[0])
                        pointsY_cp.append(pop_cp[j].fitness.values[1])
                        pointsZ_cp.append(pop_cp[j].fitness.values[2])
                    all_z.append(pointsZ)
                    all_z_cp.append(pointsZ_cp)
                else:
                    for j in xrange(len(pop)):
                        pointsX.append(pop[j].fitness.values[0])
                        pointsY.append(pop[j].fitness.values[1])
                        pointsX_cp.append(pop_cp[j].fitness.values[0])
                        pointsY_cp.append(pop_cp[j].fitness.values[1])


            elif exp.moea_alg=='P':

                # Select the next generation individuals
                ga.SPEA2Selection_Pure(pop, archive,  partialn, partial_rejn, firstloop_continue=i)
                pareto.update(pop)


                #partial
                partial=[]
                partial_rej=[]

                ga.SPEA2SelectionCOIN(pop_cp, archive_cp, my_world, ga, exp.tour, exp.type, best_gmm, kmm, partial, partial_rej, firstloop_continue=i)

                #save points of evolution to ORIGINAL and TEST
                pointsX =[]
                pointsY=[]
                pointsZ=[]
                pointsX_cp =[]
                pointsY_cp=[]
                pointsZ_cp=[]


                aaa = ga.GetFitness(archive)
                aaa.sort(key=itemgetter(1))
                bbb = ga.GetFitness(archive_cp)
                bbb.sort(key=itemgetter(1))
                if len(archive) != len(archive_cp):
                    print "XxXZZZZZZZZZZZZLLLLLLLLLLLLALALALALALALALAALAL"
                    print len(archive_cp)

                #Zorro
                if exp.type == 'D' or exp.type == 'E':
                    for j in xrange(len(archive)):
                        pointsX.append(archive[j].fitness.values[0])
                        pointsY.append(archive[j].fitness.values[1])
                        pointsZ.append(archive[j].fitness.values[2])
                        pointsX_cp.append(archive_cp[j].fitness.values[0])
                        pointsY_cp.append(archive_cp[j].fitness.values[1])
                        pointsZ_cp.append(archive_cp[j].fitness.values[2])
                    all_z.append(pointsZ)
                    all_z_cp.append(pointsZ_cp)
                else:
                    for j in xrange(len(archive)):
                        pointsX.append(archive[j].fitness.values[0])
                        pointsY.append(archive[j].fitness.values[1])
                        pointsX_cp.append(archive_cp[j].fitness.values[0])
                        pointsY_cp.append(archive_cp[j].fitness.values[1])

            #if SMS-EMOA
            elif exp.moea_alg=='S':

                # Select the next generation individuals
                #ga.SMSSelection_Pure(pop, partialn, partial_rejn)
                #pareto.update(pop)


                #partial
                partial=[]
                partial_rej=[]



                ga.SMSSelection_COIN(pop_cp, my_world, ga, exp.tour, exp.type, best_gmm, kmm, partial, partial_rej)
                #ga.NSGASelection_Pure(pop_cp)
                #this is the front that contains the previous population
                # pareto_cp.update(pop_cp)
                # print len(pareto_cp)

                #save points of evolution to ORIGINAL and TEST
                pointsX =[]
                pointsY=[]
                pointsZ=[]
                pointsX_cp =[]
                pointsY_cp=[]
                pointsZ_cp=[]


                #Zorro
                if exp.type == 'D' or exp.type == 'E':
                    for j in xrange(len(pop)):
                        pointsX.append(pop[j].fitness.values[0])
                        pointsY.append(pop[j].fitness.values[1])
                        pointsZ.append(pop[j].fitness.values[2])
                        pointsX_cp.append(pop_cp[j].fitness.values[0])
                        pointsY_cp.append(pop_cp[j].fitness.values[1])
                        pointsZ_cp.append(pop_cp[j].fitness.values[2])
                    all_z.append(pointsZ)
                    all_z_cp.append(pointsZ_cp)
                else:
                    for j in xrange(len(pop)):
                        pointsX.append(pop[j].fitness.values[0])
                        pointsY.append(pop[j].fitness.values[1])
                        pointsX_cp.append(pop_cp[j].fitness.values[0])
                        pointsY_cp.append(pop_cp[j].fitness.values[1])


            all_x.append(pointsX)
            all_y.append(pointsY)
            all_x_cp.append(pointsX_cp)
            all_y_cp.append(pointsY_cp)
            #
            # #partial points
            # partial_x=[]
            # partial_y=[]
            # for j in partial:
            #     partial_x.append(j.fitness.values[0])
            #     partial_y.append(j.fitness.values[1])
            # # scat = plt.scatter(pointsX,pointsY, c='r')
            # allpartial_x.append(partial_x)
            # allpartial_y.append(partial_y)
            #
            # partialrej_x=[]
            # partialrej_y=[]
            # for j in partial_rej:
            #     partialrej_x.append(j.fitness.values[0])
            #     partialrej_y.append(j.fitness.values[1])
            # # scat = plt.scatter(pointsX,pointsY, c='r')
            # allpartialrej_x.append(partialrej_x)
            # allpartialrej_y.append(partialrej_y)
            # #print partial_rej
            #
            # #Tirar
            # #partial points
            # partial_xn=[]
            # partial_yn=[]
            # for j in partialn:
            #     partial_xn.append(j.fitness.values[0])
            #     partial_yn.append(j.fitness.values[1])
            # # scat = plt.scatter(pointsX,pointsY, c='r')
            # allpartial_xn.append(partial_xn)
            # allpartial_yn.append(partial_yn)
            #
            # partialrej_xn=[]
            # partialrej_yn=[]
            # for j in partial_rejn:
            #     partialrej_xn.append(j.fitness.values[0])
            #     partialrej_yn.append(j.fitness.values[1])
            # # scat = plt.scatter(pointsX,pointsY, c='r')
            # allpartialrej_xn.append(partialrej_xn)
            # allpartialrej_yn.append(partialrej_yn)


            #
            # fitnesses = ga.GetFitnessPAIR(pop_cp)
            # ga.AttachFitness(pop_cp,fitnesses)
            # # #assign the crowding distance to the individuals
            # ga.SetCrowdingDistance(pop_cp)

            print "  generation: ", format(i+1, '03d')



        #if SPEA2, take the archive as the Pareto Front
        if exp.moea_alg=='P':
            pop = archive
            pop_cp = archive_cp
            #print "yuka"

        pareto=tools.ParetoFront()
        pareto_cp=tools.ParetoFront()
        pareto.update(pop)
        pareto_cp.update(pop_cp)
        #print "PARETO: " + str(len(pareto_cp))



        #####
        #MP4#
        #####
        # print "andei imprimir"
        #first generation
        expX= literal_eval(exp.paretoX_gen1)
        expY= literal_eval(exp.paretoY_gen1)
        #def update_plot(i, data, pata, paretoX, paretoY,  paretoX_gen1, paretoY_gen1, paretoX_cp, paretoY_cp, data_cp, pata_cp,  last, scat):
        def update_plot(i, data, pata,last):

            #scat.set_array(data[i])
            #plt.pause(0.5)
            plt.clf()
            plt.xlabel('f1')
            plt.ylabel('f2')
            plt.title('the evolution')
            #draw first pop

            #Zorro
            if exp.type == 'B' or exp.type == 'D' :

                xz=1
                #2-D bivariate normal gaussian mixture
                # for j in best_gmm.means_:
                #     #plt.plot(j[0], j[1],  'gD', markersize=8)
                #     cir1 = plt.Circle((j[0], j[1]),0.02,color='k',fill=False)
                #     cir2 = plt.Circle((j[0], j[1]),0.04,color='g',fill=False)
                #     cir3 = plt.Circle((j[0], j[1]),0.06,color='y',fill=False)
                #     fig = plt.gcf()
                #     fig.gca().add_artist(cir1)
                #     fig.gca().add_artist(cir2)

            elif exp.type == 'C':

                #kmeans
                for j in kmm.cluster_centers_:
                    plt.plot(j[0], j[1],  'gD', markersize=8)

            else:
                plt.text(mean1-15,-240-15,'1-D cost-mean')
                plt.plot(mean1, -240,  'g^', markersize=8)
                plt.text(mean2-15,-240-15,'1-D cost-mean')
                plt.plot(mean2, -240,  'g^', markersize=8)

            if i>=(last-1):

                #final

                #first generation
                scat = plt.plot(expX, expY, c='k', label='first generation front')
                scat = plt.scatter(expX, expY, c='k')

                #fronts
                #scat = plt.plot(data[0],pata[0], c='c', label='initial regular front')
                scat = plt.plot(paretoX_gen1,paretoY_gen1, c='r', label='initial COIN front')
                scat = plt.plot(paretoX,paretoY, c='b', label='final regular front')

                scat = plt.scatter(data[loop-1],pata[loop-1], c='b')
                scat = plt.scatter(data[0],pata[0], c='r')

                #COIN
                scat = plt.plot(paretoX_cp,paretoY_cp, c='y', label='final COIN front')
                scat = plt.scatter(all_x_cp[loop-1],all_y_cp[loop-1], c='y')


            else:
                #generations
                #scat = plt.scatter(data[0],pata[0], c='r', label='initial individuals')
                #scat = plt.scatter(data[i],pata[i], c='b', label='regular individuals')

                #COIN
                #scat = plt.scatter(all_x_cp[i],all_y_cp[i], c='y', label='COIN individuals')

                scat = plt.scatter(all_x_cp[i],all_y_cp[i], c='cyan', alpha=1, label='relevant individuals')
                scat = plt.plot(paretoX,paretoY, c='darkgrey', alpha=1, label='Pareto front')


                #PARTIAL
                #scat = plt.scatter(allpartialrej_x[i], allpartialrej_y[i], c='r', label='Partial Rejected')
                #scat = plt.scatter(allpartial_x[i], allpartial_y[i], c='c', label='Partial Ordered individuals')

            plt.legend()

            return scat,

        #Zorro
        def update_plot3D(i, data, pata,last):

            #scat.set_array(data[i])
            #plt.pause(0.5)
            #plt.clf()
            # ax.set_xlabel('f1')
            # ax.set_ylabel('f2')
            # ax.set_ylabel('f3')

            #plt.title('the evolution')
            #draw first pop

            #Zorro

            #generations
            #scat = plt.scatter(data[0],pata[0], c='r', label='initial individuals')
            #scat = plt.scatter(data[i],pata[i], c='b', label='regular individuals')

            #COIN
            #scat = plt.scatter(all_x_cp[i],all_y_cp[i], c='y', label='COIN individuals')

            #scat = ax.scatter(all_x_cp[i],all_y_cp[i],all_z_cp[i], c='cyan', alpha=1, label='relevant individuals')
            scat = ax.scatter(paretoX,paretoY,paretoZ, marker='+', c='lightgrey', alpha=1, label='Pareto front')
            scat = ax.scatter(paretoX_cp ,paretoY_cp,paretoZ_cp, c='k', alpha=1, label='Pareto front')


            #PARTIAL
            #scat = plt.scatter(allpartialrej_x[i], allpartialrej_y[i], c='r', label='Partial Rejected')
            #scat = plt.scatter(allpartial_x[i], allpartial_y[i], c='c', label='Partial Ordered individuals')

            #plt.legend()

            return scat,


        def update_plot_partial_coin(i, data, pata,last):

            #scat.set_array(data[i])
            #plt.pause(0.5)
            plt.clf()
            plt.xlabel('f1')
            plt.ylabel('f2')
            plt.title('the evolution')
            #draw first pop

            if exp.type == 'B':

                #2-D bivariate normal gaussian mixture
                for j in best_gmm.means_:
                    #plt.plot(j[0], j[1],  'gD', markersize=8)
                    cir1 = plt.Circle((j[0], j[1]),0.02,color='k',fill=False)
                    cir2 = plt.Circle((j[0], j[1]),0.04,color='g',fill=False)
                    cir3 = plt.Circle((j[0], j[1]),0.06,color='y',fill=False)
                    fig = plt.gcf()
                    fig.gca().add_artist(cir1)
                    fig.gca().add_artist(cir2)

            elif exp.type == 'C':

                #kmeans
                for j in kmm.cluster_centers_:
                    plt.plot(j[0], j[1],  'gD', markersize=8)

            else:
                plt.text(mean1-15,-240-15,'1-D cost-mean')
                plt.plot(mean1, -240,  'g^', markersize=8)
                plt.text(mean2-15,-240-15,'1-D cost-mean')
                plt.plot(mean2, -240,  'g^', markersize=8)

            if i>=(last-1):

                #final

                #first generation
                scat = plt.plot(expX, expY, c='k', label='first generation front')
                scat = plt.scatter(expX, expY, c='k')

                #fronts
                #scat = plt.plot(data[0],pata[0], c='c', label='initial regular front')
                scat = plt.plot(paretoX_gen1,paretoY_gen1, c='r', label='initial COIN front')
                scat = plt.plot(paretoX,paretoY, c='b', label='final regular front')

                scat = plt.scatter(data[loop-1],pata[loop-1], c='b')
                scat = plt.scatter(data[0],pata[0], c='r')

                #COIN
                scat = plt.plot(paretoX_cp,paretoY_cp, c='y', label='final COIN front')
                scat = plt.scatter(all_x_cp[loop-1],all_y_cp[loop-1], c='y')


            else:
                #generations
                #scat = plt.scatter(data[0],pata[0], c='r', label='initial individuals')
                #scat = plt.scatter(data[i],pata[i], c='b', label='regular individuals')

                #COIN
                #scat = plt.scatter(all_x_cp[i],all_y_cp[i], c='y', label='COIN individuals')

                #PARTIAL
                scat = plt.scatter(allpartialrej_x[i], allpartialrej_y[i], c='r', label='Partial Rejected')
                scat = plt.scatter(allpartial_x[i][:150], allpartial_y[i][:150], c='c', label='Partial Ordered individuals')
            plt.legend()
            plt.ylim(0,2)
            plt.xlim(-0.2,1.2)

            return scat,


        def update_plot_partial_nsga(i, data, pata,last):

            #scat.set_array(data[i])
            #plt.pause(0.5)
            plt.clf()
            plt.xlabel('f1')
            plt.ylabel('f2')
            plt.title('the evolution')
            #draw first pop

            if exp.type == 'B':
                kk=1
                #2-D bivariate normal gaussian mixture
                # for j in best_gmm.means_:
                #     #plt.plot(j[0], j[1],  'gD', markersize=8)
                #     cir1 = plt.Circle((j[0], j[1]),0.02,color='k',fill=False)
                #     cir2 = plt.Circle((j[0], j[1]),0.04,color='g',fill=False)
                #     cir3 = plt.Circle((j[0], j[1]),0.06,color='y',fill=False)
                #     fig = plt.gcf()
                #     fig.gca().add_artist(cir1)
                #     fig.gca().add_artist(cir2)

            elif exp.type == 'C':

                #kmeans
                for j in kmm.cluster_centers_:
                    plt.plot(j[0], j[1],  'gD', markersize=8)

            else:
                plt.text(mean1-15,-240-15,'1-D cost-mean')
                plt.plot(mean1, -240,  'g^', markersize=8)
                plt.text(mean2-15,-240-15,'1-D cost-mean')
                plt.plot(mean2, -240,  'g^', markersize=8)

            if i>=(last-1):

                #final

                #first generation
                scat = plt.plot(expX, expY, c='k', label='first generation front')
                scat = plt.scatter(expX, expY, c='k')

                #fronts
                #scat = plt.plot(data[0],pata[0], c='c', label='initial regular front')
                scat = plt.plot(paretoX_gen1,paretoY_gen1, c='r', label='initial COIN front')
                scat = plt.plot(paretoX,paretoY, c='b', label='final regular front')

                scat = plt.scatter(data[loop-1],pata[loop-1], c='b')
                scat = plt.scatter(data[0],pata[0], c='r')

                #COIN
                scat = plt.plot(paretoX,paretoY, c='y', label='final COIN front')
                scat = plt.scatter(all_x[loop-1],all_y[loop-1], c='y')


            else:
                #generations
                #scat = plt.scatter(data[0],pata[0], c='r', label='initial individuals')
                #scat = plt.scatter(data[i],pata[i], c='b', label='regular individuals')

                #COIN
                #scat = plt.scatter(all_x[i],all_y[i], c='y', label='individuals')

                #PARTIAL
                scat = plt.scatter(allpartialrej_xn[i], allpartialrej_yn[i], c='r', label='Partial Rejected')
                scat = plt.scatter(allpartial_xn[i], allpartial_yn[i], c='c', label='Partial Ordered individuals')

            plt.legend()
            plt.ylim(0,2)
            plt.xlim(-0.2,1.2)

            return scat,


        paretoX =[]
        paretoY=[]
        paretoZ=[]
        #Zorro
        if exp.type == 'D' or exp.type =='E':
            for j in pareto:
                paretoX.append(j.fitness.values[0])
                paretoY.append(j.fitness.values[1])
                paretoZ.append(j.fitness.values[2])
        else:
            for j in pareto:
                paretoX.append(j.fitness.values[0])
                paretoY.append(j.fitness.values[1])

        paretoX_cp =[]
        paretoY_cp=[]
        paretoZ_cp=[]
        #Zorro
        if exp.type == 'D' or exp.type == 'E':
            for j in pareto_cp:
                paretoX_cp.append(j.fitness.values[0])
                paretoY_cp.append(j.fitness.values[1])
                paretoZ_cp.append(j.fitness.values[2])
        else:
            for j in pareto_cp:
                paretoX_cp.append(j.fitness.values[0])
                paretoY_cp.append(j.fitness.values[1])

        # paretoX_cp =[]
        # paretoY_cp=[]
        # for j in pareto_coin:
        #     paretoX_cp.append(j.fitness.values[0])
        #     paretoY_cp.append(j.fitness.values[1])
        #print paretoZ_cp
        # if exp.type == 'D':
        #     ssd=82
        #     ani = animation.FuncAnimation(fig1, update_plot3D, frames=xrange((loop)+8), fargs=(all_x,all_y, loop-1),blit=False)
        #     ani.save('polls/static/videos/bang__' + str(exp.id) + "__" + str(num_block) + '.mp4', fps=3, writer="ffmpeg", extra_args=['-vcodec', 'libx264'])
        # else:
        #     ani = animation.FuncAnimation(fig1, update_plot, frames=xrange((loop)+8), fargs=(all_x,all_y, loop-1),blit=False)
        #     ani.save('polls/static/videos/bang__' + str(exp.id) + "__" + str(num_block) + '.mp4', fps=3, writer="ffmpeg", extra_args=['-vcodec', 'libx264'])

        #ani = animation.FuncAnimation(fig1, update_plot_partial_coin, frames=xrange((loop)+8), fargs=(all_x,all_y, loop-1),blit=False)
        #ani.save('polls/static/videos/bang__partialCOIN_' + str(exp.id) + "__" + str(num_block) + '.mp4', fps=3, writer="ffmpeg", extra_args=['-vcodec', 'libx264'])

        #ani = animation.FuncAnimation(fig1, update_plot_partial_nsga, frames=xrange((loop)+8), fargs=(all_x,all_y, loop-1),blit=False)
        #ani.save('polls/static/videos/bang__partialNSGA_' + str(exp.id) + "__" + str(num_block) + '.mp4', fps=3, writer="ffmpeg", extra_args=['-vcodec', 'libx264'])
        # if exp.type == 'D' or exp.type =='E':
        #     scat = ax.scatter(paretoX,paretoY,paretoZ, marker='+', c='lightgrey', alpha=1, label='Pareto front')
        #     scat = ax.scatter(paretoX_cp ,paretoY_cp,paretoZ_cp, c='k', alpha=1, label='Pareto front')
        # else:
        #     #scat = plt.scatter(all_x_cp[i],all_y_cp[i], c='cyan', alpha=1, label='relevant individuals')
        #     #scat = plt.plot(paretoX,paretoY, c='darkgrey', alpha=1, label='Pareto front')
        #     plt.legend()

        #pp = PdfPages('polls/static/fronts/front_' + str(exp.id) + '__' + str(num_block) + '.pdf')
        #pp.savefig(fig1)
        #pp.close()

        #YUKA
        if num_block > 38:

            if exp.type == 'D' or exp.type == 'E':
                scat = ax.scatter(paretoX,paretoY,paretoZ, marker='+', c='lightgrey', alpha=1, label='Pareto front')
                scat = ax.scatter(paretoX_cp ,paretoY_cp,paretoZ_cp, c='k', alpha=1, label='Relevant Individuals 3D')
                plt.legend()

            else:
                #plt.ylim([-0.1,1.0])
                #plt.xlim([-0.1,1.0])

                # for j in best_gmm.means_:
                #     #plt.plot(j[0], j[1],  'gD', markersize=8)
                #     cir1 = plt.Circle((j[0], j[1]),0.02,color='k',fill=False)
                #     cir2 = plt.Circle((j[0], j[1]),0.04,color='g',fill=False)
                #     cir3 = plt.Circle((j[0], j[1]),0.06,color='y',fill=False)
                #     fig = plt.gcf()
                #     fig.gca().add_artist(cir1)
                #     fig.gca().add_artist(cir2)

                scat = plt.scatter(all_x_cp[i],all_y_cp[i], c='cyan', alpha=1, label='relevant individuals')
                #scat = plt.plot(paretoX,paretoY, c='darkgrey', alpha=1, label='Pareto front')
                plt.legend()

            pp = PdfPages('polls/static/fronts/front_' + str(exp.id) + '__' + str(num_block) + '.pdf')
            pp.savefig(fig1)
            pp.close()

        #print "video ready!"
        # print "sim, imprimi"

        #update actual generations with block_size
        exp.actual_gen += loop

        #save Generations Block
        #check Experiment type
        #Zorro
        if exp.type == 'D':

            if exp.type_prob=='F':
                #2-D bivariate normal gaussian mixture
                gen = Generation(experiment=exp, block=num_block, comparisons="not necessary anymore",all_x=str(all_x),
                                 all_y=str(all_y),all_x_cp=str(all_x_cp), all_y_cp=str(all_y_cp), means=means, covar=covar,
                                 weights=weights, fitness_points2D=str(lst_points2D), num_k=num_clusters)

            else:

                #3-D bivariate normal gaussian mixture
                gen = Generation(experiment=exp, block=num_block, comparisons="not necessary anymore",all_x=str(all_x),
                                 all_y=str(all_y),all_x_cp=str(all_x_cp), all_y_cp=str(all_y_cp), means=means, covar=covar,
                                 weights=weights, fitness_points2D=str(lst_points3D), num_k=num_clusters)
            gen.save()

        elif exp.type == 'E':

            if exp.type=='F':
                #2-D bivariate normal gaussian mixture
                gen = Generation(experiment=exp, block=num_block, comparisons="not necessary anymore",all_x=str(all_x),
                                 all_y=str(all_y),all_x_cp=str(all_x_cp), all_y_cp=str(all_y_cp), means=means, covar=covar,
                                 weights=weights, fitness_points2D=str(lst_points2D), num_k=num_clusters)

            else:

                #3-D bivariate normal gaussian mixture
                gen = Generation(experiment=exp, block=num_block, comparisons="not necessary anymore",all_x=str(all_x),
                                 all_y=str(all_y),all_x_cp=str(all_x_cp), all_y_cp=str(all_y_cp),
                                  fitness_points2D=str(lst_points3D), num_k=num_clusters, centroids=centroids)
            gen.save()

        elif exp.type == 'B':

            #2-D bivariate normal gaussian mixture
            gen = Generation(experiment=exp, block=num_block, comparisons="not necessary anymore",all_x=str(all_x),
                             all_y=str(all_y),all_x_cp=str(all_x_cp), all_y_cp=str(all_y_cp), means=means, covar=covar,
                             weights=weights, fitness_points2D=str(lst_points2D), num_k=num_clusters)
            gen.save()


        elif exp.type == 'C':

            #kmeans
            gen = Generation(experiment=exp, block=num_block, comparisons="not necessary anymore",all_x=str(all_x),
                             all_y=str(all_y),all_x_cp=str(all_x_cp), all_y_cp=str(all_y_cp), fitness_points2D=str(lst_points2D), num_k=num_clusters, centroids=centroids)
            gen.save()

        else:

            #type: 1-D Gaussian over the objective 1 (cost) only
            #here I fixed only at one objective ... and 2 componentes (cost x anti-cost)
            gen = Generation(experiment=exp, block=num_block, comparisons="not necessary anymore",all_x=str(all_x),
                             all_y=str(all_y),all_x_cp=str(all_x_cp), all_y_cp=str(all_y_cp), mean_1=mean1, sigma_1=sigma1,p_1=p1,
                              mean_2=mean2, sigma_2=sigma2,p_2=p2)
            gen.save()



        #save population and Front into database
        with transaction.atomic ():
            for ind in xrange(len(pop_cp)):

                temp_pop_cp=[]
                temp_pop=[]
                for i in pop_cp[ind]:
                    temp_pop_cp.append(i)
                for j in pop[ind]:
                    temp_pop.append(j)

                population = Population(generation=gen, chromosome=str(temp_pop_cp), index=count, chromosome_original=str(temp_pop))
                population.save()

            count=0
            for ind in pareto_cp:

                # temp="["
                # for i in ind:
                #     temp = temp + str(i) + ", "
                # temp = temp + "]"
                temp=[]
                for i in ind:
                    temp.append(i)

                pfront = PFront(generation=gen, chromosome=str(temp), index=count)
                pfront.save()
                count +=1

            #xfront=[]
            #for i in pareto_cp:
            #    xfront.append([i.fitness.values[0],i.fitness.values[1]])

            #xfit = ga.GetFitness(pareto_cp)
            #ga.AttachFitness(pareto_cp,xfit)

            #print xfront
            #print count



        #set the next status according the number of generations
        if exp.actual_gen >= exp.NGEN:
            exp.flag = 'F'
        else:
            exp.flag = 'R'

        exp.save()


        #print str(len(pareto_cp))
        #run robots
        if exp.actual_gen < exp.NGEN:

            if exp.vote == 'P':
                #Zorro
                if exp.type == 'D' or exp.type == 'E':
                    self.RunRobots(exp, num_block, ga, my_world,pareto_cp, exp.type_prob)
                else:
                    self.RunRobots(exp, num_block, ga, my_world,pareto_cp)


            else:
                self.RunRobotsAll(exp, num_block, ga, my_world,pareto_cp, pop_cp)
        else:
            print "game over"


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
    def GetFront(self, problem='P'):

        #get last generation inserted
        gen = Generation.objects.latest('id')
        pop=[]

        #get the LAST Front
        for ind in PFront.objects.all().filter(generation=gen.id):
            if problem == 'A':
                pop.append(literal_eval(ind.chromosome))
            else:
                pop.append(ind.chromosome)


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
    def RunRobots(self, exp, num_block, ga, my_world, pf, problem='A'):

        #pop=[]
        out=[]
        robot_lst=[]
        #get Pareto Front
        #pf=self.GetFront()


        #fake population
        # pop=ga.SetPopulationFake(my_world)
        # del pop[len(pf):]
        #
        # #replace by front
        # count=0
        # for ind in pf:
        #    pop[count][0] = ind
        #    count +=1
        #
        # # set FITNESS to the entire front
        # fitnesses = ga.GetFitness(pop)
        # ga.AttachFitness(pop,fitnesses)
        #

        #create list of fitness
        front_fit=[]

        #Zorro
        if problem == 'G' or problem == 'H' or problem == 'I'  or exp.type_prob=='J' or exp.type_prob=='L' or exp.type_prob=='M':

            for i in pf:
                front_fit.append([i.fitness.values[0],i.fitness.values[1],i.fitness.values[2]])

        else:

            for i in pf:
                front_fit.append([i.fitness.values[0],i.fitness.values[1]])

        #print "PARETO dentro Run: " + str(len(pf))

        #####################
        #GetRobotComparisons#
        #####################


        #get a list of ROBOT PLAYERS
        if num_block>1:
            #get plays from first round
            k=Play.objects.all().filter(play_experiment=exp.id, level=1, play_player__type = 'C').values('play_player').distinct()

            for i in k:
                robot_lst.append(Player.objects.get(id=i['play_player']))

        else:
            #robot creation: it must be created first
            #create Robots
            robot_lst = self.CreateRobots(exp.num_robots, exp.bots_points, problem)

        #Here I have the robot list
        #if Type BOTs is N-points random, I have  many (from the start page) robots to choose the answers
        #else I will fix only 3 robots to make 3 maximum reference points  (depends on the Start Page selection of BOTs points)

        out = self.GetRobotComparisons(pf, front_fit, robot_lst, exp, num_block)

        #return robot_lst[6].PrintStyle()

        #return out

    def RunRobotsAll(self, exp, num_block, ga, my_world, pf, pop, problem='A'):

        #pop=[]
        out=[]
        robot_lst=[]
        #get Pareto Front
        #pf=self.GetFront()


        #fake population
        # pop=ga.SetPopulationFake(my_world)
        # del pop[len(pf):]
        #
        # #replace by front
        # count=0
        # for ind in pf:
        #    pop[count][0] = ind
        #    count +=1
        #
        # # set FITNESS to the entire front
        # fitnesses = ga.GetFitness(pop)
        # ga.AttachFitness(pop,fitnesses)
        #

        #create list of fitness
        front_fit=[]
        for i in pf:
            front_fit.append([i.fitness.values[0],i.fitness.values[1]])
        pop_fit=[]
        for i in pop:
            pop_fit.append([i.fitness.values[0],i.fitness.values[1]])


        #####################
        #GetRobotComparisons#
        #####################


        #get a list of ROBOT PLAYERS
        if num_block>1:
            #get plays from first round
            k=Play.objects.all().filter(play_experiment=exp.id, level=1, play_player__type = 'C').values('play_player').distinct()

            for i in k:
                robot_lst.append(Player.objects.get(id=i['play_player']))

        else:
            #robot creation: it must be created first
            #create Robots
            robot_lst = self.CreateRobots(exp.num_robots, exp.bots_points, problem)

        #Here I have the robot list
        #if Type BOTs is N-points random, I have  many (from the start page) robots to choose the answers
        #else I will fix only 3 robots to make 3 maximum reference points  (depends on the Start Page selection of BOTs points)

        out = self.GetRobotComparisonsAll(pop ,pf, front_fit, pop_fit, robot_lst, exp, num_block)

        #return robot_lst[6].PrintStyle()

        #return out


    #create NUM robots
    def CreateRobots(self,num_robots, type, problem='A'):
        #list of robots
        robot_lst=[]

        #minimum is 3 robots
        if num_robots < 3:
            num_robots = 3

        ###############
        #create robots#
        ###############

        #change this .... not now ... i am in a rush

        #####################
        #CREATE BASIC ROBOTS#
        #####################
        robot = Player.objects.filter(objective1_pref=1, type='C')
        #create the 3 basic styles: Pro Cost; Anti Cost (pro Prod); Random (in the middle)
        if not robot:
            #100% COST
            if problem == 'A':
                robot = Player(username='Robot1.00_0', email='robot@noemail.com', password='robot', schooling='robot',
                               gender='robot', age=2001, name='Robot 1.00/0', type='C', objective1_pref=1,f1_pref=60, f2_pref=-80)
            #Zorro
            elif problem =='G' or problem =='H' or problem =='I'  or problem=='J' or problem=='L' or problem=='M':
                robot = Player(username='Robot1.00_0', email='robot@noemail.com', password='robot', schooling='robot',
                               gender='robot', age=2001, name='Robot 1.00/0', type='C', objective1_pref=1,f1_pref=0, f2_pref=0, f3_pref=0)

            else:
                robot = Player(username='Robot1.00_0', email='robot@noemail.com', password='robot', schooling='robot',
                               gender='robot', age=2001, name='Robot 1.00/0', type='C', objective1_pref=1,f1_pref=0.6, f2_pref=0)
            robot.save()

        robot = Player.objects.filter(objective1_pref=0, type='C')
        if not robot:
            #0% Cost
            if problem == 'A':
                robot = Player(username='Robot0_1.00', email='robot@noemail.com', password='robot', schooling='robot',
                               gender='robot', age=2001, name='Robot 0/1.00', type='C', objective1_pref=0,f1_pref=110, f2_pref=-200)
            #Zorro
            elif problem =='G' or problem =='H' or problem =='I'  or problem=='J' or problem=='L' or problem=='M':
                robot = Player(username='Robot0_1.00', email='robot@noemail.com', password='robot', schooling='robot',
                               gender='robot', age=2001, name='Robot 0/1.00', type='C', objective1_pref=0,f1_pref=0.5, f2_pref=0.5, f3_pref=0.5)
            else:
                robot = Player(username='Robot0_1.00', email='robot@noemail.com', password='robot', schooling='robot',
                               gender='robot', age=2001, name='Robot 0/1.00', type='C', objective1_pref=0,f1_pref=0, f2_pref=0.6)
            robot.save()

        robot = Player.objects.filter(objective1_pref=.5, type='C')
        if not robot:
            #.5 COST
            if problem == 'A':
                robot = Player(username='Robot0.50_0.50', email='robot@noemail.com', password='robot', schooling='robot',
                               gender='robot', age=2001, name='Robot 0.50/0.50', type='C', objective1_pref=.5,f1_pref=60, f2_pref=-180)
            elif problem =='G' or problem =='H' or problem =='I'  or problem=='J' or problem=='L' or problem=='M':
                robot = Player(username='Robot0.50_0.50', email='robot@noemail.com', password='robot', schooling='robot',
                               gender='robot', age=2001, name='Robot 0.50/0.50', type='C', objective1_pref=.5,f1_pref=0.9, f2_pref=0.9, f3_pref=0.9)
            else:
                robot = Player(username='Robot0.50_0.50', email='robot@noemail.com', password='robot', schooling='robot',
                               gender='robot', age=2001, name='Robot 0.50/0.50', type='C', objective1_pref=.5,f1_pref=0.2, f2_pref=0.2)
            robot.save()
        #####################
        #####################



        if type == 'A':
            robot = Player.objects.get(objective1_pref=.5, type='C')
            robot_lst.append(robot)

        elif type == 'B':

            robot = Player.objects.get(objective1_pref=1, type='C')
            robot_lst.append(robot)
            robot = Player.objects.get(objective1_pref=0, type='C')
            robot_lst.append(robot)

        elif type == 'C':

            robot = Player.objects.get(objective1_pref=1, type='C')
            robot_lst.append(robot)
            robot = Player.objects.get(objective1_pref=0, type='C')
            robot_lst.append(robot)
            robot = Player.objects.get(objective1_pref=.5, type='C')
            robot_lst.append(robot)

        #if Type Bots is N-points random, let many (from the start page) robots choose the answers
        #else I will fix only 3 robots to make 3 maximum reference points
        elif type == 'D':

            robot = Player.objects.get(objective1_pref=1, type='C')
            robot_lst.append(robot)
            robot = Player.objects.get(objective1_pref=0, type='C')
            robot_lst.append(robot)
            robot = Player.objects.get(objective1_pref=.5, type='C')
            robot_lst.append(robot)
            #create more random robots
            for i in xrange(num_robots-3):
                p = round(random.uniform(0, 1.0), 2)
                robot = Player.objects.filter(objective1_pref=p, type='C')
                if not robot:
                    #cria
                    robot = Player(username='Robot'+str(p)+'_'+str(1-p), email='robot@noemail.com', password='robot', schooling='robot',
                                   gender='robot', age=2001, name='Robot '+str(p)+'/'+str(1-p), type='C', objective1_pref=p)
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

        robot_users =[]

        #create clean structure
        for i in xrange(front_size):
            clean_answer.append(deepcopy(clean_individual_answer))
        #clean answer done!

        #copy clean answer
        robot_comparison=deepcopy(clean_answer)

        #print "PARETO front dentro Comp: " + str(front_size)

        #Zorro - check 3D
        if exp.type_prob == 'G' or exp.type_prob == 'H' or exp.type_prob == 'I'  or exp.type_prob=='J' or exp.type_prob=='L' or exp.type_prob=='M':

            #for each Robot, get the comparisons
            if exp.bots_points == 'A':

                ref0=[robot_lst[0].f1_pref, robot_lst[0].f2_pref, robot_lst[0].f3_pref]
                #here I know who  is Robot 50% Cost : it is robot_lst[2]
                #loop from 50 to 100 random
                i=0
                for j in xrange(randint(50,100)):

                    #choose 2 points: random elements for ind1 and ind2 FROM front
                    a = randint(0,front_size-1)
                    b = randint(0,front_size-1)
                    while a == b:
                        b = randint(0,front_size-1)

                    #keep their indexes
                    #ask for comparisons to robo_lst[i]
                    ans = bot.TakeDecisionDist3D(ref0, front_fit[a], front_fit[b])

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
                    #for bench ... the chromossome info is wrong... but ok ... I do not need it for Benchmark
                    # if a > 95 or b > 95:
                    #     print "sss pauuuuuu " + str(front_size)
                    play = Play(play_experiment=exp, play_player=robot_lst[i], answer_time=0, points=0, answer=ans,
                                level=num_block, chromosomeOne= pop[a][0], chromosomeOneIndex=a, chromosomeTwo= pop[b][0], chromosomeTwoIndex=b)
                    play.save()

                all_comparisons.append(deepcopy(robot_comparison))

            elif exp.bots_points == 'B':

                ref0=[robot_lst[0].f1_pref, robot_lst[0].f2_pref, robot_lst[0].f3_pref]
                ref1=[robot_lst[1].f1_pref, robot_lst[1].f2_pref, robot_lst[1].f3_pref]
                #here I know who is Robot 100% Cost AND Robot 0% of Cost : it is robot_lst[0] and robot_lst[1]
                #loop from 100 to 200 random
                for j in xrange(randint(100,200)):

                    #choose 2 points: random elements for ind1 and ind2 FROM front
                    a = randint(0,front_size-1)
                    b = randint(0,front_size-1)
                    while a == b:
                        b = randint(0,front_size-1)

                    #keep their indexes
                    #ask for comparisons to robo_lst[i], choose between 2 randomly
                    coin = round(random.uniform(0, 1.0), 2)
                    if coin <= .5:
                        ans = bot.TakeDecisionDist3D(ref0, front_fit[a], front_fit[b])
                        i=0
                    else:
                        ans = bot.TakeDecisionDist3D(ref1, front_fit[a], front_fit[b])
                        i=1


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

            elif exp.bots_points == 'C':

                ref0=[robot_lst[0].f1_pref, robot_lst[0].f2_pref, robot_lst[0].f3_pref]
                ref1=[robot_lst[1].f1_pref, robot_lst[1].f2_pref, robot_lst[1].f3_pref]
                ref2=[robot_lst[2].f1_pref, robot_lst[2].f2_pref, robot_lst[2].f3_pref]

                #loop from 100 to 200 random
                for j in xrange(randint(150,300)):

                    #choose 2 points: random elements for ind1 and ind2 FROM front
                    a = randint(0,front_size-1)
                    b = randint(0,front_size-1)
                    while a == b:
                        b = randint(0,front_size-1)

                    #keep their indexes
                    #ask for comparisons to robo_lst[i], choose between 2 randomly
                    coin = round(random.uniform(0, 1.0), 2)
                    if coin <= .33:
                        ans = bot.TakeDecisionDist3D(ref0, front_fit[a], front_fit[b])
                        i=0
                    elif coin <= .66:
                        ans = bot.TakeDecisionDist3D(ref1, front_fit[a], front_fit[b])
                        i=1
                    else:
                        ans = bot.TakeDecisionDist3D(ref2, front_fit[a], front_fit[b])
                        i=2


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

            else:

                for i in xrange(robot_size):

                    #so i is the robot index

                    #loop from 1 to 10 random
                    for j in xrange(randint(1,10)):

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




        else:

            #for each Robot, get the comparisons
            if exp.bots_points == 'A':

                ref0=[robot_lst[0].f1_pref, robot_lst[0].f2_pref]
                #here I know who  is Robot 50% Cost : it is robot_lst[2]
                #loop from 50 to 100 random
                i=0
                for j in xrange(randint(50,100)):

                    #choose 2 points: random elements for ind1 and ind2 FROM front
                    a = randint(0,front_size-1)
                    b = randint(0,front_size-1)
                    while a == b:
                        b = randint(0,front_size-1)

                    #keep their indexes
                    #ask for comparisons to robo_lst[i]
                    ans = bot.TakeDecisionDist(ref0, front_fit[a], front_fit[b])

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
                    #for bench ... the chromossome info is wrong... but ok ... I do not need it for Benchmark
                    # if a > 95 or b > 95:
                    #     print "sss pauuuuuu " + str(front_size)
                    play = Play(play_experiment=exp, play_player=robot_lst[i], answer_time=0, points=0, answer=ans,
                                level=num_block, chromosomeOne= pop[a][0], chromosomeOneIndex=a, chromosomeTwo= pop[b][0], chromosomeTwoIndex=b)
                    play.save()

                all_comparisons.append(deepcopy(robot_comparison))

            elif exp.bots_points == 'B':

                ref0=[robot_lst[0].f1_pref, robot_lst[0].f2_pref]
                ref1=[robot_lst[1].f1_pref, robot_lst[1].f2_pref]
                #here I know who is Robot 100% Cost AND Robot 0% of Cost : it is robot_lst[0] and robot_lst[1]
                #loop from 100 to 200 random
                for j in xrange(randint(100,200)):

                    #choose 2 points: random elements for ind1 and ind2 FROM front
                    a = randint(0,front_size-1)
                    b = randint(0,front_size-1)
                    while a == b:
                        b = randint(0,front_size-1)

                    #keep their indexes
                    #ask for comparisons to robo_lst[i], choose between 2 randomly
                    coin = round(random.uniform(0, 1.0), 2)
                    if coin <= .5:
                        ans = bot.TakeDecisionDist(ref0, front_fit[a], front_fit[b])
                        i=0
                    else:
                        ans = bot.TakeDecisionDist(ref1, front_fit[a], front_fit[b])
                        i=1


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

            elif exp.bots_points == 'C':

                ref0=[robot_lst[0].f1_pref, robot_lst[0].f2_pref]
                ref1=[robot_lst[1].f1_pref, robot_lst[1].f2_pref]
                ref2=[robot_lst[2].f1_pref, robot_lst[2].f2_pref]

                #loop from 100 to 200 random
                for j in xrange(randint(300,600)):

                    #choose 2 points: random elements for ind1 and ind2 FROM front
                    a = randint(0,front_size-1)
                    b = randint(0,front_size-1)
                    while a == b:
                        b = randint(0,front_size-1)

                    #keep their indexes
                    #ask for comparisons to robo_lst[i], choose between 2 randomly
                    coin = round(random.uniform(0, 1.0), 2)
                    if coin <= .33:
                        ans = bot.TakeDecisionDist(ref0, front_fit[a], front_fit[b])
                        i=0
                    elif coin <= .66:
                        ans = bot.TakeDecisionDist(ref1, front_fit[a], front_fit[b])
                        i=1
                    else:
                        ans = bot.TakeDecisionDist(ref2, front_fit[a], front_fit[b])
                        i=2


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

            else:

                for i in xrange(robot_size):

                    #so i is the robot index

                    #loop from 1 to 10 random
                    for j in xrange(randint(1,10)):

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



    #create Robots comparisons
    def GetRobotComparisonsAll(self, pop,  front, front_fit, pop_fit, robot_lst, exp, num_block):
        #example of row
        # [first_robot] = [...] = [ [8,9],[0,4], ..., [5,6] ]
        #first_robot[0][0]= 8
        #first_robot[0][1]= 9

        bot = Robot()
        clean_individual_answer=[0,0]
        clean_answer=[]
        front_size = len(front)
        pop_size = len(pop)
        robot_size = len(robot_lst)

        all_comparisons = []
        robot_comparison = []

        robot_users =[]

        #create clean structure
        for i in xrange(pop_size):
            clean_answer.append(deepcopy(clean_individual_answer))
        #clean answer done!

        #copy clean answer
        robot_comparison=deepcopy(clean_answer)

        #manage indexes of front and population
        PF_indexes=[]
        count=0
        for i in pop:
            if i in front:
                PF_indexes.append(count)
            count +=1
        #PF_index has all the individuals inside Front



        #for each Robot, get the comparisons
        if exp.bots_points == 'A':

            ref0=[robot_lst[0].f1_pref, robot_lst[0].f2_pref]
            #here I know who  is Robot 50% Cost : it is robot_lst[2]
            #loop from 50 to 100 random
            i=0
            for j in xrange(randint(100,200)):


                ##########################
                #choose 2 points: random elements for ind1 and ind2 FROM front and population
                #a receives one from the front
                #b receives one from the population - different from "a" itself
                #OBS: here, as it is a robot, a will be always the front and b the population... because robot does not have conscience so far
                a = randint(0,len(PF_indexes)-1) #get one ind. that belongs to front
                b = randint(0,pop_size-1) #get anyone

                #replace a for the real index
                a = PF_indexes[a]

                #if they are the same, do it again
                while a == b:
                    b = randint(0,pop_size-1)

                #keep their indexes
                #ask for comparisons to robo_lst[i]
                ans = bot.TakeDecisionDist(ref0, pop_fit[a], pop_fit[b])
                ###########################################################################

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
                ##########AQUIIII################
                play = Play(play_experiment=exp, play_player=robot_lst[i], answer_time=0, points=0, answer=ans,
                            level=num_block, chromosomeOne= pop[a][0], chromosomeOneIndex=a, chromosomeTwo= pop[b][0], chromosomeTwoIndex=b)
                play.save()

            all_comparisons.append(deepcopy(robot_comparison))

        elif exp.bots_points == 'B':

            ref0=[robot_lst[0].f1_pref, robot_lst[0].f2_pref]
            ref1=[robot_lst[1].f1_pref, robot_lst[1].f2_pref]
            #here I know who is Robot 100% Cost AND Robot 0% of Cost : it is robot_lst[0] and robot_lst[1]
            #loop from 100 to 200 random
            for j in xrange(randint(200,300)):

                ##########################
                #choose 2 points: random elements for ind1 and ind2 FROM front and population
                #a receives one from the front
                #b receives one from the population - different from "a" itself
                #OBS: here, as it is a robot, a will be always the front and b the population... because robot does not have conscience so far
                a = randint(0,len(PF_indexes)-1) #get one ind. that belongs to front
                b = randint(0,pop_size-1) #get anyone

                #replace a for the real index
                a = PF_indexes[a]

                #if they are the same, do it again
                while a == b:
                    b = randint(0,pop_size-1)

                #keep their indexes
                #ask for comparisons to robo_lst[i], choose between 2 randomly
                coin = round(random.uniform(0, 1.0), 2)
                if coin <= .5:
                    ans = bot.TakeDecisionDist(ref0, pop_fit[a], pop_fit[b])
                    i=0
                else:
                    ans = bot.TakeDecisionDist(ref1, pop_fit[a], pop_fit[b])
                    i=1
                ###########################################################################

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
                  ##########AQUIIII################
                play = Play(play_experiment=exp, play_player=robot_lst[i], answer_time=0, points=0, answer=ans,
                            level=num_block, chromosomeOne= pop[a][0], chromosomeOneIndex=a, chromosomeTwo= pop[b][0], chromosomeTwoIndex=b)
                play.save()

            all_comparisons.append(deepcopy(robot_comparison))

        elif exp.bots_points == 'C':

            ref0=[robot_lst[0].f1_pref, robot_lst[0].f2_pref]
            ref1=[robot_lst[1].f1_pref, robot_lst[1].f2_pref]
            ref2=[robot_lst[2].f1_pref, robot_lst[2].f2_pref]

            #loop from 100 to 200 random
            for j in xrange(randint(300,400)):

                ##########################
                #choose 2 points: random elements for ind1 and ind2 FROM front and population
                #a receives one from the front
                #b receives one from the population - different from "a" itself
                #OBS: here, as it is a robot, a will be always the front and b the population... because robot does not have conscience so far
                a = randint(0,len(PF_indexes)-1) #get one ind. that belongs to front
                b = randint(0,pop_size-1) #get anyone

                #replace a for the real index
                a = PF_indexes[a]

                #if they are the same, do it again
                while a == b:
                    b = randint(0,pop_size-1)

                #keep their indexes
                #ask for comparisons to robo_lst[i], choose between 2 randomly
                coin = round(random.uniform(0, 1.0), 2)
                if coin <= .33:
                    ans = bot.TakeDecisionDist(ref0, pop_fit[a], pop_fit[b])
                    i=0
                elif coin <= .66:
                    ans = bot.TakeDecisionDist(ref1, pop_fit[a], pop_fit[b])
                    i=1
                else:
                    ans = bot.TakeDecisionDist(ref2, pop_fit[a], pop_fit[b])
                    i=2

                ###########################################################################


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
                  ##########AQUIIII################
                play = Play(play_experiment=exp, play_player=robot_lst[i], answer_time=0, points=0, answer=ans,
                            level=num_block, chromosomeOne= pop[a][0], chromosomeOneIndex=a, chromosomeTwo= pop[b][0], chromosomeTwoIndex=b)
                play.save()

            all_comparisons.append(deepcopy(robot_comparison))

        else:

            for i in xrange(robot_size):

                #so i is the robot index

                #loop from 1 to 10 random
                for j in xrange(randint(10,20)):

                    ##########################
                    #choose 2 points: random elements for ind1 and ind2 FROM front and population
                    #a receives one from the front
                    #b receives one from the population - different from "a" itself
                    #OBS: here, as it is a robot, a will be always the front and b the population... because robot does not have conscience so far
                    a = randint(0,len(PF_indexes)-1) #get one ind. that belongs to front
                    b = randint(0,pop_size-1) #get anyone

                    #replace a for the real index
                    a = PF_indexes[a]


                    #if they are the same, do it again
                    while a == b:
                        b = randint(0,pop_size-1)



                    #keep their indexes
                    #ask for comparisons to robo_lst[i]
                    k=robot_lst[i].objective1_pref
                    ans = bot.TakeDecision(k, pop_fit[a], pop_fit[b])
                    ###########################################################################


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
                      ##########AQUIIII################
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

    def seeBiGaussiansContour(self):

        from axes3Dedited import Axes3D

        import numpy as np
        from matplotlib.mlab import bivariate_normal
        import matplotlib.pyplot as plt
        import matplotlib.mlab as mlab



        #get last ID of the Experiment
        exp = Experiment.objects.latest('id')
        #get last generation inserted
        gen = Generation.objects.latest('id')
        #get World
        wrl = exp.world

        #define axis of chart
        if exp.type_prob == 'A':
            delta = wrl.delta
            x = np.arange(0, wrl.x_line, delta)
            y = np.arange(wrl.y_line, 0, delta)
        else:
            delta = 0.05
            x = np.arange(-0.1, 1.1, delta)
            y = np.arange(-0.1, 1.1, delta)
        X, Y = np.meshgrid(x, y)

        # delta = 0.02
        # x = np.arange(55.0, 59.0, delta)
        # y = np.arange(50.0, 58.0, delta)
        # X, Y = np.meshgrid(x, y)
        #

        #get objects from database
        objs = []
        means = literal_eval(gen.means)
        covar = literal_eval(gen.covar)


        for i in xrange(len(means)):
            z = bivariate_normal(X, Y, sigmax=np.sqrt(covar[i][0]), sigmay=np.sqrt(covar[i][0]),mux=means[i][0], muy=means[i][1])
            objs.append(z)


        for i in xrange(len(objs)):
            if i==0:
                Z = objs[i]
            else:
                Z += objs[i]

        #Z = 10.0 * (Z2 + Z1 + Z3)
        Z = 10.0 * Z

        #now, use the gmm object to get prediction.
        clusteOBJ = Cluster()
        lst_points2D = literal_eval(gen.fitness_points2D)
        if exp.type_prob == 'A':
            t = np.array(lst_points2D, np.int32)
        else:
            t = np.array(lst_points2D, np.float)
        best_gmm = clusteOBJ.EM(t, gen.num_k)



        # delta = 0.02
        # x = np.arange(55.0, 59.0, delta)
        # y = np.arange(50.0, 58.0, delta)
        # X, Y = np.meshgrid(x, y)
        # Z1 = bivariate_normal(X, Y, 0.5, 0.5, 57.0, 52.0)
        # Z2 = bivariate_normal(X, Y, 0.7, 0.7, 57.0, 55.0)
        # # difference of Gaussians
        # Z = 10.0 * (Z2 - Z1)
        # # #plt.figure()
        # # CS = plt.contour(X, Y, Z)
        # # plt.clabel(CS, inline=1, fontsize=10)
        # #plt.title('Simplest default with labels')
        #
        # #plt.subplot(2, 1, 1)
        # plt=fig.add_subplot(111)
        # CS = plt.contour(X, Y, Z)
        # plt.clabel(CS, inline=1, fontsize=10)

        fig=Figure()
        #print cantour
        plt=fig.add_subplot(111)

 #print scatter
        colors=[]
        for i in best_gmm.predict(t):
            if i==0:
                colors.append('r')
            elif i==1:
                colors.append('b')
            elif i==2:
                colors.append('g')
            elif i==3:
                colors.append('c')
            elif i==4:
                colors.append('y')
            elif i==5:
                colors.append('m')
            else:
                colors.append('k')

        #ax = plt.gca()
        #ax.scatter(t[:,0], t[:,1], c=colors, alpha=0.8)
        plt.scatter(t[:,0], t[:,1], c=colors, alpha=0.8)


        CS = plt.contour(X, Y, Z)
        #plt.clabel(CS, inline=1, fontsize=10)
        plt.clabel(CS)



        fig.autofmt_xdate()
        canvas=FigureCanvas(fig)
        response=django.http.HttpResponse(content_type='image/png')
        canvas.print_png(response)
        return response

    def seeBiGaussiansBell(self):

        from axes3Dedited import Axes3D

        import numpy as np
        from matplotlib.mlab import bivariate_normal
        import matplotlib.pyplot as plt
        import matplotlib.mlab as mlab
        from pylab import *


        #get last ID of the Experiment
        exp = Experiment.objects.latest('id')
        #get last generation inserted
        gen = Generation.objects.latest('id')
        #get World
        wrl = exp.world

        #define axis of chart
        #define axis of chart
        if exp.type_prob == 'A':
            delta = wrl.delta
            x = np.arange(0, wrl.x_line, delta)
            y = np.arange(wrl.y_line, 0, delta)
        else:
            delta = 0.003
            x = np.arange(-0.1, 1.1, delta)
            y = np.arange(-0.1, 1.1, delta)
        X, Y = np.meshgrid(x, y)
        # delta = 0.02
        # x = np.arange(55.0, 59.0, delta)
        # y = np.arange(50.0, 58.0, delta)
        # X, Y = np.meshgrid(x, y)
        #

        #get objects from database
        objs = []
        means = literal_eval(gen.means)
        covar = literal_eval(gen.covar)


        for i in xrange(len(means)):
            z = bivariate_normal(X, Y, sigmax=np.sqrt(covar[i][0]), sigmay=np.sqrt(covar[i][0]),mux=means[i][0], muy=means[i][1])
            objs.append(z)


        for i in xrange(len(objs)):
            if i==0:
                Z = objs[i]
            else:
                Z += objs[i]

        #Z = 10.0 * (Z2 + Z1 + Z3)
        Z = 10.0 * Z

        #print bell
        fig=Figure()
        plt=fig.add_subplot(111)
        ax = fig.add_subplot(111, projection='3d')
        ax.plot_surface(X, Y, Z, cmap= wrl.cmap)


        # from axes3Dedited import Axes3D
        #
        # import numpy as np
        # from matplotlib.mlab import bivariate_normal
        # import matplotlib.pyplot as plt
        # import matplotlib.mlab as mlab
        # from pylab import *
        #
        # fig=Figure()
        #
        # delta = 0.02
        # x = np.arange(55.0, 59.0, delta)
        # y = np.arange(50.0, 58.0, delta)
        # X, Y = np.meshgrid(x, y)
        # Z1 = bivariate_normal(X, Y, 0.5, 0.5, 57.0, 52.0)
        # Z2 = bivariate_normal(X, Y, 0.7, 0.7, 57.0, 55.0)
        # # difference of Gaussians
        # Z = 10.0 * (Z2 + Z1)
        # # #plt.figure()
        # # CS = plt.contour(X, Y, Z)
        # # plt.clabel(CS, inline=1, fontsize=10)
        # #plt.title('Simplest default with labels')
        #
        # #plt.subplot(2, 1, 1)
        # plt=fig.add_subplot(111)
        #
        # ax = fig.add_subplot(111, projection='3d')
        # ax.plot_surface(X, Y, Z, cmap= 'Greens')
        #title('Greens')
        #plt.show()

        fig.autofmt_xdate()
        canvas=FigureCanvas(fig)
        response=django.http.HttpResponse(content_type='image/png')
        canvas.print_png(response)
        pp = PdfPages('polls/static/fronts/bigauss_dist_color_' + str(exp.id) + '__' + str(gen.id) + '.pdf')
        pp.savefig(fig)
        pp.close()
        return response

    def seeBiGaussiansContour_b(self):

        from axes3Dedited import Axes3D

        import numpy as np
        from matplotlib.mlab import bivariate_normal
        import matplotlib.pyplot as plt
        import matplotlib.mlab as mlab



        #get last ID of the Experiment
        exp = Experiment.objects.latest('id')
        #get last generation inserted
        gen = Generation.objects.latest('id')
        #get World
        wrl = exp.world

        #define axis of chart
        if exp.type_prob == 'A':
            delta = wrl.delta
            x = np.arange(0, wrl.x_line, delta)
            y = np.arange(wrl.y_line, 0, delta)
        else:
            delta = 0.05
            x = np.arange(-0.1, 1.1, delta)
            y = np.arange(-0.1, 1.1, delta)
        X, Y = np.meshgrid(x, y)
        # delta = 0.02
        # x = np.arange(55.0, 59.0, delta)
        # y = np.arange(50.0, 58.0, delta)
        # X, Y = np.meshgrid(x, y)
        #

        #get objects from database
        objs = []
        means = literal_eval(gen.means)
        covar = literal_eval(gen.covar)


        for i in xrange(len(means)):
            z = bivariate_normal(X, Y, sigmax=np.sqrt(covar[i][0]), sigmay=np.sqrt(covar[i][0]),mux=means[i][0], muy=means[i][1])
            objs.append(z)


        for i in xrange(len(objs)):
            if i==0:
                Z = objs[i]
            else:
                Z += objs[i]

        #Z = 10.0 * (Z2 + Z1 + Z3)
        Z = 10.0 * Z

        #now, use the gmm object to get prediction.
        clusteOBJ = Cluster()
        lst_points2D = literal_eval(gen.fitness_points2D)
        if exp.type_prob == 'A':
            t = np.array(lst_points2D, np.int32)
        else:
            t = np.array(lst_points2D, np.float)
        best_gmm = clusteOBJ.EM(t, gen.num_k)



        # delta = 0.02
        # x = np.arange(55.0, 59.0, delta)
        # y = np.arange(50.0, 58.0, delta)
        # X, Y = np.meshgrid(x, y)
        # Z1 = bivariate_normal(X, Y, 0.5, 0.5, 57.0, 52.0)
        # Z2 = bivariate_normal(X, Y, 0.7, 0.7, 57.0, 55.0)
        # # difference of Gaussians
        # Z = 10.0 * (Z2 - Z1)
        # # #plt.figure()
        # # CS = plt.contour(X, Y, Z)
        # # plt.clabel(CS, inline=1, fontsize=10)
        # #plt.title('Simplest default with labels')
        #
        # #plt.subplot(2, 1, 1)
        # plt=fig.add_subplot(111)
        # CS = plt.contour(X, Y, Z)
        # plt.clabel(CS, inline=1, fontsize=10)

        fig=Figure()
        #print cantour
        plt=fig.add_subplot(111)

        #print scatter
        colors=[]
        for i in best_gmm.predict(t):
            if i==0:
                colors.append('r')
            elif i==1:
                colors.append('b')
            elif i==2:
                colors.append('g')
            elif i==3:
                colors.append('c')
            elif i==4:
                colors.append('y')
            elif i==5:
                colors.append('k')
            else:
                colors.append('m')

        #ax = plt.gca()
        #ax.scatter(t[:,0], t[:,1], c=colors, alpha=0.8)
        plt.scatter(t[:,0], t[:,1], c=colors, alpha=0.8)


        CS = plt.contour(X, Y, Z, colors='k')
        #plt.clabel(CS, inline=1, fontsize=10)
        plt.clabel(CS)



        fig.autofmt_xdate()
        canvas=FigureCanvas(fig)
        response=django.http.HttpResponse(content_type='image/png')
        canvas.print_png(response)
        return response

    def seeBiGaussiansBell_b(self):

        from axes3Dedited import Axes3D

        import numpy as np
        from matplotlib.mlab import bivariate_normal
        import matplotlib.pyplot as plt
        import matplotlib.mlab as mlab
        from pylab import *


        #get last ID of the Experiment
        exp = Experiment.objects.latest('id')
        #get last generation inserted
        gen = Generation.objects.latest('id')
        #get World
        wrl = exp.world

        #define axis of chart
        if exp.type_prob == 'A':
            delta = wrl.delta
            x = np.arange(0, wrl.x_line, delta)
            y = np.arange(wrl.y_line, 0, delta)
        else:
            delta = 0.003
            x = np.arange(-0.1, 1.1, delta)
            y = np.arange(-0.1, 1.1, delta)
        X, Y = np.meshgrid(x, y)
        # delta = 0.02
        # x = np.arange(55.0, 59.0, delta)
        # y = np.arange(50.0, 58.0, delta)
        # X, Y = np.meshgrid(x, y)
        #

        #get objects from database
        objs = []
        means = literal_eval(gen.means)
        covar = literal_eval(gen.covar)


        for i in xrange(len(means)):
            z = bivariate_normal(X, Y, sigmax=np.sqrt(covar[i][0]), sigmay=np.sqrt(covar[i][0]),mux=means[i][0], muy=means[i][1])
            objs.append(z)


        for i in xrange(len(objs)):
            if i==0:
                Z = objs[i]
            else:
                Z += objs[i]

        #Z = 10.0 * (Z2 + Z1 + Z3)
        Z = 10.0 * Z

        #print bell
        fig=Figure()
        plt=fig.add_subplot(111)
        ax = fig.add_subplot(111, projection='3d')
        ax.plot_surface(X, Y, Z, cmap= 'Greys')


        # from axes3Dedited import Axes3D
        #
        # import numpy as np
        # from matplotlib.mlab import bivariate_normal
        # import matplotlib.pyplot as plt
        # import matplotlib.mlab as mlab
        # from pylab import *
        #
        # fig=Figure()
        #
        # delta = 0.02
        # x = np.arange(55.0, 59.0, delta)
        # y = np.arange(50.0, 58.0, delta)
        # X, Y = np.meshgrid(x, y)
        # Z1 = bivariate_normal(X, Y, 0.5, 0.5, 57.0, 52.0)
        # Z2 = bivariate_normal(X, Y, 0.7, 0.7, 57.0, 55.0)
        # # difference of Gaussians
        # Z = 10.0 * (Z2 + Z1)
        # # #plt.figure()
        # # CS = plt.contour(X, Y, Z)
        # # plt.clabel(CS, inline=1, fontsize=10)
        # #plt.title('Simplest default with labels')
        #
        # #plt.subplot(2, 1, 1)
        # plt=fig.add_subplot(111)
        #
        # ax = fig.add_subplot(111, projection='3d')
        # ax.plot_surface(X, Y, Z, cmap= 'Greens')
        #title('Greens')
        #plt.show()

        fig.autofmt_xdate()
        canvas=FigureCanvas(fig)
        response=django.http.HttpResponse(content_type='image/png')
        canvas.print_png(response)
        pp = PdfPages('polls/static/fronts/bigauss_dist_b_' + str(exp.id) + '__' + str(gen.id) + '.pdf')
        pp.savefig(fig)
        pp.close()
        return response
        return response


    def kmeans(self):

        from axes3Dedited import Axes3D

        import numpy as np
        import matplotlib.pyplot as plt
        import matplotlib.mlab as mlab



        #get last ID of the Experiment
        exp = Experiment.objects.latest('id')
        #get last generation inserted
        gen = Generation.objects.latest('id')



        #get objects from database
        objs = []
        means = literal_eval(gen.centroids)

        #now, use the kmeans object to get prediction.
        clusteOBJ = Cluster()
        lst_points2D = literal_eval(gen.fitness_points2D)
        if exp.type_prob == 'A':
            t = np.array(lst_points2D, np.int32)
        else:
            t = np.array(lst_points2D, np.float)
        kmm = clusteOBJ.Kmeans(t, gen.num_k)

        fig=Figure()
        #print cantour
        plt=fig.add_subplot(111)

        colors=[]
        for i in kmm.predict(t):
            if i==0:
                colors.append('r')
            elif i==1:
                colors.append('b')
            elif i==2:
                colors.append('g')
            elif i==3:
                colors.append('c')
            elif i==4:
                colors.append('y')
            elif i==5:
                colors.append('k')
            else:
                colors.append('m')

        #ax = plt.gca()
        #ax.scatter(t[:,0], t[:,1], c=colors, alpha=0.8)
        plt.scatter(t[:,0], t[:,1], c=colors, alpha=0.8)

        # for j in kmm.cluster_centers_:
        #     plt.plot(j[0], j[1],  'gD', markersize=8)



        fig.autofmt_xdate()
        canvas=FigureCanvas(fig)
        response=django.http.HttpResponse(content_type='image/png')
        canvas.print_png(response)

        pp = PdfPages('polls/static/fronts/front_color_kmeans_group_' + str(exp.id) + '__' + str(gen.id) + '.pdf')
        pp.savefig(fig)
        pp.close()

        return response

    def kmeans_b(self):

        from axes3Dedited import Axes3D

        import numpy as np
        import matplotlib.pyplot as plt
        import matplotlib.mlab as mlab



        #get last ID of the Experiment
        exp = Experiment.objects.latest('id')
        #get last generation inserted
        gen = Generation.objects.latest('id')


        #get objects from database

        means = literal_eval(gen.centroids)

        #now, use the kmeans object to get prediction.
        clusteOBJ = Cluster()
        lst_points2D = literal_eval(gen.fitness_points2D)
        if exp.type_prob == 'A':
            t = np.array(lst_points2D, np.int32)
        else:
            t = np.array(lst_points2D, np.float)
        kmm = clusteOBJ.Kmeans(t, gen.num_k)

        fig=Figure()
        #print cantour
        plt=fig.add_subplot(111)

        mark=''
        count=0
        for i in kmm.predict(t):
            if i==0:
                mark='o'
            elif i==1:
                mark='x'
            elif i==2:
                mark='^'
            elif i==3:
                mark='*'
            elif i==4:
                mark='D'
            elif i==5:
                mark='1'
            else:
                mark='s'
            plt.scatter(t[count][0], t[count][1], marker=mark, c='k', alpha=0.8)
            count +=1

            # if i==0:
            #     mark.append('o')
            # elif i==1:
            #     mark.append('*')
            # elif i==2:
            #     mark.append('s')
            # elif i==3:
            #     mark.append('v')
            # elif i==4:
            #     mark.append('+')
            # elif i==5:
            #     mark.append('x')
            # else:
            #     mark.append('^')

        #ax = plt.gca()
        #ax.scatter(t[:,0], t[:,1], c=colors, alpha=0.8)
        #plt.scatter(t[:,0], t[:,1], marker=mark, c='k', alpha=0.8)

        # for j in kmm.cluster_centers_:
        #     plt.plot(j[0], j[1],  'kD', markersize=8)



        fig.autofmt_xdate()
        canvas=FigureCanvas(fig)
        response=django.http.HttpResponse(content_type='image/png')
        canvas.print_png(response)

        pp = PdfPages('polls/static/fronts/front_b_kmeans_group_' + str(exp.id) + '__' + str(gen.id) + '.pdf')
        pp.savefig(fig)
        pp.close()

        return response

    def seefinalfront_kmeans(self):


        import numpy as np

        #get last ID of the Experiment
        exp = Experiment.objects.latest('id')

        #get World config
        wrl = exp.world
        gameworld = GameWorld.objects.get(id=wrl.id)
        my_world = World(gameworld.m,gameworld.n)
        my_world.experimentTYPE = exp.type

        #declare Genetic Algorithm for the problem
        ga = GA(my_world,exp.CXPB, exp.MUTPB, exp.NGEN, exp.NPOP,exp.type_prob)

        #number of the next block of generations
        gen = Generation.objects.latest('id')

        # gen_guess = Generation.objects.get(id=gen.id-1)

        #########################
        pfront = self.GetFront()

        #fake population
        pop_front=ga.SetPopulationFakeBench(my_world, problem=exp.type_prob)
        del pop_front[len(pfront):]

        #replace by front
        count=0
        for ind in pfront:

           chromo = literal_eval(ind)
           for i in xrange(len(chromo)):
               pop_front[count][i] = chromo[i]
           count +=1

        # set FITNESS to the entire front
        fitnesses = ga.GetFitness(pop_front)
        ga.AttachFitness(pop_front,fitnesses)

        #create list of fitness
        front_fit=[]
        for i in pop_front:
            front_fit.append([i.fitness.values[0],i.fitness.values[1]])


        ##########################

        #fake population
        pop=ga.SetPopulationFakeBench(my_world, problem=exp.type_prob)

        #get the LAST Population
        count=0
        for ind in Population.objects.all().filter(generation=gen.id):

           chromo_original = literal_eval(ind.chromosome_original)
           for i in xrange(len(chromo)):
               pop[count][i]= chromo_original[i]
           count +=1
        #ok, the real population!!
        fitnesses = ga.GetFitness(pop)
        ga.AttachFitness(pop,fitnesses)

        pop_fit=[]
        for i in pop:
            pop_fit.append([i.fitness.values[0],i.fitness.values[1]])


        #################################
        means = literal_eval(gen.centroids)

        if exp.type_prob == 'A':
            t = np.array(front_fit, np.int32)
            u = np.array(pop_fit, np.int32)
        else:
            t = np.array(front_fit, np.float)
            u = np.array(pop_fit, np.float)

        fig=Figure()
        #print cantour
        plt=fig.add_subplot(111)

        plt.scatter(u[:,0], u[:,1], c='mistyrose', alpha=1, label='Pareto front')
        plt.scatter(t[:,0], t[:,1], c='cyan', alpha=1, label='relevant individuals')


        #kmeans
        # for j in means:
        #     plt.plot(j[0], j[1],  'gD', markersize=8)
        # plt.plot(means[0][0]+ 0.01, means[0][1]+0.08,  'gD', c='k', markersize=8)
        # plt.plot(means[1][0]+ 0.02, means[1][1]+0.04,  'gD', c='k', markersize=8)
        # plt.plot(means[2][0]+0.05, means[2][1]-0.01,  'gD', c='k', markersize=8)


        plt.legend()


        # plt.annotate('centroid', xy=(means[0][0], means[0][1]+0.15),  xycoords='data',
        #             xytext=(-60, 40), textcoords='offset points',
        #             arrowprops=dict(arrowstyle="->",
        #                             connectionstyle="arc,angleA=0,armA=30,rad=10"),
        #             )

        fig.autofmt_xdate()
        canvas=FigureCanvas(fig)
        response=django.http.HttpResponse(content_type='image/png')
        canvas.print_png(response)
        pp = PdfPages('polls/static/fronts/front_color_' + str(exp.id) + '__' + str(gen.id) + '.pdf')
        pp.savefig(fig)
        pp.close()

        return response


    def seefinalfront_kmeans_b(self):

        import numpy as np

        #get last ID of the Experiment
        exp = Experiment.objects.latest('id')

        #get World config
        wrl = exp.world
        gameworld = GameWorld.objects.get(id=wrl.id)
        my_world = World(gameworld.m,gameworld.n)
        my_world.experimentTYPE = exp.type

        #declare Genetic Algorithm for the problem
        ga = GA(my_world,exp.CXPB, exp.MUTPB, exp.NGEN, exp.NPOP,exp.type_prob)

        #number of the next block of generations
        gen = Generation.objects.latest('id')

        # gen_guess = Generation.objects.get(id=gen.id-1)

        #########################
        pfront = self.GetFront()

        #fake population
        pop_front=ga.SetPopulationFakeBench(my_world, problem=exp.type_prob)
        del pop_front[len(pfront):]

        #replace by front
        count=0
        for ind in pfront:

           chromo = literal_eval(ind)
           for i in xrange(len(chromo)):
               pop_front[count][i] = chromo[i]
           count +=1

        # set FITNESS to the entire front
        fitnesses = ga.GetFitness(pop_front)
        ga.AttachFitness(pop_front,fitnesses)

        #create list of fitness
        front_fit=[]
        for i in pop_front:
            front_fit.append([i.fitness.values[0],i.fitness.values[1]])


        ##########################

        #fake population
        pop=ga.SetPopulationFakeBench(my_world, problem=exp.type_prob)

        #get the LAST Population
        count=0
        for ind in Population.objects.all().filter(generation=gen.id):

           chromo_original = literal_eval(ind.chromosome_original)
           for i in xrange(len(chromo)):
               pop[count][i]= chromo_original[i]
           count +=1
        #ok, the real population!!
        fitnesses = ga.GetFitness(pop)
        ga.AttachFitness(pop,fitnesses)

        pop_fit=[]
        for i in pop:
            pop_fit.append([i.fitness.values[0],i.fitness.values[1]])


        #################################
        means = literal_eval(gen.centroids)

        if exp.type_prob == 'A':
            t = np.array(front_fit, np.int32)
            u = np.array(pop_fit, np.int32)
        else:
            t = np.array(front_fit, np.float)
            u = np.array(pop_fit, np.float)

        fig=Figure()
        #print cantour
        plt=fig.add_subplot(111)

        plt.scatter(u[:,0], u[:,1], c='lightgrey', alpha=1, label='Pareto front')
        plt.scatter(t[:,0], t[:,1], c='k', alpha=1, label='relevant individuals')


        #kmeans
        # for j in means:
        #     plt.plot(j[0], j[1],  'gD', c='k', markersize=8)
        plt.plot(means[0][0]+0.2, means[0][1]+0.2,  'gD', markersize=8, label='centroids')
        # plt.plot(means[1][0]+0.05, means[1][1],  'gD', markersize=8, label='centroids')

        plt.legend()

        #robot reference point for vote
        #setinha para centroid e para robot point

        fig.autofmt_xdate()
        canvas=FigureCanvas(fig)
        response=django.http.HttpResponse(content_type='image/png')
        canvas.print_png(response)
        pp = PdfPages('polls/static/fronts/front_b_' + str(exp.id) + '__' + str(gen.id) + '.pdf')
        pp.savefig(fig)
        pp.close()
        return response

    def seefinalfront_kmeans_line(self):




        import numpy as np

        #get last ID of the Experiment
        exp = Experiment.objects.latest('id')

        #get World config
        wrl = exp.world
        gameworld = GameWorld.objects.get(id=wrl.id)
        my_world = World(gameworld.m,gameworld.n)
        my_world.experimentTYPE = exp.type

        #declare Genetic Algorithm for the problem
        ga = GA(my_world,exp.CXPB, exp.MUTPB, exp.NGEN, exp.NPOP,exp.type_prob)

        #number of the next block of generations
        gen = Generation.objects.latest('id')

        #########################
        pfront = self.GetFront()

        #fake population
        pop_front=ga.SetPopulationFakeBench(my_world, problem=exp.type_prob)
        del pop_front[len(pfront):]

        #replace by front
        count=0
        for ind in pfront:

           chromo = literal_eval(ind)
           for i in xrange(len(chromo)):
               pop_front[count][i] = chromo[i]
           count +=1

        # set FITNESS to the entire front
        fitnesses = ga.GetFitness(pop_front)
        ga.AttachFitness(pop_front,fitnesses)

        #create list of fitness
        front_fit=[]
        for i in pop_front:
            front_fit.append([i.fitness.values[0],i.fitness.values[1]])


        ##########################

        #fake population
        pop=ga.SetPopulationFakeBench(my_world, problem=exp.type_prob)

        #get the LAST Population
        count=0
        for ind in Population.objects.all().filter(generation=gen.id):

           chromo_original = literal_eval(ind.chromosome_original)
           for i in xrange(len(chromo)):
               pop[count][i]= chromo_original[i]
           count +=1
        #ok, the real population!!
        fitnesses = ga.GetFitness(pop)
        ga.AttachFitness(pop,fitnesses)

        pop_fit=[]
        for i in pop:
            pop_fit.append([i.fitness.values[0],i.fitness.values[1]])
        pareto=tools.ParetoFront()
        pareto.update(pop)
        # pareto_fit=[]
        # for i in pop:
        #     pareto_fit.append([i.fitness.values[0],i.fitness.values[1]])
        paretoX_gen1 =[]
        paretoY_gen1=[]
        for j in pareto:
            paretoX_gen1.append(j.fitness.values[0])
            paretoY_gen1.append(j.fitness.values[1])


        #################################
        means = literal_eval(gen.centroids)

        if exp.type_prob == 'A':
            t = np.array(front_fit, np.int32)
            # u = np.array(pop_fit, np.int32)
            # u = np.array(pareto_fit, np.int32)
        else:
            t = np.array(front_fit, np.float)
            # u = np.array(pop_fit, np.float)
            # u = np.array(pareto_fit, np.float)

        fig=Figure()
        #print cantour
        plt=fig.add_subplot(111)

        # plt.plot(u[:,0], u[:,1], c='lightgrey', alpha=1)
        plt.plot(paretoX_gen1, paretoY_gen1, c='darkgrey', alpha=1, label='Pareto front')
        plt.scatter(t[:,0], t[:,1], c='cyan', alpha=1, label='relevant individuals')



        #kmeans
        # for j in means:
        #     plt.plot(j[0], j[1],  'gD', markersize=8)
        # plt.plot(means[0][0]+ 0.01, means[0][1]+0.08,  'gD', c='k', markersize=8)
        # plt.plot(means[1][0]+ 0.02, means[1][1]+0.04,  'gD', c='k', markersize=8)
        # plt.plot(means[2][0]+0.05, means[2][1]-0.01,  'gD', c='k', markersize=8)


        plt.legend()

        # plt.annotate('centroid', xy=(means[0][0], means[0][1]+0.15),  xycoords='data',
        #             xytext=(-60, 40), textcoords='offset points',
        #             arrowprops=dict(arrowstyle="->",
        #                             connectionstyle="arc,angleA=0,armA=30,rad=10"),
        #             )

        #kmeans
        # for j in means:
        #     plt.plot(j[0], j[1],  'gD', markersize=8)


        fig.autofmt_xdate()
        canvas=FigureCanvas(fig)
        response=django.http.HttpResponse(content_type='image/png')
        canvas.print_png(response)
        pp = PdfPages('polls/static/fronts/front_line_' + str(exp.id) + '__' + str(gen.id) + '.pdf')
        pp.savefig(fig)
        pp.close()
        return response

    def seefinalfront_kmeans_line_b(self):


        import numpy as np

        #get last ID of the Experiment
        exp = Experiment.objects.latest('id')

        #get World config
        wrl = exp.world
        gameworld = GameWorld.objects.get(id=wrl.id)
        my_world = World(gameworld.m,gameworld.n)
        my_world.experimentTYPE = exp.type

        #declare Genetic Algorithm for the problem
        ga = GA(my_world,exp.CXPB, exp.MUTPB, exp.NGEN, exp.NPOP,exp.type_prob)

        #number of the next block of generations
        gen = Generation.objects.latest('id')

        #########################
        pfront = self.GetFront()

        #fake population
        pop_front=ga.SetPopulationFakeBench(my_world, problem=exp.type_prob)
        del pop_front[len(pfront):]

        #replace by front
        count=0
        for ind in pfront:

           chromo = literal_eval(ind)
           for i in xrange(len(chromo)):
               pop_front[count][i] = chromo[i]
           count +=1

        # set FITNESS to the entire front
        fitnesses = ga.GetFitness(pop_front)
        ga.AttachFitness(pop_front,fitnesses)

        #create list of fitness
        front_fit=[]
        for i in pop_front:
            front_fit.append([i.fitness.values[0],i.fitness.values[1]])


        ##########################

        #fake population
        pop=ga.SetPopulationFakeBench(my_world, problem=exp.type_prob)

        #get the LAST Population
        count=0
        for ind in Population.objects.all().filter(generation=gen.id):

           chromo_original = literal_eval(ind.chromosome_original)
           for i in xrange(len(chromo)):
               pop[count][i]= chromo_original[i]
           count +=1
        #ok, the real population!!
        fitnesses = ga.GetFitness(pop)
        ga.AttachFitness(pop,fitnesses)

        pop_fit=[]
        for i in pop:
            pop_fit.append([i.fitness.values[0],i.fitness.values[1]])
        pareto=tools.ParetoFront()
        pareto.update(pop)
        # pareto_fit=[]
        # for i in pop:
        #     pareto_fit.append([i.fitness.values[0],i.fitness.values[1]])
        paretoX_gen1 =[]
        paretoY_gen1=[]
        for j in pareto:
            paretoX_gen1.append(j.fitness.values[0])
            paretoY_gen1.append(j.fitness.values[1])


        #################################
        means = literal_eval(gen.centroids)

        if exp.type_prob == 'A':
            t = np.array(front_fit, np.int32)
            # u = np.array(pop_fit, np.int32)
            # u = np.array(pareto_fit, np.int32)
        else:
            t = np.array(front_fit, np.float)
            # u = np.array(pop_fit, np.float)
            # u = np.array(pareto_fit, np.float)

        fig=Figure()
        #print cantour
        plt=fig.add_subplot(111)

        # plt.plot(u[:,0], u[:,1], c='lightgrey', alpha=1)
        plt.plot(paretoX_gen1, paretoY_gen1, c='darkgrey', alpha=1, label='Pareto front')
        plt.scatter(t[:,0], t[:,1], c='k', alpha=1, label='relevant individuals')



        #kmeans
        # for j in means:
        #     plt.plot(j[0], j[1],  'gD', c='k', markersize=8)
        # plt.plot(means[0][0]+ 0.01, means[0][1]+0.08,  'gD', c='k', markersize=8)
        # plt.plot(means[1][0]+ 0.02, means[1][1]+0.04,  'gD', c='k', markersize=8)
        # plt.plot(means[2][0]+0.05, means[2][1]-0.01,  'gD', c='k', markersize=8)

        plt.legend()

        # plt.annotate('centroid', xy=(means[0][0]+ 0.01, means[0][1]+0.15),  xycoords='data',
        #             xytext=(-60, 40), textcoords='offset points',
        #             arrowprops=dict(arrowstyle="->",
        #                             connectionstyle="arc,angleA=0,armA=30,rad=10"),
        #             )

        #kmeans
        # for j in means:
        #     plt.plot(j[0], j[1],  'gD', markersize=8)


        fig.autofmt_xdate()
        canvas=FigureCanvas(fig)
        response=django.http.HttpResponse(content_type='image/png')
        canvas.print_png(response)
        pp = PdfPages('polls/static/fronts/front_line_b_' + str(exp.id) + '__' + str(gen.id) + '.pdf')
        pp.savefig(fig)
        pp.close()
        return response

    def seefinalfront_gauss(self):


        import numpy as np

        #get last ID of the Experiment
        exp = Experiment.objects.latest('id')

        #get World config
        wrl = exp.world
        gameworld = GameWorld.objects.get(id=wrl.id)
        my_world = World(gameworld.m,gameworld.n)
        my_world.experimentTYPE = exp.type

        #declare Genetic Algorithm for the problem
        ga = GA(my_world,exp.CXPB, exp.MUTPB, exp.NGEN, exp.NPOP,exp.type_prob)

        #number of the next block of generations
        gen = Generation.objects.latest('id')

        # gen_guess = Generation.objects.get(id=gen.id-1)

        #########################
        pfront = self.GetFront()

        #fake population
        pop_front=ga.SetPopulationFakeBench(my_world, problem=exp.type_prob)
        del pop_front[len(pfront):]

        #replace by front
        count=0
        for ind in pfront:

           chromo = literal_eval(ind)
           for i in xrange(len(chromo)):
               pop_front[count][i] = chromo[i]
           count +=1

        # set FITNESS to the entire front
        fitnesses = ga.GetFitness(pop_front)
        ga.AttachFitness(pop_front,fitnesses)

        #create list of fitness
        front_fit=[]
        for i in pop_front:
            front_fit.append([i.fitness.values[0],i.fitness.values[1]])


        ##########################

        #fake population
        pop=ga.SetPopulationFakeBench(my_world, problem=exp.type_prob)

        #get the LAST Population
        count=0
        for ind in Population.objects.all().filter(generation=gen.id):

           chromo_original = literal_eval(ind.chromosome_original)
           for i in xrange(len(chromo)):
               pop[count][i]= chromo_original[i]
           count +=1
        #ok, the real population!!
        fitnesses = ga.GetFitness(pop)
        ga.AttachFitness(pop,fitnesses)

        pop_fit=[]
        for i in pop:
            pop_fit.append([i.fitness.values[0],i.fitness.values[1]])


        #################################
        means = literal_eval(gen.means)

        if exp.type_prob == 'A':
            t = np.array(front_fit, np.int32)
            u = np.array(pop_fit, np.int32)
        else:
            t = np.array(front_fit, np.float)
            u = np.array(pop_fit, np.float)

        fig=Figure()
        #print cantour
        plt=fig.add_subplot(111)

        plt.scatter(u[:,0], u[:,1], c='mistyrose', alpha=1, label='Pareto front')
        plt.scatter(t[:,0], t[:,1], c='cyan', alpha=1, label='relevant individuals')



        # plt.plot(means[0][0]+ 0.01, means[0][1]+0.08,  'gD', c='k', markersize=8)
        # plt.plot(means[1][0]+ 0.02, means[1][1]+0.04,  'gD', c='k', markersize=8)
        # plt.plot(means[2][0]+0.05, means[2][1]-0.01,  'gD', c='k', markersize=8)


        plt.legend()


        # plt.annotate('gaussian mean', xy=(means[0][0], means[0][1]+0.3),  xycoords='data',
        #             xytext=(-60, 40), textcoords='offset points',
        #             arrowprops=dict(arrowstyle="->",
        #                             connectionstyle="arc,angleA=0,armA=30,rad=10"),
        #             )

        fig.autofmt_xdate()
        canvas=FigureCanvas(fig)
        response=django.http.HttpResponse(content_type='image/png')
        canvas.print_png(response)
        pp = PdfPages('polls/static/fronts/front_color_' + str(exp.id) + '__' + str(gen.id) + '.pdf')
        pp.savefig(fig)
        pp.close()

        return response


    def seefinalfront_gauss_b(self):


        import numpy as np

        #get last ID of the Experiment
        exp = Experiment.objects.latest('id')

        #get World config
        wrl = exp.world
        gameworld = GameWorld.objects.get(id=wrl.id)
        my_world = World(gameworld.m,gameworld.n)
        my_world.experimentTYPE = exp.type

        #declare Genetic Algorithm for the problem
        ga = GA(my_world,exp.CXPB, exp.MUTPB, exp.NGEN, exp.NPOP,exp.type_prob)

        #number of the next block of generations
        gen = Generation.objects.latest('id')

        # gen_guess = Generation.objects.get(id=gen.id-1)

        #########################
        pfront = self.GetFront()

        #fake population
        pop_front=ga.SetPopulationFakeBench(my_world, problem=exp.type_prob)
        del pop_front[len(pfront):]

        #replace by front
        count=0
        for ind in pfront:

           chromo = literal_eval(ind)
           for i in xrange(len(chromo)):
               pop_front[count][i] = chromo[i]
           count +=1

        # set FITNESS to the entire front
        fitnesses = ga.GetFitness(pop_front)
        ga.AttachFitness(pop_front,fitnesses)

        #create list of fitness
        front_fit=[]
        for i in pop_front:
            front_fit.append([i.fitness.values[0],i.fitness.values[1]])


        ##########################

        #fake population
        pop=ga.SetPopulationFakeBench(my_world, problem=exp.type_prob)

        #get the LAST Population
        count=0
        for ind in Population.objects.all().filter(generation=gen.id):

           chromo_original = literal_eval(ind.chromosome_original)
           for i in xrange(len(chromo)):
               pop[count][i]= chromo_original[i]
           count +=1
        #ok, the real population!!
        fitnesses = ga.GetFitness(pop)
        ga.AttachFitness(pop,fitnesses)

        pop_fit=[]
        for i in pop:
            pop_fit.append([i.fitness.values[0],i.fitness.values[1]])


        #################################
        means = literal_eval(gen.means)

        if exp.type_prob == 'A':
            t = np.array(front_fit, np.int32)
            u = np.array(pop_fit, np.int32)
        else:
            t = np.array(front_fit, np.float)
            u = np.array(pop_fit, np.float)

        fig=Figure()
        #print cantour
        plt=fig.add_subplot(111)

        plt.scatter(u[:,0], u[:,1], c='lightgray', alpha=1, label='Pareto front')
        plt.scatter(t[:,0], t[:,1], c='k', alpha=1, label='relevant individuals')



        # plt.plot(means[0][0]+ 0.01, means[0][1]+0.08,  'gD', c='k', markersize=8)
        # plt.plot(means[1][0]+ 0.02, means[1][1]+0.04,  'gD', c='k', markersize=8)
        # plt.plot(means[2][0]+0.05, means[2][1]-0.01,  'gD', c='k', markersize=8)


        plt.legend()


        # plt.annotate('mean', xy=(means[0][0], means[0][1]+0.3),  xycoords='data',
        #             xytext=(-60, 40), textcoords='offset points',
        #             arrowprops=dict(arrowstyle="->",
        #                             connectionstyle="arc,angleA=0,armA=30,rad=10"),
        #             )

        fig.autofmt_xdate()
        canvas=FigureCanvas(fig)
        response=django.http.HttpResponse(content_type='image/png')
        canvas.print_png(response)
        pp = PdfPages('polls/static/fronts/front_b_' + str(exp.id) + '__' + str(gen.id) + '.pdf')
        pp.savefig(fig)
        pp.close()

        return response

    def seefinalfront_gauss_line_b(self):


        import numpy as np

        #get last ID of the Experiment
        exp = Experiment.objects.latest('id')

        #get World config
        wrl = exp.world
        gameworld = GameWorld.objects.get(id=wrl.id)
        my_world = World(gameworld.m,gameworld.n)
        my_world.experimentTYPE = exp.type

        #declare Genetic Algorithm for the problem
        ga = GA(my_world,exp.CXPB, exp.MUTPB, exp.NGEN, exp.NPOP,exp.type_prob)

        #number of the next block of generations
        gen = Generation.objects.latest('id')

        #########################
        pfront = self.GetFront()

        #fake population
        pop_front=ga.SetPopulationFakeBench(my_world, problem=exp.type_prob)
        del pop_front[len(pfront):]

        #replace by front
        count=0
        for ind in pfront:

           chromo = literal_eval(ind)
           for i in xrange(len(chromo)):
               pop_front[count][i] = chromo[i]
           count +=1

        # set FITNESS to the entire front
        fitnesses = ga.GetFitness(pop_front)
        ga.AttachFitness(pop_front,fitnesses)

        #create list of fitness
        front_fit=[]
        for i in pop_front:
            front_fit.append([i.fitness.values[0],i.fitness.values[1]])


        ##########################

        #fake population
        pop=ga.SetPopulationFakeBench(my_world, problem=exp.type_prob)

        #get the LAST Population
        count=0
        for ind in Population.objects.all().filter(generation=gen.id):

           chromo_original = literal_eval(ind.chromosome_original)
           for i in xrange(len(chromo)):
               pop[count][i]= chromo_original[i]
           count +=1
        #ok, the real population!!
        fitnesses = ga.GetFitness(pop)
        ga.AttachFitness(pop,fitnesses)

        pop_fit=[]
        for i in pop:
            pop_fit.append([i.fitness.values[0],i.fitness.values[1]])
        pareto=tools.ParetoFront()
        pareto.update(pop)
        # pareto_fit=[]
        # for i in pop:
        #     pareto_fit.append([i.fitness.values[0],i.fitness.values[1]])
        paretoX_gen1 =[]
        paretoY_gen1=[]
        for j in pareto:
            paretoX_gen1.append(j.fitness.values[0])
            paretoY_gen1.append(j.fitness.values[1])


        #################################
        means = literal_eval(gen.means)

        if exp.type_prob == 'A':
            t = np.array(front_fit, np.int32)
            u = np.array(pop_fit, np.int32)
        else:
            t = np.array(front_fit, np.float)
            u = np.array(pop_fit, np.float)

        fig=Figure()
        #print cantour
        plt=fig.add_subplot(111)

        plt.plot(paretoX_gen1, paretoY_gen1, c='darkgrey', alpha=1, label='Pareto front')
        plt.scatter(t[:,0], t[:,1], c='k', alpha=1, label='relevant individuals')




        # plt.plot(means[0][0]+ 0.01, means[0][1]+0.08,  'gD', c='k', markersize=8)
        # plt.plot(means[1][0]+ 0.02, means[1][1]+0.04,  'gD', c='k', markersize=8)
        # plt.plot(means[2][0]+0.05, means[2][1]-0.01,  'gD', c='k', markersize=8)


        plt.legend()


        # plt.annotate('mean', xy=(means[0][0], means[0][1]+0.3),  xycoords='data',
        #             xytext=(-60, 40), textcoords='offset points',
        #             arrowprops=dict(arrowstyle="->",
        #                             connectionstyle="arc,angleA=0,armA=30,rad=10"),
        #             )

        fig.autofmt_xdate()
        canvas=FigureCanvas(fig)
        response=django.http.HttpResponse(content_type='image/png')
        canvas.print_png(response)
        pp = PdfPages('polls/static/fronts/front_b_line_' + str(exp.id) + '__' + str(gen.id) + '.pdf')
        pp.savefig(fig)
        pp.close()

        return response


    def seefinalfront_gauss_line(self):


        import numpy as np

        #get last ID of the Experiment
        exp = Experiment.objects.latest('id')

        #get World config
        wrl = exp.world
        gameworld = GameWorld.objects.get(id=wrl.id)
        my_world = World(gameworld.m,gameworld.n)
        my_world.MaxArea(gameworld.max_areas)
        my_world.MaxUnits(gameworld.max_units)
        my_world.Production(gameworld.prod_unit0,gameworld.prod_unit1)
        my_world.Costs(gameworld.cost_gateway,gameworld.cost_unit0,gameworld.cost_unit1)
        my_world.experimentTYPE = exp.type

        for area in Area.objects.all().filter(world=gameworld.id):
            my_world.CreateArea(area.x,area.y,area.length)


        #declare Genetic Algorithm for the problem
        ga = GA(my_world,exp.CXPB, exp.MUTPB, exp.NGEN, exp.NPOP,exp.type_prob)

        #number of the next block of generations
        gen = Generation.objects.latest('id')

        #########################
        pfront = self.GetFront(problem=exp.type_prob)

        #replace by front
        count=0
        if exp.type_prob=='A':
            #fake population
            pop_front=ga.SetPopulationFake(my_world)
            del pop_front[len(pfront):]

            for ind in pfront:
               pop_front[count][0]  = ind
               count +=1

        else:
            pop_front=ga.SetPopulationFakeBench(my_world, problem=exp.type_prob)
            del pop_front[len(pfront):]
            for ind in pfront:

               chromo = literal_eval(ind)
               for i in xrange(len(chromo)):
                   pop_front[count][i] = chromo[i]
               count +=1

        # set FITNESS to the entire front
        fitnesses = ga.GetFitness(pop_front)
        ga.AttachFitness(pop_front,fitnesses)

        #create list of fitness
        front_fit=[]
        for i in pop_front:
            front_fit.append([i.fitness.values[0],i.fitness.values[1]])


        ##########################

        if exp.type_prob=='A':
            #fake population
            pop=ga.SetPopulationFake(my_world)

            #get the LAST Population
            count=0
            for ind in Population.objects.all().filter(generation=gen.id):
               pop[count][0] = literal_eval(ind.chromosome_original)
               count +=1
        else:

            #fake population
            pop=ga.SetPopulationFakeBench(my_world, problem=exp.type_prob)

            #get the LAST Population
            count=0
            for ind in Population.objects.all().filter(generation=gen.id):

               chromo_original = literal_eval(ind.chromosome_original)
               for i in xrange(len(chromo)):
                   pop[count][i]= chromo_original[i]
               count +=1


        #ok, the real population!!
        fitnesses = ga.GetFitness(pop)
        ga.AttachFitness(pop,fitnesses)




        pop_fit=[]
        for i in pop:
            pop_fit.append([i.fitness.values[0],i.fitness.values[1]])
        pareto=tools.ParetoFront()
        pareto.update(pop)
        # pareto_fit=[]
        # for i in pop:
        #     pareto_fit.append([i.fitness.values[0],i.fitness.values[1]])
        paretoX_gen1 =[]
        paretoY_gen1=[]
        for j in pareto:
            paretoX_gen1.append(j.fitness.values[0])
            paretoY_gen1.append(j.fitness.values[1])


        #################################
        means = literal_eval(gen.means)

        if exp.type_prob == 'A':
            t = np.array(front_fit, np.int32)
            u = np.array(pop_fit, np.int32)
        else:
            t = np.array(front_fit, np.float)
            u = np.array(pop_fit, np.float)

        fig=Figure()
        #print cantour
        plt=fig.add_subplot(111)

        plt.plot(paretoX_gen1, paretoY_gen1, c='darkgrey', alpha=1, label='Pareto front')
        #plt.scatter(paretoX_gen1, paretoY_gen1, c='darkgrey', alpha=1, label='Pareto front')
        plt.scatter(t[:,0], t[:,1], c='cyan', alpha=1, label='relevant individuals')




        # plt.plot(means[0][0]+ 0.01, means[0][1]+0.08,  'gD', c='k', markersize=8)
        # plt.plot(means[1][0]+ 0.02, means[1][1]+0.04,  'gD', c='k', markersize=8)
        # plt.plot(means[2][0]+0.05, means[2][1]-0.01,  'gD', c='k', markersize=8)


        plt.legend()


        # plt.annotate('mean', xy=(means[0][0], means[0][1]+0.3),  xycoords='data',
        #             xytext=(-60, 40), textcoords='offset points',
        #             arrowprops=dict(arrowstyle="->",
        #                             connectionstyle="arc,angleA=0,armA=30,rad=10"),
        #             )

        fig.autofmt_xdate()
        canvas=FigureCanvas(fig)
        response=django.http.HttpResponse(content_type='image/png')
        canvas.print_png(response)
        pp = PdfPages('polls/static/fronts/front_color_line_' + str(exp.id) + '__' + str(gen.id) + '.pdf')
        pp.savefig(fig)
        pp.close()

        return response

