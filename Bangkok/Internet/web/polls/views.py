# Create your views here.
from django.http import Http404
from service.models import Experiment, Generation, GameWorld, GameWorld, Area
from django.shortcuts import get_object_or_404, render
from django.http import HttpResponseRedirect, HttpResponse
from django.core.urlresolvers import reverse
from interface import InterfaceGA
from service.rungame import Game
from robot import Robot
from world import World
from emoa import GA
from ast import literal_eval
import pickle
import json
from deap.benchmarks.tools import diversity, convergence
import numpy as np
from scipy.spatial import distance
from scipy.spatial import ConvexHull




#
# from django.template import RequestContext, loader
# import numpy as np
# import matplotlib.pyplot as plt
# import matplotlib.mlab as mlab
# from matplotlib import pylab
# from pylab import *
# import PIL
# import PIL.Image
# import StringIO


def start(request):

    #get world map
    if GameWorld.objects.all().count() == 0:
        #create basic world
        wrl = GameWorld(name= 'Mundo 20x20')
        wrl.save()

        area = Area(world=wrl, x=3, y=3, length=7)
        area.save()
        area = Area(world=wrl, x=18, y=15, length=2)
        area.save()
        area = Area(world=wrl, x=12, y=8, length=5)
        area.save()
        area = Area(world=wrl, x=13, y=1, length=3)
        area.save()
        area = Area(world=wrl, x=2, y=15, length=4)
        area.save()
        area = Area(world=wrl, x=7, y=12, length=3)
        area.save()

        #wrl = None

    wrl = GameWorld.objects.all()


    #if first time
    if Experiment.objects.all().count() == 0:
        latest_exp = -1
        context = {'latest_exp': latest_exp, 'world': wrl}
        return render(request, 'polls/new.html', context)
        #return HttpResponseRedirect(reverse('polls:new'))
        #return render(request, 'polls/new.html',)

    #if NEW evolution, but with a previous experiment in the database
    elif Experiment.objects.latest('id').flag == 'F':
        latest_exp = -2
        latest_type = Experiment.objects.latest('id').type

        context = {'latest_exp': latest_exp, 'world': wrl, 'latest_type': latest_type}
        return render(request, 'polls/new.html', context)
        #return HttpResponseRedirect(reverse('polls:new'))
        # return render(request, 'polls/new.html')

    #if Continue Evolution
    elif Experiment.objects.latest('id').type_prob == 'A':
        #get last experiment
        latest_exp = Experiment.objects.latest('id')
        context = {'latest_exp': latest_exp}
        return render(request, 'polls/start.html', context)
        # return render(request, 'polls/start.html')

    #error different experiment is running.
    else:
            return render(request, 'polls/error.html', {
                   'exp': 1,
                   'error_message':  "Hey! You still have another kind of experiment running (DTLZ or ZDT). Please, finish it first.",
               })



def startBench(request):

    #get world map
    if GameWorld.objects.all().count() == 0:
        #create basic world
        wrl = GameWorld(name= 'Mundo 20x20')
        wrl.save()

        area = Area(world=wrl, x=3, y=3, length=7)
        area.save()
        area = Area(world=wrl, x=18, y=15, length=2)
        area.save()
        area = Area(world=wrl, x=12, y=8, length=5)
        area.save()
        area = Area(world=wrl, x=13, y=1, length=3)
        area.save()
        area = Area(world=wrl, x=2, y=15, length=4)
        area.save()
        area = Area(world=wrl, x=7, y=12, length=3)
        area.save()


    #if first time
    if Experiment.objects.all().count() == 0:
        latest_exp = -1
        context = {'latest_exp': latest_exp,}
        return render(request, 'polls/newBench.html', context)
        #return HttpResponseRedirect(reverse('polls:new'))
        #return render(request, 'polls/new.html',)

    #if NEW evolution, but with a previous experiment in the database
    elif Experiment.objects.latest('id').flag == 'F':
        latest_exp = -2


        latest_type = Experiment.objects.latest('id').type

        context = {'latest_exp': latest_exp, 'latest_type': latest_type}
        return render(request, 'polls/newBench.html', context)
        #return HttpResponseRedirect(reverse('polls:new'))
        # return render(request, 'polls/new.html')

    #if Continue Evolution
    elif Experiment.objects.latest('id').type_prob != 'A':
        #get last experiment
        latest_exp = Experiment.objects.latest('id')
        context = {'latest_exp': latest_exp}
        return render(request, 'polls/startBench.html', context)
        # return render(request, 'polls/start.html')

    #error different experiment is running.
    else:
            return render(request, 'polls/error.html', {
                   'exp': 1,
                   'error_message':  "Hey! You still have another kind of experiment running (Resource Distribution). Please, finish it first.",
               })


def articles(request):
    return render(request, 'polls/articles.html', )


def home(request):

    return render(request, 'polls/home.html',)
    # return render(request, 'polls/start.html')


def new(request):
    #create new experiment!

    #read all fields
    try:
        selected_btn = request.POST['gorun']
        name = request.POST['name']
        date = request.POST['date']
        type = request.POST['type']
        description = request.POST['description']
        # flag = request.POST['status']
        flag = 'W'
        num_population = request.POST['pop']
        num_robots = request.POST['robots']
        num_levels = request.POST['numlevel']
        num_gen = request.POST['generations']
        block_size = request.POST['block']
        gen_threshold = request.POST['gen_threshold']
        # actual_gen = request.POST['actual_gen']
        actual_gen = 0
        # player = request.POST['player']
        player = 7
        mutation = request.POST['mutation']
        cross = request.POST['cross']
        wrld = request.POST['wrld']
        first_loop = request.POST['floop']
        ropoints = request.POST['ropoints']
        txtfreeK = request.POST['freeK']
        moea_alg = request.POST['moea_alg']
        tour = request.POST['tour']
        vote = request.POST['vote']
        type_prob = 'A'
        keep_interactivity = request.POST['keep_i']

        if txtfreeK == "0":
            freeK = False
        else:
            freeK = True



    except (KeyError, Experiment.DoesNotExist):
        # Redisplay the question voting form.
        return render(request, 'polls/error.html', {
            'exp': "error",
            'error_message': "Error on the form!.",        })
    else:

        #create and start
        inter = InterfaceGA()

        #if Start or Continue button
        if selected_btn == "run":

            #run through the end using only robots vote
            if keep_interactivity == "0":

                #loop de testes
                for i in xrange(1):

                    #start generations, does not matter if it is the first start or continuing evolution
                    exp = inter.StartEvolution(name,date,type,description, flag, int(num_population), int(num_robots),
                                               int(num_levels), int(num_gen), int(block_size), int(gen_threshold),
                                               int(actual_gen), player, float(mutation), float(cross), int(wrld),
                                               int(first_loop), ropoints, freeK, moea_alg, tour, vote, type_prob)

                    for i in xrange(int(num_levels)):
                        print "block: " + str(i+1)
                        inter.ContinueEvolution()

            else:

                #start generations, does not matter if it is the first start or continuing evolution
                exp = inter.StartEvolution(name,date,type,description, flag, int(num_population), int(num_robots),
                                           int(num_levels), int(num_gen), int(block_size), int(gen_threshold),
                                           int(actual_gen), player, float(mutation), float(cross), int(wrld),
                                           int(first_loop), ropoints, freeK, moea_alg, tour, vote, type_prob)


            #game = Game.get_instance()
            #game.start_experiment(exp, inter)

            #get last experiment
            # latest_exp = Experiment.objects.latest('id')
            # context = {'latest_exp': latest_exp.id}
            # return render(request, 'polls/start.html', context)
            #
            # return HttpResponseRedirect(reverse('polls:results', args=(p.id,)))
            return HttpResponseRedirect(reverse('polls:start'))

        #user chose one of the gets or sets
        elif selected_btn == "scenario":

                #get scenario
                return render(request, 'polls/getresults.html', {
                    'title': "Get Area",
                    'msg': str(inter.GetArea()),
                })

        #user chose one of the gets or sets
        elif selected_btn == "status":

                #check if exp is null

                #get status
                return render(request, 'polls/getresults.html', {
                    'title': "Get Status",
                    'msg': inter.GetStatus(),
                })

        #user chose one of the gets or sets
        elif selected_btn == "pfront":


                #check if exp is null

                #get pfront
                return render(request, 'polls/getresults.html', {
                    'title': "Get Pareto Front",
                    'msg': str(inter.GetFront()),
                })

        #user chose one of the gets or sets
        elif selected_btn == "population":

                #check if exp is null

                #get population
                return render(request, 'polls/getresults.html', {
                    'title': "Get Population",
                    'msg': str(inter.GetPopulation()),
                })

        #user chose one of the gets or sets
        elif selected_btn == "set":

                #get robots comparisons
                #return HttpResponseRedirect(reverse('polls:gauss'))
                return render(request, 'polls/gaussianmix.html', {
                    'title': "Univariate Gaussian Mixture",
                })

                #set results
                # return render(request, 'polls/setresults.html', {
                #     'title': "Set Comparisons",
                #     'exp_id': exp,
                # })

        #user chose one of the gets or sets
        elif selected_btn == "bivariate":

                #get robots comparisons
                #return HttpResponseRedirect(reverse('polls:gauss'))
                return render(request, 'polls/bigaussianmix.html', {
                    'title': "Multivariate Gaussian Mixture",
                })

                #set results
                # return render(request, 'polls/setresults.html', {
                #     'title': "Set Comparisons",
                #     'exp_id': exp,
                # })
        #user chose one of the gets or sets
        elif selected_btn == "kmeans":

                #get robots comparisons
                #return HttpResponseRedirect(reverse('polls:gauss'))
                return render(request, 'polls/kmeans.html', {
                    'title': "K-means",
                })

                #set results
                # return render(request, 'polls/setresults.html', {
                #     'title': "Set Comparisons",
                #     'exp_id': exp,
                # })

        #see evolution
        elif selected_btn == "evol":

                #check num block
                exp_last = Experiment.objects.latest('id')
                gen = Generation.objects.latest('id')

                #show evolution
                return render(request, 'polls/showevolution.html', {
                    'title': "Evolution",
                    'msg': "videos/bang__" + str(exp_last.id) + "__" + str(gen.block) + ".mp4",


                })

        #show last front
        elif selected_btn == "final_gauss":

                #get robots comparisons
                #return HttpResponseRedirect(reverse('polls:gauss'))
                return render(request, 'polls/finalfront_gauss.html', {
                    'title': "Gaussian Mixture",
                })

        #show last front
        elif selected_btn == "final_kmeans":

                #get robots comparisons
                #return HttpResponseRedirect(reverse('polls:gauss'))
                return render(request, 'polls/finalfront_kmeans.html', {
                    'title': "K-means",
                })

        #user chose one of the gets or sets
        elif selected_btn == "fitfront":


                #check if exp is null

                #get pfront
                return render(request, 'polls/getresults.html', {
                    'title': "Get Fitness Pareto Front",
                    'msg': str(inter.GetFitnessFront()),
                })
        else:
            return render(request, 'polls/error.html', {
                   'exp': 1,
                   'error_message':  "You did not select an action.",
               })


    #return render(request, 'polls/error.html', {
    #'exp': "ssss",
    #'error_message': num_levels,        })


def PolyArea2D(pts):
    lines = np.hstack([pts,np.roll(pts,-1,axis=0)])
    area = 0.5*abs(sum(x1*y2-x2*y1 for x1,y1,x2,y2 in lines))
    return area



def newBench(request):
    #create new experiment!

    #read all fields
    try:
        selected_btn = request.POST['gorun']
        name = request.POST['name']
        date = request.POST['date']
        type = request.POST['type']
        description = request.POST['description']
        # flag = request.POST['status']
        flag = 'W'
        num_population = request.POST['pop']
        num_robots = request.POST['robots']
        num_levels = request.POST['numlevel']
        num_gen = request.POST['generations']
        block_size = request.POST['block']
        gen_threshold = request.POST['gen_threshold']
        # actual_gen = request.POST['actual_gen']
        actual_gen = 0
        # player = request.POST['player']
        player = 7
        mutation = request.POST['mutation']
        cross = request.POST['cross']
        first_loop = request.POST['floop']
        ropoints = request.POST['ropoints']
        txtfreeK = request.POST['freeK']
        moea_alg = request.POST['moea_alg']
        tour = request.POST['tour']
        vote = request.POST['vote']
        type_prob = request.POST['type_prob']
        keep_interactivity = request.POST['keep_i']

        if txtfreeK == "0":
            freeK = False
        else:
            freeK = True



    except (KeyError, Experiment.DoesNotExist):
        # Redisplay the question voting form.
        return render(request, 'polls/error.html', {
            'exp': "error",
            'error_message': "Error on the form!.",        })
    else:

        #create and start
        inter = InterfaceGA()

        #if Start or Continue button
        if selected_btn == "run":


            #run through the end using only robots vote
            if keep_interactivity == "0":


                volume=[]
                conv=[]
                divers=[]
                disper=[]
                convexarea=[]
                convexpts=""
                nonconvexpts=""
                #amp=[]
                #loop de testes
                #YUKA
                for i in xrange(20):

                    #start generations, does not matter if it is the first start or continuing evolution
                    exp = inter.StartEvolutionBench(name,date,type,description, flag, int(num_population), int(num_robots),
                                               int(num_levels), int(num_gen), int(block_size), int(gen_threshold),
                                               int(actual_gen), player, float(mutation), float(cross),
                                               int(first_loop), ropoints, freeK, moea_alg, tour, vote, type_prob)


                    for j in xrange(int(num_levels)):
                        print "block: " + str(j+1)
                        inter.ContinueEvolutionBench()


                    #########################
                    gameworld = GameWorld.objects.get(id=1)
                    my_world = World(gameworld.m,gameworld.n)
                    my_world.experimentTYPE = type

                    #declare Genetic Algorithm for the problem
                    ga = GA(my_world,float(cross), float(mutation), int(num_gen), int(num_population),type_prob)

                    pfront = inter.GetFront()

                    #fake population
                    pop_front=ga.SetPopulationFakeBench(my_world, problem=type_prob)
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

                    #Dispersion
                    p = np.array(fitnesses, np.float)
                    dispersion = np.var(p[:])
                    disper.append(float(dispersion))


                    ##################
                    # CONVEX HULL 2D #
                    ##################
                    #print points
                    # strx = ""
                    # strx += "**"
                    # strx += str(len(fitnesses))
                    # strx += "**"
                    # strx += "["
                    # for k in fitnesses:
                    #     strx += "[" +  str(k[0]) + "," + str(k[1]) + "],"
                    #     #strx += "," + str(k[0]) + "," + str(k[1]) + "," + str(k[2])
                    #     #stry += "[" + str(k[0]) + "," + str(k[1]) + "],"
                    # strx += "]"
                    # print strx
                    # #print " "
                    # #print stry
                    # #print len(fitnesses)
                    #
                    #calculate volume
                    hull = ConvexHull(fitnesses)
                    pts=[]
                    for vert in hull.vertices:
                        pts.append(fitnesses[vert])

                    area =  PolyArea2D(pts)
                    print area
                    convexarea.append(float(area))
                    # convexpts += strx
                    #
                    #
                    #SAVE area COnvex Hull
                    with open('ZDT1 NSGA2 Convex Volume KM.txt', 'wb') as f:
                         pickle.dump(convexarea, f)
                    # with open('ZDTX NSGA2 Convex POINTS GM .txt', 'wb') as f:
                    #      pickle.dump(convexpts, f)

                    ##################
                    # CONVEX HULL 3D #
                    ##################

                    #####################
                    # NONCONVEX HULL 2D #
                    #####################

                    #####################
                    # NONCONVEX HULL 3D #
                    #####################
                    # #print points
                    # strx = ""
                    # strx += "**"
                    # strx += str(len(fitnesses))
                    # strx += "**"
                    # #strx += "["
                    # for k in fitnesses:
                    #     #strx += "[" +  str(k[0]) + "," + str(k[1]) + "],"
                    #     strx += "," + str(k[0]) + "," + str(k[1]) + "," + str(k[2])
                    #     #stry += "[" + str(k[0]) + "," + str(k[1]) + "],"
                    # #strx += "]"
                    # print strx
                    # #print " "
                    # #print stry
                    # #print len(fitnesses)
                    # nonconvexpts += strx
                    #
                    #
                    # #SAVE area NON-Convex Hull
                    # with open('ZDTX NSGA2 NONConvex POINTS GM .txt', 'wb') as f:
                    #      pickle.dump(nonconvexpts, f)

                    #Amplitude
                    #amplitude = distance.euclidean(p[0], p[-1])
                    #amp.append(float(amplitude))

                    # if type_prob == 'P' or type_prob == 'Q' or type_prob == 'R' or type_prob == 'S' or type_prob == 'T' or type_prob == 'F':
                        #volume.append(ga.Hypervolume2D(pop_front, (11.0,11.0)))
                        # var_hyper = ga.Hypervolume2D(pop_front, (11.0,11.0))


                    # else:
                        #volume.append(ga.Hypervolume3D(pop_front, (11.0,11.0,11.0)))
                        # var_hyper = ga.Hypervolume3D(pop_front, (11.0,11.0,11.0))

                    #optimal_front = json.load(open("dtlz3_front.json"))
                    #var_conv = convergence(pop_front, optimal_front)
                    #var_divers = diversity(pop_front, optimal_front[0], optimal_front[-1])

                    #volume.append(var_hyper)
                    #conv.append(var_conv)
                    #divers.append(var_divers)
                    print "##########"
                    #print var_hyper
                    #print var_conv
                    #print var_divers
                    print dispersion
                    #print amplitude

                    #SAVE dispersion
                    with open('ZDT1 NSGA2 Dispersion KM.txt', 'wb') as f:
                         pickle.dump(disper, f)

                    #SAVE amplitude
                    #with open('ZDT1 NSGA2 Amplitude GM.txt', 'wb') as f:
                    #     pickle.dump(amp, f)


                    #SAVE hypervolume
                    # with open('DTLZ7 SMS2 Hypervolume KM.txt', 'wb') as f:
                    #     pickle.dump(volume, f)
                    #
                    # SAVE objective space
                    # with open('ZDT1 SPEA2 Objective Space.txt', 'wb') as f:
                    #     pickle.dump(fitnesses, f)

                    #
                    # #SA2VE Convergence
                    # with open('DTLZ3 SMS2 Convergence GM.txt', 'wb') as f:
                    #     pickle.dump(conv, f)
                    #
                    # #SA2VE Diversity
                    # with open('DTLZ3 SMS2 Diversity GM.txt', 'wb') as f:
                    #     pickle.dump(divers, f)

            else:

                #start generations, does not matter if it is the first start or continuing evolution
                exp = inter.StartEvolutionBench(name,date,type,description, flag, int(num_population), int(num_robots),
                                           int(num_levels), int(num_gen), int(block_size), int(gen_threshold),
                                           int(actual_gen), player, float(mutation), float(cross),
                                           int(first_loop), ropoints, freeK, moea_alg, tour, vote, type_prob)



            #game = Game.get_instance()
            #game.start_experiment(exp, inter)

            #get last experiment
            # latest_exp = Experiment.objects.latest('id')
            # context = {'latest_exp': latest_exp.id}
            # return render(request, 'polls/start.html', context)
            #
            # return HttpResponseRedirect(reverse('polls:results', args=(p.id,)))
            return HttpResponseRedirect(reverse('polls:startBench'))

        #user chose one of the gets or sets
        elif selected_btn == "scenario":

                #get scenario
                return render(request, 'polls/getresults.html', {
                    'title': "Get Area",
                    'msg': str(inter.GetArea()),
                })

        #user chose one of the gets or sets
        elif selected_btn == "status":

                #check if exp is null

                #get status
                return render(request, 'polls/getresults.html', {
                    'title': "Get Status",
                    'msg': inter.GetStatus(),
                })

        #user chose one of the gets or sets
        elif selected_btn == "pfront":


                #check if exp is null

                #get pfront
                return render(request, 'polls/getresults.html', {
                    'title': "Get Pareto Front",
                    'msg': str(inter.GetFront()),
                })

        #user chose one of the gets or sets
        elif selected_btn == "population":

                #check if exp is null

                #get population
                return render(request, 'polls/getresults.html', {
                    'title': "Get Population",
                    'msg': str(inter.GetPopulation()),
                })

        #user chose one of the gets or sets
        elif selected_btn == "set":

                #get robots comparisons
                #return HttpResponseRedirect(reverse('polls:gauss'))
                return render(request, 'polls/gaussianmix.html', {
                    'title': "Univariate Gaussian Mixture",
                })

                #set results
                # return render(request, 'polls/setresults.html', {
                #     'title': "Set Comparisons",
                #     'exp_id': exp,
                # })

        #user chose one of the gets or sets
        elif selected_btn == "bivariate":

                #get robots comparisons
                #return HttpResponseRedirect(reverse('polls:gauss'))
                return render(request, 'polls/bigaussianmix.html', {
                    'title': "Multivariate Gaussian Mixture",
                })

                #set results
                # return render(request, 'polls/setresults.html', {
                #     'title': "Set Comparisons",
                #     'exp_id': exp,
                # })
        #user chose one of the gets or sets
        elif selected_btn == "kmeans":

                #get robots comparisons
                #return HttpResponseRedirect(reverse('polls:gauss'))
                return render(request, 'polls/kmeans.html', {
                    'title': "K-means",
                })

                #set results
                # return render(request, 'polls/setresults.html', {
                #     'title': "Set Comparisons",
                #     'exp_id': exp,
                # })

        #see evolution
        elif selected_btn == "evol":

                #check num block
                exp_last = Experiment.objects.latest('id')
                gen = Generation.objects.latest('id')

                #show evolution
                return render(request, 'polls/showevolution.html', {
                    'title': "Evolution",
                    'msg': "videos/bang__" + str(exp_last.id) + "__" + str(gen.block) + ".mp4",


                })


        #show last front
        elif selected_btn == "final_gauss":

                #get robots comparisons
                #return HttpResponseRedirect(reverse('polls:gauss'))
                return render(request, 'polls/finalfront_gauss.html', {
                    'title': "Gaussian Mixture",
                })

        #show last front
        elif selected_btn == "final_kmeans":

                #get robots comparisons
                #return HttpResponseRedirect(reverse('polls:gauss'))
                return render(request, 'polls/finalfront_kmeans.html', {
                    'title': "K-means",
                })
        #user chose one of the gets or sets
        elif selected_btn == "fitfront":


                #check if exp is null

                #get pfront
                return render(request, 'polls/getresults.html', {
                    'title': "Get Fitness Pareto Front",
                    'msg': str(inter.GetFitnessFront()),
                })
        else:
            return render(request, 'polls/error.html', {
                   'exp': 1,
                   'error_message':  "You did not select an action.",
               })


    #return render(request, 'polls/error.html', {
    #'exp': "ssss",
    #'error_message': num_levels,        })

def go(request, exp_id):

    #get selected action (button)
    try:
        selected_btn = request.POST['gorun']
    except (KeyError, Generation.DoesNotExist):
        # Redisplay the question voting form.
        return render(request, 'polls/error.html', {
            'exp': exp_id,
            'error_message': "You did not select an action.",        })
    else:

        #if id

        #get experiment
        exp = get_object_or_404(Experiment, pk=exp_id)

        inter = InterfaceGA()

        #if Start or Continue button
        if selected_btn == "run":
            #check if exp is null or 'F'

            #check if exp is 'R'

            if exp.flag == 'R':

                #start generations, does not matter if it is the first start or continuing evolution
                inter.ContinueEvolution()

                return HttpResponseRedirect(reverse('polls:start'))
            else:
                return render(request, 'polls/error.html', {
                    'exp': exp,
                    'error_message': "You are trying to run a wrong experiment.",
                })
        #user chose one of the gets or sets
        elif selected_btn == "scenario":

                #get scenario
                return render(request, 'polls/getresults.html', {
                    'title': "Get Area",
                    'msg': str(inter.GetArea()),
                })

        #user chose one of the gets or sets
        elif selected_btn == "status":

                #check if exp is null

                #get status
                return render(request, 'polls/getresults.html', {
                    'title': "Get Status",
                    'msg': inter.GetStatus(),
                })

        #user chose one of the gets or sets
        elif selected_btn == "pfront":


                #check if exp is null

                #get pfront
                return render(request, 'polls/getresults.html', {
                    'title': "Get Pareto Front",
                    'msg': str(inter.GetFront()),
                })

        #user chose one of the gets or sets
        elif selected_btn == "population":

                #check if exp is null

                #get population
                return render(request, 'polls/getresults.html', {
                    'title': "Get Population",
                    'msg': str(inter.GetPopulation()),
                })

        #user chose one of the gets or sets
        elif selected_btn == "set":

                #get robots comparisons
                #return HttpResponseRedirect(reverse('polls:gauss'))
                return render(request, 'polls/gaussianmix.html', {
                    'title': "Univariate Gaussian Mixture",
                })

                #set results
                # return render(request, 'polls/setresults.html', {
                #     'title': "Set Comparisons",
                #     'exp_id': exp,
                # })

        #user chose one of the gets or sets
        elif selected_btn == "bivariate":

                #get robots comparisons
                #return HttpResponseRedirect(reverse('polls:gauss'))
                return render(request, 'polls/bigaussianmix.html', {
                    'title': "Multivariate Gaussian Mixture",
                })

                #set results
                # return render(request, 'polls/setresults.html', {
                #     'title': "Set Comparisons",
                #     'exp_id': exp,
                # })
        #user chose one of the gets or sets
        elif selected_btn == "kmeans":

                #get robots comparisons
                #return HttpResponseRedirect(reverse('polls:gauss'))
                return render(request, 'polls/kmeans.html', {
                    'title': "K-means",
                })

                #set results
                # return render(request, 'polls/setresults.html', {
                #     'title': "Set Comparisons",
                #     'exp_id': exp,
                # })

        #user chose one of the gets or sets
        elif selected_btn == "humanvote":

                #get robots comparisons
                #return HttpResponseRedirect(reverse('polls:gauss'))
                return render(request, 'polls/humanvote.html', {
                    'title': "Choose your scenario",
                })

                #set results
                # return render(request, 'polls/setresults.html', {
                #     'title': "Set Comparisons",
                #     'exp_id': exp,
                # })


        #see evolution
        elif selected_btn == "evol":

                #check num block
                gen = Generation.objects.latest('id')

                #show evolution
                return render(request, 'polls/showevolution.html', {
                    'title': "Evolution",
                    'msg': "videos/bang__" + str(exp.id) + "__" + str(gen.block) + ".mp4",


                })


        #user chose one of the gets or sets
        elif selected_btn == "fitfront":


                #check if exp is null

                #get pfront
                return render(request, 'polls/getresults.html', {
                    'title': "Get Fitness Pareto Front",
                    'msg': str(inter.GetFitnessFront()),
                })


        #show last front
        elif selected_btn == "final_gauss":

                #get robots comparisons
                #return HttpResponseRedirect(reverse('polls:gauss'))
                return render(request, 'polls/finalfront_gauss.html', {
                    'title': "Gaussian Mixture",
                })

        #show last front
        elif selected_btn == "final_kmeans":

                #get robots comparisons
                #return HttpResponseRedirect(reverse('polls:gauss'))
                return render(request, 'polls/finalfront_kmeans.html', {
                    'title': "K-means",
                })

        else:
            return render(request, 'polls/error.html', {
                   'exp': exp,
                   'error_message':  "You did not select an action.",
               })



def goBench(request, exp_id):

    #get selected action (button)
    try:
        selected_btn = request.POST['gorun']
    except (KeyError, Generation.DoesNotExist):
        # Redisplay the question voting form.
        return render(request, 'polls/error.html', {
            'exp': exp_id,
            'error_message': "You did not select an action.",        })
    else:

        #if id

        #get experiment
        exp = get_object_or_404(Experiment, pk=exp_id)

        inter = InterfaceGA()

        #if Start or Continue button
        if selected_btn == "run":
            #check if exp is null or 'F'

            #check if exp is 'R'

            if exp.flag == 'R':

                #start generations, does not matter if it is the first start or continuing evolution
                inter.ContinueEvolutionBench()

                return HttpResponseRedirect(reverse('polls:startBench'))
            else:
                return render(request, 'polls/error.html', {
                    'exp': exp,
                    'error_message': "You are trying to run a wrong experiment.",
                })
        #user chose one of the gets or sets
        elif selected_btn == "scenario":

                #get scenario
                return render(request, 'polls/getresults.html', {
                    'title': "Get Area",
                    'msg': str(inter.GetArea()),
                })

        #user chose one of the gets or sets
        elif selected_btn == "status":

                #check if exp is null

                #get status
                return render(request, 'polls/getresults.html', {
                    'title': "Get Status",
                    'msg': inter.GetStatus(),
                })

        #user chose one of the gets or sets
        elif selected_btn == "pfront":


                #check if exp is null

                #get pfront
                return render(request, 'polls/getresults.html', {
                    'title': "Get Pareto Front",
                    'msg': str(inter.GetFront()),
                })

        #user chose one of the gets or sets
        elif selected_btn == "population":

                #check if exp is null

                #get population
                return render(request, 'polls/getresults.html', {
                    'title': "Get Population",
                    'msg': str(inter.GetPopulation()),
                })

        #user chose one of the gets or sets
        elif selected_btn == "set":

                #get robots comparisons
                #return HttpResponseRedirect(reverse('polls:gauss'))
                return render(request, 'polls/gaussianmix.html', {
                    'title': "Univariate Gaussian Mixture",
                })

                #set results
                # return render(request, 'polls/setresults.html', {
                #     'title': "Set Comparisons",
                #     'exp_id': exp,
                # })

        #user chose one of the gets or sets
        elif selected_btn == "bivariate":

                #get robots comparisons
                #return HttpResponseRedirect(reverse('polls:gauss'))
                return render(request, 'polls/bigaussianmix.html', {
                    'title': "Multivariate Gaussian Mixture",
                })

                #set results
                # return render(request, 'polls/setresults.html', {
                #     'title': "Set Comparisons",
                #     'exp_id': exp,
                # })
        #user chose one of the gets or sets
        elif selected_btn == "kmeans":

                #get robots comparisons
                #return HttpResponseRedirect(reverse('polls:gauss'))
                return render(request, 'polls/kmeans.html', {
                    'title': "K-means",
                })

                #set results
                # return render(request, 'polls/setresults.html', {
                #     'title': "Set Comparisons",
                #     'exp_id': exp,
                # })

        #user chose one of the gets or sets
        elif selected_btn == "humanvote":

                #get robots comparisons
                #return HttpResponseRedirect(reverse('polls:gauss'))
                return render(request, 'polls/humanvote.html', {
                    'title': "Choose your scenario",
                })

                #set results
                # return render(request, 'polls/setresults.html', {
                #     'title': "Set Comparisons",
                #     'exp_id': exp,
                # })


        #see evolution
        elif selected_btn == "evol":

                #check num block
                gen = Generation.objects.latest('id')

                #show evolution
                return render(request, 'polls/showevolution.html', {
                    'title': "Evolution",
                    'msg': "videos/bang__" + str(exp.id) + "__" + str(gen.block) + ".mp4",


                })


        #user chose one of the gets or sets
        elif selected_btn == "fitfront":


                #check if exp is null

                #get pfront
                return render(request, 'polls/getresults.html', {
                    'title': "Get Fitness Pareto Front",
                    'msg': str(inter.GetFitnessFront()),
                })


        #show last front
        elif selected_btn == "final_gauss":

                #get robots comparisons
                #return HttpResponseRedirect(reverse('polls:gauss'))
                return render(request, 'polls/finalfront_gauss.html', {
                    'title': "Gaussian Mixture",
                })

        #show last front
        elif selected_btn == "final_kmeans":

                #get robots comparisons
                #return HttpResponseRedirect(reverse('polls:gauss'))
                return render(request, 'polls/finalfront_kmeans.html', {
                    'title': "K-means",
                })

        else:
            return render(request, 'polls/error.html', {
                   'exp': exp,
                   'error_message':  "You did not select an action.",
               })



def error(request, exp_id):
    #response = "You're looking at the results of exp_id %s."
    exp = get_object_or_404(Experiment, pk=exp_id)
    return render(request, 'polls/error.html', {'exp': exp})


def getresults(request, exp_id):
    #response = "You're looking at the results of exp_id %s."

    exp = get_object_or_404(Experiment, pk=exp_id)
    return render(request, 'polls/getresults.html', {'exp': exp})

def setresults(request):
    #response = "You're looking at the results of exp_id %s."

    inter = InterfaceGA()
    inter.SetComparisonsResult(request.POST['output'])

    return HttpResponseRedirect(reverse('polls:start'))

#
# def index(request):
#     latest_exp_list = Experiment.objects.order_by('-date')[:5]
#     #output = ', '.join([p.T001_Name for p in latest_exp_list])
#     #template = loader.get_template('polls/index.html')
#     #context = RequestContext(request, {
#     #    'latest_exp_list': latest_exp_list,
#     #})
#     context = {'latest_exp_list': latest_exp_list}
#     return render(request, 'polls/index.html', context)
#
# def detail(request, exp_id):
#     #try:
#     #    exp = T001_Experiment.objects.get(pk=exp_id)
#     #except T001_Experiment.DoesNotExist:
#     #    raise Http404
#     exp = get_object_or_404(Experiment, pk=exp_id)
#     return render(request, 'polls/detail.html', {'exp': exp})
#

def results(request, exp_id):
    #response = "You're looking at the results of exp_id %s."
    exp = get_object_or_404(Experiment, pk=exp_id)
    return render(request, 'polls/results.html', {'exp': exp})

def vote(request, exp_id):
    #return HttpResponse("You're voting on question %s." % exp_id)
    p = get_object_or_404(Experiment, pk=exp_id)
    try:
        selected_block = p.generation_set.get(pk=request.POST['gen'])
    except (KeyError, Generation.DoesNotExist):
        # Redisplay the question voting form.
        return render(request, 'polls/detail.html', {
            'question': p,
            'error_message': "You didn't select a choice.",
        })
    else:
        selected_block.block = 100
        selected_block.save()
        # Always return an HttpResponseRedirect after successfully dealing
        # with POST data. This prevents data from being posted twice if a
        # user hits the Back button.
        return HttpResponseRedirect(reverse('polls:results', args=(p.id,)))

def gauss(request):

    inter = InterfaceGA()
    kk = inter.seeRobotGaussians()
    return kk

def bigausscontour(request):

    inter = InterfaceGA()
    kk = inter.seeBiGaussiansContour()
    return kk

def bigaussbell(request):

    inter = InterfaceGA()
    kk = inter.seeBiGaussiansBell()
    return kk

def bigausscontour_b(request):

    inter = InterfaceGA()
    kk = inter.seeBiGaussiansContour_b()
    return kk

def bigaussbell_b(request):

    inter = InterfaceGA()
    kk = inter.seeBiGaussiansBell_b()
    return kk

def kmeans(request):

    inter = InterfaceGA()
    kk = inter.kmeans()
    return kk

def kmeans_b(request):

    inter = InterfaceGA()
    kk = inter.kmeans_b()
    return kk


def humanvote(request):

    #inter = InterfaceGA()
    #kk = inter.kmeans_b()
    print "quiiiii"
    return "wwwwwwwwwwwwwww",9


def finalfront_gauss(request):

    inter = InterfaceGA()
    kk = inter.seefinalfront_gauss()
    return kk


def finalfront_gauss_b(request):


    inter = InterfaceGA()
    kk = inter.seefinalfront_gauss_b()
    return kk




def finalfront_kmeans(request):

    inter = InterfaceGA()
    kk = inter.seefinalfront_kmeans()
    return kk


def finalfront_kmeans_b(request):

    inter = InterfaceGA()
    kk = inter.seefinalfront_kmeans_b()
    return kk

def finalfront_gauss_line(request):

    inter = InterfaceGA()
    kk = inter.seefinalfront_gauss_line()
    return kk



def finalfront_gauss_line_b(request):

    inter = InterfaceGA()
    kk = inter.seefinalfront_gauss_line_b()
    return kk


def finalfront_kmeans_line(request):

    inter = InterfaceGA()
    kk = inter.seefinalfront_kmeans_line()
    return kk

def finalfront_kmeans_line_b(request):

    inter = InterfaceGA()
    kk = inter.seefinalfront_kmeans_line_b()
    return kk
