# Create your views here.
from django.http import Http404
from service.models import Experiment, Generation, GameWorld
from django.shortcuts import get_object_or_404, render
from django.http import HttpResponseRedirect, HttpResponse
from django.core.urlresolvers import reverse
from interface import InterfaceGA
from service.rungame import Game
from robot import Robot

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
        wrl = None
    else:
        wrl = GameWorld.objects.all()


    #if first time
    if Experiment.objects.all().count() == 0:
        latest_exp = -1
        context = {'latest_exp': latest_exp, 'world': wrl}
        return render(request, 'polls/new.html', context)
        #return HttpResponseRedirect(reverse('polls:new'))

    #if NEW evolution, but with a previous experiment in the database
    elif Experiment.objects.latest('id').flag == 'F':
        latest_exp = -2
        context = {'latest_exp': latest_exp, 'world': wrl}
        return render(request, 'polls/new.html', context)
        #return HttpResponseRedirect(reverse('polls:new'))

    #if Continue Evolution
    else:
        #get last experiment
        latest_exp = Experiment.objects.latest('id')
        context = {'latest_exp': latest_exp}
        return render(request, 'polls/start.html', context)

        # return render(request, 'polls/error.html', {
        # 'exp': "ssss",
        # 'error_message': num_levels,        })
        #

def new(request):
    #create new experiment!

    #read all fields
    try:
        selected_btn = request.POST['gorun']
        name = request.POST['name']
        date = request.POST['date']
        type = request.POST['type']
        description = request.POST['description']
        flag = request.POST['status']
        num_population = request.POST['pop']
        num_robots = request.POST['robots']
        num_levels = request.POST['numlevel']
        num_gen = request.POST['generations']
        block_size = request.POST['block']
        gen_threshold = request.POST['gen_threshold']
        actual_gen = request.POST['actual_gen']
        player = request.POST['player']
        mutation = request.POST['mutation']
        cross = request.POST['cross']
        wrld = request.POST['wrld']
        first_loop = request.POST['floop']


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


            #start generations, does not matter if it is the first start or continuing evolution
            exp = inter.StartEvolution(name,date,type,description, flag, int(num_population), int(num_robots), int(num_levels), float(num_gen),
                                 int(block_size), int(gen_threshold), int(actual_gen), player, float(mutation), float(cross), int(wrld), int(first_loop))

            game = Game.get_instance()
            game.start_experiment(exp, inter)

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
                    'title': "Gaussian Mixture",
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
                    'title': "Gaussian Mixture",
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


def index(request):
    latest_exp_list = Experiment.objects.order_by('-date')[:5]
    #output = ', '.join([p.T001_Name for p in latest_exp_list])
    #template = loader.get_template('polls/index.html')
    #context = RequestContext(request, {
    #    'latest_exp_list': latest_exp_list,
    #})
    context = {'latest_exp_list': latest_exp_list}
    return render(request, 'polls/index.html', context)

def detail(request, exp_id):
    #try:
    #    exp = T001_Experiment.objects.get(pk=exp_id)
    #except T001_Experiment.DoesNotExist:
    #    raise Http404
    exp = get_object_or_404(Experiment, pk=exp_id)
    return render(request, 'polls/detail.html', {'exp': exp})


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