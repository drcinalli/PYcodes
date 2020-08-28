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
import matplotlib.animation as animation




def update_plot(i, data, pata, scat):

    #scat.set_array(data[i])
    plt.clf()

    plt.xlabel('$Cost')
    plt.ylabel('#Production')
    plt.title('do the evolution')
    plt.plot(3, 6,  'g^', markersize=8)

    plt.xticks([0.15, 0.68, 0.97])
    plt.yticks([0.2, 0.55, 0.76])

    #k = axis.xaxis.get_majorticklocs()

    #extraticks = [1, 1.2, 2]
    #plt.xticks(list(plt.xticks()[0]) + extraticks)
    plt.axvline(x=1,  ymin=0.25, ymax=0.75, linewidth=2, color='g',  ls='--', label='kkkkk')
    plt.text(1,0.75,'blah')
    plt.legend()

    scat = plt.scatter(data[i],pata[i], c='b')
    return scat,

fig1 = plt.figure()
pointsX =[]
pointsY=[]


for j in xrange(50):
    pointsX.append(j)
    pointsY.append(j)
scat = plt.scatter(pointsX,pointsY, c='r')

color_data = np.random.random((10, 50))
ani = animation.FuncAnimation(fig1, update_plot, frames=xrange(10), fargs=( color_data,color_data, scat),blit=False)
ani.save('static/videos/testao.mp4', fps=30, writer="ffmpeg", extra_args=['-vcodec', 'libx264'])

plt.show()

