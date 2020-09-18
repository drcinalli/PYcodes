from axes3Dedited import Axes3D

import numpy as np
from matplotlib.mlab import bivariate_normal


import itertools
from scipy import linalg
import matplotlib.pyplot as plt
import matplotlib as mpl
from sklearn import mixture

#
import matplotlib
import matplotlib.cm as cm
import matplotlib.mlab as mlab

matplotlib.rcParams['xtick.direction'] = 'out'
matplotlib.rcParams['ytick.direction'] = 'out'
#
#
# fitness = [[74, 15], [74, 11], [69, 16], [73, 21], [68, 28], [60, 27], [69, 33], [65, 27], [50, 40], [45, 35], [66, 46], [59, 46], [39, 41], [49, 55], [44, 52], [44, 39], [43, 64], [30, 45], [32, 63], [31, 63], [23, 69], [27, 69], [15, 76], [12, 77], [23, 61], [16, 81]]
# f1 = []
# f2 = []
# for i in fitness:
#     f1.append(i[0])
#     f2.append(i[1])
#
#
#
# t = np.array(fitness, np.int32)
# gmm = mixture.GMM(n_components=2, covariance_type='spherical')
# gmm.fit(fitness)
#

# fig = plt.figure(figsize=(10, 7))
# ax = fig.gca(projection='3d')
# x = np.linspace(-5, 5, 200)
# y = x
#
# x = f1
# y = f2
# X,Y = np.meshgrid(x, y)
# Z = bivariate_normal(X, Y, sigmax=gmm.covars_[0][0], sigmay=gmm.covars_[0][0],mux=gmm.means_[0][0], muy=gmm.means_[0][1], sigmaxy=0.0)




#
# delta = 1
# x = np.arange(-10, 200, delta)
# y = np.arange(-40, 200, delta)
# X, Y = np.meshgrid(x, y)

# delta = 0.025
# x = np.arange(-3.0, 3.0, delta)
# y = np.arange(-2.0, 2.0, delta)
# X, Y = np.meshgrid(x, y)
#
# Z1 = mlab.bivariate_normal(X, Y, 1.0, 1.0, 0.0, 0.0)
# Z2 = mlab.bivariate_normal(X, Y, 1.5, 0.5, 1, 1)
# # difference of Gaussians
# Z = 10.0 * (Z2 - Z1)
#
#Z1 = mlab.bivariate_normal(X, Y, sigmax=29, sigmay=17,mux=50, muy=50)
#
# delta = 1
# x = np.arange(0.0, 100.0, delta)
# y = np.arange(0.0, 100.0, delta)
# X, Y = np.meshgrid(x, y)
# Z1 = mlab.bivariate_normal(X, Y, 1.0, 1.0, 50.0, 50.0)
# Z2 = mlab.bivariate_normal(X, Y, 1.5, 0.5, 62.0, 62.0)
# # difference of Gaussians
# Z = 10.0 * (Z2 - Z1)
# Z1 = bivariate_normal(X, Y, sigmax=np.sqrt(gmm.covars_[0][0]), sigmay=np.sqrt(gmm.covars_[0][0]),mux=gmm.means_[0][0], muy=gmm.means_[0][1])
# Z2 = bivariate_normal(X, Y, sigmax=np.sqrt(gmm.covars_[1][0]), sigmay=np.sqrt(gmm.covars_[1][0]),mux=gmm.means_[1][0], muy=gmm.means_[1][1])
# Z = 10.0 * (Z2 + Z1)
#
#
# # Create a simple contour plot with labels using default colors.  The
# # inline argument to clabel will control whether the labels are draw
# # over the line segments of the contour, removing the lines beneath
# # the label
# plt.figure()
# CS = plt.contour(X, Y, Z)
# plt.clabel(CS, inline=1, fontsize=10)
# plt.title('Simplest default with labels')
#

#plt.show()

from pylab import *
from axes3Dedited import Axes3D

delta = 0.018
x = np.arange(55.0, 59.0, delta)
y = np.arange(50.0, 58.0, delta)
X, Y = np.meshgrid(x, y)
Z1 = mlab.bivariate_normal(X, Y, 0.5, 0.5, 57.0, 52.0)
Z2 = mlab.bivariate_normal(X, Y, 0.7, 0.7, 57.0, 55.0)
# difference of Gaussians
Z = 10.0 * (Z2 - Z1)

count=0
for cmap in colormaps():
    fig = figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(X, Y, Z, cmap= cmap)
    ax.grid(False)
    for a in (ax.w_xaxis, ax.w_yaxis, ax.w_zaxis):
        for t in a.get_ticklines()+a.get_ticklabels():
            t.set_visible(False)
        a.line.set_visible(False)
        a.pane.set_visible(False)
    title(cmap)
    plt.savefig('logogauss/3dgauss' + str(count) + '.png')
    plt.show()
    count += 1

#rstride=4, cstride=4, color='g', alpha=0.7
#rstride=1, cstride=1, cmap = cm.gray_r, alpha=0.9, linewidth=1
#Blues
#Blues_r
#BuGn
#BuGn_r
#GnBu_r
#Greens
#Greens_r
#Greys
#Greys_r
#OrRd
#Oranges
#PyBu
