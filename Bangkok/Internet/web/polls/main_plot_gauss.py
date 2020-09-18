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

from ast import literal_eval


matplotlib.rcParams['xtick.direction'] = 'out'
matplotlib.rcParams['ytick.direction'] = 'out'


fake = [[78.911204246880132, -23.0], [78.911204246880132, -23.0], [78.911204246880132, -23.0], [78.911204246880132, -23.0], [78.911204246880132, -23.0], [78.911204246880132, -23.0], [78.911204246880132, -23.0], [78.911204246880132, -23.0], [78.911204246880132, -23.0], [82.833563725395152, -46.0], [82.833563725395152, -46.0], [82.833563725395152, -46.0], [82.833563725395152, -46.0], [82.833563725395152, -46.0], [82.833563725395152, -46.0], [82.833563725395152, -46.0], [82.833563725395152, -46.0], [82.833563725395152, -46.0], [82.833563725395152, -46.0], [82.833563725395152, -46.0], [82.833563725395152, -46.0], [82.833563725395152, -46.0], [82.833563725395152, -46.0], [82.833563725395152, -46.0], [88.404743229436974, -58.0], [88.404743229436974, -58.0], [88.404743229436974, -58.0], [88.404743229436974, -58.0], [88.404743229436974, -58.0], [88.404743229436974, -58.0], [88.404743229436974, -58.0], [88.404743229436974, -58.0], [88.404743229436974, -58.0], [88.404743229436974, -58.0], [88.635650570838351, -69.0], [88.635650570838351, -69.0], [88.635650570838351, -69.0], [88.635650570838351, -69.0], [88.635650570838351, -69.0], [88.635650570838351, -69.0], [88.635650570838351, -69.0], [88.635650570838351, -69.0], [93.884715335422769, -81.0], [93.884715335422769, -81.0], [93.884715335422769, -81.0], [93.884715335422769, -81.0], [93.884715335422769, -81.0], [93.884715335422769, -81.0], [97.404743229436974, -93.0], [97.404743229436974, -93.0], [97.404743229436974, -93.0], [97.404743229436974, -93.0], [97.404743229436974, -93.0], [97.404743229436974, -93.0], [103.57772817821186, -104.0], [103.57772817821186, -104.0], [103.57772817821186, -104.0], [103.57772817821186, -104.0], [103.57772817821186, -104.0], [103.57772817821186, -104.0], [103.57772817821186, -104.0], [109.80326398495583, -128.0], [109.80326398495583, -128.0], [128.92636961057349, -163.0], [131.40224255080034, -175.0], [131.40224255080034, -175.0], [136.67004637770998, -186.0], [136.67004637770998, -186.0], [136.67004637770998, -186.0], [136.67004637770998, -186.0], [144.18033988749895, -198.0], [152.90038970424035, -210.0], [152.90038970424035, -210.0], [152.90038970424035, -210.0]]
t = np.array(fake, np.int32)
gmm = mixture.GMM(n_components=3, covariance_type='spherical')
gmm.fit(t)


delta = 0.51
x = np.arange(0, 250.0, delta)
y = np.arange(-250.0, 0.0, delta)
X, Y = np.meshgrid(x, y)

objs = []
Z1 = bivariate_normal(X, Y, sigmax=np.sqrt(gmm.covars_[0][0]), sigmay=np.sqrt(gmm.covars_[0][0]),mux=gmm.means_[0][0], muy=gmm.means_[0][1])
objs.append(Z1)
Z2 = bivariate_normal(X, Y, sigmax=np.sqrt(gmm.covars_[1][0]), sigmay=np.sqrt(gmm.covars_[1][0]),mux=gmm.means_[1][0], muy=gmm.means_[1][1])
objs.append(Z2)
Z3 = bivariate_normal(X, Y, sigmax=np.sqrt(gmm.covars_[2][0]), sigmay=np.sqrt(gmm.covars_[2][0]),mux=gmm.means_[2][0], muy=gmm.means_[2][1])
objs.append(Z3)


for i in xrange(len(objs)):
    if i==0:
        Z = objs[i]
    else:
        Z += objs[i]

#Z = 10.0 * (Z2 + Z1 + Z3)
Z = 10.0 * Z






#Contour

# Create a simple contour plot with labels using default colors.  The
# inline argument to clabel will control whether the labels are draw
# over the line segments of the contour, removing the lines beneath
# the label
plt.figure()

#colors = ['r' if i==0 else 'g' for i in gmm.predict(t)]
colors=[]
for i in gmm.predict(t):
    if i==0:
        colors.append('r')
    elif i==1:
        colors.append('b')
    else:
        colors.append('g')



ax = plt.gca()
ax.scatter(t[:,0], t[:,1], c=colors, alpha=0.8)
#plt.scatter(t[:, 0], t[:, 1], alpha=0.5, s=1)

CS = plt.contour(X, Y, Z)
plt.clabel(CS, inline=1, fontsize=10)
plt.title('Simplest default with labels')


plt.show()


# AQUI entrara o gauss

from pylab import *
from axes3Dedited import Axes3D

fig = figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(X, Y, Z, cmap= 'Greens')
title('Greens')
plt.show()

#
# from pylab import *
# from axes3Dedited import Axes3D
#
# delta = 0.02
# x = np.arange(55.0, 59.0, delta)
# y = np.arange(50.0, 58.0, delta)
# X, Y = np.meshgrid(x, y)
# Z1 = mlab.bivariate_normal(X, Y, 0.5, 0.5, 57.0, 52.0)
# Z2 = mlab.bivariate_normal(X, Y, 0.7, 0.7, 57.0, 55.0)
# # difference of Gaussians
# Z = 10.0 * (Z2 - Z1)
#
# for cmap in colormaps():
#     fig = figure()
#     ax = fig.add_subplot(111, projection='3d')
#     ax.plot_surface(X, Y, Z, cmap= cmap)
#     title(cmap)
#     plt.show()