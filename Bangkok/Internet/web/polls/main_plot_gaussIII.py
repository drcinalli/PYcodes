from pylab import *
from axes3Dedited import Axes3D

x = linspace(-5, 5, 200)
y = x
X,Y = meshgrid(x, y)
Z = bivariate_normal(X, Y)

for cmap in colormaps():
    fig = figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(X, Y, Z, cmap= cmap)
    title(cmap)
    plt.show()