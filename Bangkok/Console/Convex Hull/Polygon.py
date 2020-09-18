__author__ = 'quatrosem'
import shapely.geometry as geometry
import numpy as np
import pylab as pl

# The following proves that I don't know numpy in the slightest
p= np.random.rand(211, 2)
#p = np.array([[0.93265155,  0.01096512],[ 0.98332358,  0.49974818],[ 0.98222118,  0.66835051],[ 0.96447573,  0.74072951], [ 0.78279442,  0.93237628],[ 0.09160285,  0.97272736], [ 0.02394609,  0.53555955], [ 0.0961685 ,  0.11431132], [ 0.52804919,  0.04385874]])
#x = [p.coords.xy[0] for p in points]
#y = [p.coords.xy[1] for p in points]

pl.figure(figsize=(10,10))
pl.plot(p[:,0],p[:,1],'o', color='#f16824')


point_collection = geometry.MultiPoint(list(p))
point_collection.envelope

from descartes import PolygonPatch

def plot_polygon(polygon):
    fig = pl.figure(figsize=(10,10))
    ax = fig.add_subplot(111)
    margin = .3

    x_min, y_min, x_max, y_max = polygon.bounds

    ax.set_xlim([x_min-margin, x_max+margin])
    ax.set_ylim([y_min-margin, y_max+margin])
    patch = PolygonPatch(polygon, fc='#999999', ec='#000000', fill=True, zorder=-1)
    ax.add_patch(patch)
    return fig

plot_polygon(point_collection.envelope)
pl.plot(p[:,0],p[:,1],'o', color='#f16824')

convex_hull_polygon = point_collection.convex_hull
plot_polygon(convex_hull_polygon)
pl.plot(p[:,0],p[:,1],'o', color='#f16824')


pl.show()