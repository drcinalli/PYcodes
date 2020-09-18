__author__ = 'quatrosem'
import shapely.geometry as geometry
import numpy as np
import pylab as pl
from shapely.ops import cascaded_union, polygonize
from scipy.spatial import Delaunay
import numpy as np
import math
from descartes import PolygonPatch
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.collections import LineCollection

def plot_polygon(polygon):
    fig = pl.figure(figsize=(10,10))
    ax = fig.add_subplot(111)
    margin = .3

    x_min, y_min, x_max, y_max = polygon.bounds

    ax.set_xlim([x_min-margin, x_max+margin])
    ax.set_ylim([y_min-margin, y_max+margin])
    patch = PolygonPatch(polygon, fc='lightsalmon', ec='black', fill=True, zorder=-1)
    ax.add_patch(patch)
    return fig


def alpha_shape(points, alpha):
    """
    Compute the alpha shape (concave hull) of a set of points.

    @param points: Iterable container of points.
    @param alpha: alpha value to influence the gooeyness of the border. Smaller
                  numbers don't fall inward as much as larger numbers. Too large,
                  and you lose everything!
    """
    if len(points) < 4:
        # When you have a triangle, there is no sense in computing an alpha
        # shape.
        return geometry.MultiPoint(list(points)).convex_hull

    def add_edge(edges, edge_points, coords, i, j):
        """Add a line between the i-th and j-th points, if not in the list already"""
        if (i, j) in edges or (j, i) in edges:
            # already added
            return
        edges.add( (i, j) )
        edge_points.append(coords[ [i, j] ])

    coords = np.array([point.coords[0] for point in points])

    tri = Delaunay(coords)
    edges = set()
    edge_points = []
    # loop over triangles:
    # ia, ib, ic = indices of corner points of the triangle
    for ia, ib, ic in tri.vertices:
        pa = coords[ia]
        pb = coords[ib]
        pc = coords[ic]

        # Lengths of sides of triangle
        a = math.sqrt((pa[0]-pb[0])**2 + (pa[1]-pb[1])**2)
        b = math.sqrt((pb[0]-pc[0])**2 + (pb[1]-pc[1])**2)
        c = math.sqrt((pc[0]-pa[0])**2 + (pc[1]-pa[1])**2)

        # Semiperimeter of triangle
        s = (a + b + c)/2.0

        # Area of triangle by Heron's formula
        area = math.sqrt(s*(s-a)*(s-b)*(s-c))
        circum_r = a*b*c/(4.0*area)

        # Here's the radius filter.
        #print circum_r
        if circum_r < 1.0/alpha:
            add_edge(edges, edge_points, coords, ia, ib)
            add_edge(edges, edge_points, coords, ib, ic)
            add_edge(edges, edge_points, coords, ic, ia)

    m = geometry.MultiLineString(edge_points)
    triangles = list(polygonize(m))
    return cascaded_union(triangles), edge_points

def PolyArea2D(pts):
    lines = np.hstack([pts,np.roll(pts,-1,axis=0)])
    area = 0.5*abs(sum(x1*y2-x2*y1 for x1,y1,x2,y2 in lines))
    return area

# The following proves that I don't know numpy in the slightest
p= np.random.rand(10, 2)
#p = np.array([[0.93265155,  0.01096512],[ 0.98332358,  0.49974818],[ 0.98222118,  0.66835051],[ 0.96447573,  0.74072951], [ 0.78279442,  0.93237628],[ 0.09160285,  0.97272736], [ 0.02394609,  0.53555955], [ 0.0961685 ,  0.11431132], [ 0.52804919,  0.04385874]])
#x = [p.coords.xy[0] for p in points]
#y = [p.coords.xy[1] for p in points]

#pl.figure(figsize=(10,10))
#pl.plot(p[:,0],p[:,1],'o', color='#f16824')


point_collection = geometry.MultiPoint(list(p))
point_collection.envelope

#plot_polygon(point_collection.envelope)
#pl.plot(p[:,0],p[:,1],'o', color='#f16824')

convex_hull_polygon = point_collection.convex_hull
plot_polygon(convex_hull_polygon)
pl.plot(p[:,0],p[:,1],'o', color='black')



# concave_hull, edge_points = alpha_shape(point_collection, alpha=5)
# plot_polygon(concave_hull)
# pl.plot(p[:,0],p[:,1],'o', color='#f16824')


for i in range(9):
    alpha = (i+1)*1#.6 #+ 4
    concave_hull, edge_points = alpha_shape(point_collection, alpha=alpha)

    #print concave_hull
    lines = LineCollection(edge_points)
    pl.figure(figsize=(10,10))
    pl.title('Alpha={0} Delaunay triangulation'.format(alpha))
    pl.gca().add_collection(lines)
    delaunay_points = np.array([point.coords[0] for point in point_collection])
    pl.plot(delaunay_points[:,0], delaunay_points[:,1], 'o', hold=1, color='#f16824')

    #plot_polygon(concave_hull)
    pl.plot(p[:,0],p[:,1],'o', color='#f16824')

#PolyArea2D(concave_hull)
print concave_hull.area


#3d

# The following proves that I don't know numpy in the slightest
p= np.random.rand(30, 3)


#original points
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

x = p[:,0]
y = p[:,1]
z = p[:,2]
ax.scatter(x, y, z, c='r', marker='o')

ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')


point_collection = geometry.MultiPoint(list(p))
point_collection.envelope

#alpha 3d
alpha = (1+1)*.6 + 4
#concave_hull, edge_points = alpha_shape(point_collection, alpha=alpha)

#draw 3d

#volume 3d


pl.show()