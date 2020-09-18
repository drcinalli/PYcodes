__author__ = 'quatrosem'
from scipy.spatial import ConvexHull
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from matplotlib import cm
from scipy.spatial import Delaunay
import shapely.geometry as geometry
import pylab as pl
from descartes import PolygonPatch



def PolyArea2D(pts):
    lines = np.hstack([pts,np.roll(pts,-1,axis=0)])
    area = 0.5*abs(sum(x1*y2-x2*y1 for x1,y1,x2,y2 in lines))
    return area

def tetrahedron_volume(a, b, c, d):
    return np.abs(np.einsum('ij,ij->i', a-d, np.cross(b-d, c-d))) / 6

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

#2D
points = np.random.rand(21, 2)   # 30 random points in 2-D
points =[[0.267573324636,0.499935963785],[0.313780686309,0.470941301839],[0.321679926853,0.450922448141],[0.32194988808,0.44605652089],[0.322661910174,0.445519729375],[0.329960911492,0.44336775149],[0.330193859535,0.442053030479],[0.330759362515,0.437495106484],[0.336886446059,0.431429967876],[0.340989560726,0.428167902501],[0.342490052342,0.427015077166],[0.343216508605,0.426312465565],[0.343432633518,0.425111854602],[0.344946143704,0.424891184731],[0.344957622331,0.424344386067],[0.345289239934,0.423674954902],[0.34596048499,0.423047878845],[0.346211838763,0.423040760891],[0.346543799837,0.422029322873],[0.348375557455,0.421146826007],[0.350160925466,0.419691725746],[0.350691006283,0.4188785474],[0.351957184075,0.418525426179],[0.352058049934,0.41780578734],[0.352378030186,0.417451662293],[0.352781110356,0.417399255686],[0.353032872833,0.41399985214],[0.355610653101,0.413312740825],[0.359000256588,0.413276777582],[0.359027982895,0.410784127495],[0.359341684591,0.409436172665],[0.361657265261,0.409361397805],[0.361719295511,0.40806596506],[0.361935420425,0.407278222919],[0.363169575074,0.406177100059],[0.363205304783,0.405576218026],[0.365043694504,0.403668627755],[0.366296987651,0.402511368291],[0.369107974596,0.401606361326],[0.369130138026,0.401515508394],[0.370842300058,0.401022562153],[0.372154813903,0.400284490885],[0.372247084539,0.400057611177],[0.372322461182,0.398435637538],[0.372886508229,0.398057823472],[0.372961668043,0.397163286895],[0.373341895851,0.396791641651],[0.375618322055,0.396363877749],[0.376411895912,0.396058396235],[0.376428616198,0.39520741304],[0.377365031241,0.395130223769],[0.377399897418,0.394530108266],[0.377729111977,0.394254718837],[0.377818802722,0.393720744107],[0.378188957522,0.393256754683],[0.378840016087,0.392636563477],[0.381078981841,0.39103603695],[0.38209034561,0.390612426471],[0.382114619725,0.389403937837],[0.384305378635,0.389287291825],[0.387558460944,0.388554147261],[0.387929333396,0.386910996837],[0.388092882059,0.386706220413],[0.388115045489,0.386616621749],[0.389559541457,0.385476221786],[0.389861958403,0.385283658209],[0.390591982655,0.384629665497],[0.396453245144,0.383846541823],[0.396808076749,0.38058343982],[0.3994893686,0.378841830148],[0.400726009347,0.378301428374],[0.400826875207,0.377312823661],[0.40137428012,0.375432706575],[0.402072666371,0.374562971202],[0.412522263722,0.367905577802],[0.482785790774,0.321354961897],[0.510963739434,0.300376478859]]
points = [[0.000631552621713,167.114041512] , [0.00157147685541,64.6160831032] , [0.00212747239423,47.2371097282]] #0.02 disper=3560
points = [[0.37,0.50],[0.393259735623,0.382188143127],[0.393615278841,0.381886451953],[0.39405474752,0.381512420089],
          [0.395091968519,0.380679475644],[0.395109082483,0.380614684299],[0.395295738039,0.380465232928],
          [0.408261871443,0.369316774423],[0.408275915792,0.368883285619],[0.411276818843,0.366525280021],
          [0.415895258882,0.364256954454],[0.417669839666,0.361320864006],[0.424153087509,0.356573449786],
          [0.42807425346,0.355092158185],[0.446079259803,0.339581316724],[0.44626591536,0.339440811983],[0.446912750852,0.339039528848],
          [0.447759719002,0.338318344603]]



point_collection = geometry.MultiPoint(list(points))
point_collection.envelope
convex_hull_polygon = point_collection.convex_hull
plot_polygon(convex_hull_polygon)
for i in points:
    pl.plot(i[0],i[1],'o', color='black')
#pl.plot(points[:jbnbn],points[:,1],'o', color='black')


hull = ConvexHull(points)
#plt.plot(points[:,0], points[:,1], 'o')
pts=[]
#for simplex in hull.simplices:
#    plt.plot(points[simplex,0], points[simplex,1], 'k-')

for vert in hull.vertices:
    pts.append(points[vert])

#plt.plot(points[hull.vertices,0], points[hull.vertices,1], 'r--', lw=2)
#plt.plot(points[hull.vertices[0],0], points[hull.vertices[0],1], 'ro')
#plt.show()



#area of 2D Hull
print pts
print PolyArea2D(pts)
#pts = [[0,0],[1,0],[1,1],[0,1]]
#print PolyArea2D(pts)




#3D
points = np.random.rand(10, 3)   # 30 random points in 2-D
hull = ConvexHull(points)

pts3d = []
for vert in hull.vertices:
    pts3d.append(points[vert])


#original points
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

x = points[:,0]
y = points[:,1]
z = points[:,2]
ax.scatter(x, y, z, c='r', marker='o')

ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')


#convex hull points
fig2 = plt.figure()
ax = fig2.add_subplot(111, projection='3d')


x = points[hull.vertices,0]
y = points[hull.vertices,1]
z = points[hull.vertices,2]

ax.scatter(x, y, z, c='r', marker='o')

ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')


#fill the Hull ... Anyway!
fig4 = plt.figure()
ax = fig4.add_subplot(111, projection='3d')


#tri = Delaunay(np.array([u,v]).T)
surf = ax.plot_trisurf(x, y, z, cmap=cm.jet, linewidth=0)
#surf = ax.plot_trisurf(x, y, z, triangles=tri.simplices, cmap=plt.cm.Spectral)
fig4.colorbar(surf)

ax.xaxis.set_major_locator(MaxNLocator(5))
ax.yaxis.set_major_locator(MaxNLocator(6))
ax.zaxis.set_major_locator(MaxNLocator(5))

ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')


fig4.tight_layout()

###########################################
#fill the Hull ... anyway Cuboid!
fig5 = plt.figure()
ax = fig5.add_subplot(111, projection='3d')

# u = np.array([0,0,0.5,1,1])
# v = np.array([0,1,0.5,0,1])
#
# x = u
# y = v
# z = np.array([0,0,1,0,0])

#pts3d = [[0,0,0],[0,2,0],[2,2,0],[2,0,0],[0,0,2],[0,2,2],[2,2,2],[2,0,2]]
u = np.array([0,0,2,2,0,0,2,2])
v = np.array([0,2,2,0,0,2,2,0])

x = u
y = v
z = np.array([0,0,0,0,2,2,2,2])



tri = Delaunay(np.array([u,v]).T)
#surf = ax.plot_trisurf(x, y, z, cmap=cm.jet, linewidth=0)
surf = ax.plot_trisurf(x, y, z, triangles=tri.simplices, cmap=plt.cm.Spectral)
fig5.colorbar(surf)

ax.xaxis.set_major_locator(MaxNLocator(5))
ax.yaxis.set_major_locator(MaxNLocator(6))
ax.zaxis.set_major_locator(MaxNLocator(5))

ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')

fig5.tight_layout()

###################


from scipy.spatial.distance import euclidean
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d as a3
import pylab as pl
import scipy as sp

# aspect = 0
# while aspect == 0:

# Generate random points & convex hull
#points = np.random.rand(25,3)
pointsz = np.array([[0,0,0],[0,2,0],[2,2,0],[2,0,0],[0,0,2],[0,2,2],[2,2,2],[2,0,2]])
#points = np.array(pts3d)
hull = ConvexHull(pointsz)

# Check aspect ratios of surface facets
aspectRatio = []
for simplex in hull.simplices:
    a = euclidean(pointsz[simplex[0],:], pointsz[simplex[1],:])
    b = euclidean(pointsz[simplex[1],:], pointsz[simplex[2],:])
    c = euclidean(pointsz[simplex[2],:], pointsz[simplex[0],:])
    circRad = (a*b*c)/(np.sqrt((a+b+c)*(b+c-a)*(c+a-b)*(a+b-c)))
    inRad = 0.5*np.sqrt(((b+c-a)*(c+a-b)*(a+b-c))/(a+b+c))
    aspectRatio.append(inRad/circRad)

    # # Threshold for minium allowable aspect raio of surface facets
    # print np.amin(aspectRatio)
    # if np.amin(aspectRatio) > 0:
    #     aspect = 1

ax = a3.Axes3D(pl.figure())
facetCol = sp.rand(3) #[0.0, 1.0, 0.0]

# Plot hull's vertices
for vert in hull.vertices:
   ax.scatter(pointsz[vert,0], pointsz[vert,1], zs=pointsz[vert,2])

# Plot surface traingulation
for simplex in hull.simplices:
    vtx = [pointsz[simplex[0],:], pointsz[simplex[1],:], pointsz[simplex[2],:]]
    tri = a3.art3d.Poly3DCollection([vtx], linewidths = 2, alpha = 0.8)
    tri.set_color(facetCol)
    tri.set_edgecolor('k')
    ax.add_collection3d(tri)

ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')

#plt.axis('off')

##################

from scipy.spatial.distance import euclidean
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d as a3
import pylab as pl
import scipy as sp

# aspect = 0
# while aspect == 0:

# Generate random points & convex hull
#points = np.random.rand(25,3)
#points = np.array([[0,0,0],[0,2,0],[2,2,0],[2,0,0],[0,0,2],[0,2,2],[2,2,2],[2,0,2]])
#points = np.array(pts3d)
hull = ConvexHull(points)

# Check aspect ratios of surface facets
aspectRatio = []
for simplex in hull.simplices:
    a = euclidean(points[simplex[0],:], points[simplex[1],:])
    b = euclidean(points[simplex[1],:], points[simplex[2],:])
    c = euclidean(points[simplex[2],:], points[simplex[0],:])
    circRad = (a*b*c)/(np.sqrt((a+b+c)*(b+c-a)*(c+a-b)*(a+b-c)))
    inRad = 0.5*np.sqrt(((b+c-a)*(c+a-b)*(a+b-c))/(a+b+c))
    aspectRatio.append(inRad/circRad)

    # # Threshold for minium allowable aspect raio of surface facets
    # print np.amin(aspectRatio)
    # if np.amin(aspectRatio) > 0:
    #     aspect = 1

ax = a3.Axes3D(pl.figure())
facetCol = sp.rand(3) #[0.0, 1.0, 0.0]

# Plot hull's vertices
for vert in hull.vertices:
   ax.scatter(points[vert,0], points[vert,1], zs=points[vert,2])

# Plot surface traingulation
for simplex in hull.simplices:
    vtx = [points[simplex[0],:], points[simplex[1],:], points[simplex[2],:]]
    tri = a3.art3d.Poly3DCollection([vtx], linewidths = 2, alpha = 0.8)
    tri.set_color('forestgreen')
    tri.set_edgecolor('k')
    ax.add_collection3d(tri)

ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')

#plt.axis('off')
####################


plt.show()

#Volume of Hull
#pts[hull.vertices]
#pts3d = [[0,0,0],[0,2,0],[2,2,0],[2,0,0],[0,0,2],[0,2,2],[2,2,2],[2,0,2]]
dt = Delaunay(pts3d)
tets = dt.points[dt.simplices]
vol = np.sum(tetrahedron_volume(tets[:, 0], tets[:, 1],
                                tets[:, 2], tets[:, 3]))

print vol

print "fim"