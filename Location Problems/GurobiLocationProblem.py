from gurobipy import *
import math
import numpy
from random import randint
#This section declares all model constant values
#Element i indicates the (x,y) location of client i
clients = [(100, 110), (120, 388), (413, 180), (500, 520)]
15
#Element i indicates the (x,y) location of facility i
facilities = [(50, 75), (50, 255), (200, 215), (400, 300),
(510, 65), (100, 450), (340, 50), (490, 350), (500, 500)]
#Element i indicates the cost of opening a facility of size 1 at location i
facilityUnitCosts = [1, 2, 4, 5, 1, 2, 1.5, 2, 3]
#Usefull values
numFacilities = len(facilities)
numClients = len(clients)
#Data structures to hold the model variables and constants
x = {}; n = {}; f = {}; y = {}; w = {}; l = {}; d = {}
#Number of possible different capacities for a facility
facilityCapacities = 30
# When multiplied by facility-capacity value, returns the number of cars that the
facility can hold. i.e
# if the capacity is 7, then the number of cars the facility can hold is
7*capacityMultiplier
capacityMultiplier = 10
#Total number of time units the model will run for
totalTime = 24
#Ratio of distance to facility cost in objective function
alpha = 0.5
#Given the index of a capacity, returns the number of cars that that a facility of that
#capacity can hold.
16
def capacity(c):
return capacityMultiplier * c
#Calculates the distance of the path a -> b
def distance((x0, y0), (x1, y1)):
dx = x0 - x1
dy = y0 - y1
return math.sqrt(dx * dx + dy * dy)
#Returns the demand of cars going from client i to client j at time t.
def demand(i, j, t):
if (t >= totalTime - 1): return 0
else: return randint(0, 100)
#START MODEL
# This function is responsible for adding all the variables and constants that will be used
# by the model
def addVariables(m):
#Variable X[k,c] binary variable which indicates if facility k of capacity c is open
for k in xrange(numFacilities):
for c in xrange(facilityCapacities):
x[(k, c)] = m.addVar(vtype=GRB.BINARY, name="X_%d,%d" % (k, c))
for c in xrange(facilityCapacities):
#Variable indicating the number of cars that a facility of capacity c can hold
n[c] = capacity(c)
17
for k in xrange(numFacilities):
#Variable f[k] cost of holding 1 car at facility k.
f[k] = facilityUnitCosts[k]
for t in xrange(totalTime):
for i in xrange(numClients):
for k in xrange(numFacilities):
#Variable Y_(i,k,t) variable indicating number of cars that facility k sends to client i
#at time t.
y[(i,k,t)] = m.addVar(lb=0, vtype=GRB.INTEGER, name="Y_%d,%d,%d" % (i,k,t))
#Variable W_(i,k,t) variable indicating number of cars that client i sends to facility k
#at time t.
w[(i,k,t)] = m.addVar(lb=0, vtype=GRB.INTEGER, name="W_%d,%d,%d" % (i,k,t))
for i in xrange(numClients):
for j in xrange(numClients):
for t in xrange(totalTime):
#Demand of cars needing to go from i to j at time t
l[(i, j, t)] = demand(i, k, t)
for i in xrange(numClients):
for k in xrange(numFacilities):
#Variable d_(i,k,j) indicates the distance of the path starting at facility k,
#going to client i
d[(i, k)] = distance(clients[i], facilities[k])
18
# This function is responsible for adding all the necessary constraints to the
# model
def addConstraints(m):
#Define two variables which will make constraints easier to understand
#Number of cars at location i at start time ti
def s(i, ti): return quicksum((y[(i,k,t)] - w[(i,k,t)])
for t in xrange(ti) for k in xrange(numFacilities)) +
quicksum((l[(j, i, t)] - l[(i, j, t)]) for t in xrange(ti)
for j in xrange(numClients))
#Number of clients at facility i start of time ti
def r(k, ti): return quicksum(n[c]*x[(k,c)]
for c in xrange(facilityCapacities)) +
quicksum((w[(i,k,t)] - y[(i,k,t)])
for t in xrange(ti) for i in xrange(numClients))
# Add constraints
for k in xrange(numFacilities):
#At most one size of a facility is open per location
m.addConstr(quicksum(x[(k,c)] for c in xrange(facilityCapacities)) <= 1)
for i in xrange(numClients):
#At final time, cars can only return to open facilities
m.addConstr(w[(i,k,totalTime - 1)] <= quicksum(n[c] * x[(k,c)] for
c in xrange(facilityCapacities)))
19
for t in xrange(totalTime):
for i in xrange(numClients):
#Excess cars at a given location go back to some facility
m.addConstr(quicksum(w[(i, k, t)] for k in xrange(numFacilities))
== (s(i, t) + quicksum(y[(i,k,t)] for k in xrange(numFacilities)) -
quicksum(l[(i,j,t)] for j in xrange(numClients))))
#Demand between all locations is satisfied
m.addConstr(quicksum(l[(i, j, t)] for j in xrange(numClients)) <=
quicksum(y[(i, k, t)] for k in xrange(numFacilities)) + s(i, t))
for t in xrange(totalTime):
for k in xrange(numFacilities):
#A facility can not send out more cars than it has at that time
m.addConstr(quicksum(y[(i, k, t)] for i in
xrange(numClients)) <= r(k, t))
#The number of cars at a facility must be less than its capacity
m.addConstr(r(k, t) <= quicksum(n[c] * x[(k,c)] for c in xrange(facilityCapacities)))
#All cars go back to some facility at the end of the day. Note that in order
for this to
be a viable constraint, we need to make the final unit of time
#have a demand of 0, such that cars can go back
m.addConstr(quicksum(y[(i, k, t)] - w[(i, k, t)] for t in xrange(totalTime)
for i
in xrange(numClients) for k in xrange(numFacilities)) == 0)
20
# This function is responsible for setting up the model and defining the objective
# function of the model
def startModel(model):
addVariables(model)
addConstraints(model)
model.setObjective(quicksum(n[c]*x[(k, c)]*f[k] for c in
xrange(facilityCapacities)
for k in xrange(numFacilities)) + alpha * quicksum(w[(i, k, t)] +
y[(i, k, t)] * d[(i, k)]
for i in xrange(numClients) for k in xrange(numFacilities)
for t in xrange(totalTime)) )
#NOTE: may need to multiply here by x[k, c]
def printResults(model):
vars = model.getVars()
for var in vars:
if(var.x != 0): print(var.varName + " = %.1f" % var.x)
def createTimeDict(d, md):
for (i, j, t) in md.keys():
var = md.get((i, j, t))
if (var != None and var.x != 0):
v = d.get(t)
if (v != None):
v.append((i, j, var.x))
else:
d[t] = [(i, j, var.x)]
21
def processResults():
facToLoc = {}; locToFac = {}
createTimeDict(facToLoc, y)
createTimeDict(locToFac, w)
return (facToLoc, locToFac)
#This function is responsible for setting up the model and invoking
#the Gurobi Optimizer on it.
# It returns tuple a, b, c, where:
# a : list of (k, c) where k is index of location and c is capactity,
#it only includes non-zero locations.
# b : dictionary (k, v) where k is the time unit and v is a list of
#tuples (x, y, z) which denote Y_(x, y, v) = z.
# c : dictionary (k, v) where k is the time unit and v is a list of
#tuples (x, y, z) which denote W_(x, y, v) = z.
def optimizeModel():
m = Model()
startModel(m)
m.ModelSense = GRB.MINIMIZE
m.optimize()
#printResults(m)
(b, c) = processResults()
a = [k for (k,l) in x if (x.get((k,l)).x != 0)]
return (a, b, c)
optimizeModel()
22
The following code used Python’s Tkinter to create an animation of the model’s results:
from Tkinter import *
from math import sqrt
import Tkinter as tk
import FacillityAllocationWithDemandsAndTime as FacAlloc
import time
facilityLocations = [(100,100),(100,300),(200,200),(200,300),(300,200),
(300,300)]
clientLocations = [(100,200),(200,100),(300,100)]
w = 40
class Facility(object):
def __init__(self, canvas, i, b, **kwargs):
self.canvas = canvas
(x0, y0) = facilityLocations[i]
self.x0 = x0
self.y0 = y0
self.color = "black" if b else "grey"
self.id = self.canvas.create_rectangle(x0, y0, x0 + w, y0 + w, outline = self.color)
self.text = self.canvas.create_text(x0 + w/2, y0 + w/2, text=str(i),
anchor = tk.CENTER, fill = self.color)
def updateText(self, newVal):
self.canvas.delete(self.text)
23
self.text = self.canvas.create_text(self.x0 + w/2, self.y0 + w/2, text=newVal,
anchor = tk.CENTER, fill = self.color)
class Client(object):
def __init__(self, canvas, i, **kwargs):
self.canvas = canvas
(x0, y0) = clientLocations[i]
self.id = self.canvas.create_oval(x0, y0, x0 + w, y0 + w)
self.x0 = x0
self.y0 = y0
self.text = self.canvas.create_text(x0 + w/2, y0 + w/2, text=str(i),
anchor = tk.CENTER)
def updateText(self, newVal):
self.canvas.delete(self.text)
self.text = self.canvas.create_text(self.x0 + w/2, self.y0 + w/2, text=newVal,
anchor = tk.CENTER)
class DemandLine(object):
def __init__(self, canvas, i, j, demand, **kwargs):
self.canvas = canvas
(x0, y0) = clientLocations[i]
(x1, y1) = clientLocations[j]
self.i = i
self.j = j
self.x0 = x0
self.x1 = x1
self.y0 = y0
24
self.y1 = y1
off = 0 if i < j else 8
if (x0 < x1):
self.id = self.canvas.create_line(x0 + w, y0 + w/2 + off, x1,
y1 + w/2 + off, **kwargs)
else:
self.id = self.canvas.create_line(x0, y0 + w/2 + off, x1 + w,
y1 + w/2 + off, **kwargs)
x_diff = x1 - x0; y_diff = y1 - y0
self.text = self.canvas.create_text(x0 + w/2+ x_diff/2,
y0+ w/2 + y_diff/2 + off, text= "", anchor = tk.CENTER)
def delete(self):
self.canvas.delete(self.id)
def updateText(self, time):
self.canvas.delete(self.text)
off = -5 if self.i < self.j else 5
x_diff = self.x1 - self.x0; y_diff = self.y1 - self.y0
val = FacAlloc.demand(self.i, self.j, time)
self.text = self.canvas.create_text(self.x0 + w/2+ x_diff/2,
self.y0+ w/2 + y_diff/2 + off, text= str(val), anchor = tk.CENTER)
class FlowLine(object):
def __init__(self, canvas, i, k, v, o, **kwargs):
self.canvas = canvas
25
(x0, y0) = clientLocations[i]
(x1, y1) = facilityLocations[k]
if (o == 1): off = 1
else: off = -1
self.id = self.canvas.create_line(x0 + w/2, y0 + w/2 + off,
x1 + w/2, y1 + w/2 + off, **kwargs)
def delete(self):
self.canvas.delete(self.id)
class App(object):
def __init__(self, master, **kwargs):
(self.xLoc, self.y, self.w) = FacAlloc.optimizeModel()
self.master = master
self.canvas = tk.Canvas(self.master, width = 600, height = 600)
self.canvas.pack()
self.time = 0
self.drawDemands()
self.createFacilities()
self.createClients()
self.canvas.pack()
self.master.after(0, self.animation)
def createFacilities(self):
self.facilities = [Facility(self.canvas, i, (1 if i in self.xLoc else 0))
for i in range(len(facilityLocations))]
26
def createClients(self):
self.clients = [Client(self.canvas, i) for i in range(len(clientLocations))]
def drawDemands(self):
self.currentDemands = [DemandLine(self.canvas, i, j, FacAlloc.demand(i, j, 0), arrow=tk.FIRST) for i in range(len(clientLocations))
for j in range(len(clientLocations)) if (i != j)]
def updateDemands(self, time):
for demand in self.currentDemands: demand.updateText(time)
def drawYIKT(self, time):
self.currentYIKT = []
if (self.y.get(time) != None):
for (i, k ,v) in self.y[time]:
self.currentYIKT.append(FlowLine(self.canvas, i, k, v, 1,
fill="blue", arrow=tk.FIRST, width = 2))
def drawWIKT(self, time):
self.currentWIKT = []
if (self.w.get(time) != None):
for (i, k , v) in self.w[time]:
self.currentWIKT.append(FlowLine(self.canvas, i, k, v, 0,
fill="red", arrow=tk.LAST, width = 2))
def animation(self):
currTime = self.time % 10
# if (time % self.step == 0):
# self.drawSIT(currTime)
27
if (currTime == 0):
self.updateDemands(self.time/10)
if (currTime == 3):
self.drawYIKT(self.time/10)
if (currTime == 6):
self.drawWIKT(self.time/10)
if (currTime == 9):
for line in self.currentYIKT:
line.delete()
for line in self.currentWIKT:
line.delete()
self.time += 1
self.master.after(300, self.animation)
root = tk.Tk()
app = App(root)
root.mainloop()