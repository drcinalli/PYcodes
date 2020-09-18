'''
Created on 18/08/2014

@author: quatrosem
'''
import numpy as np
import matplotlib.pyplot as plt
from random import randint
import matplotlib.path as mpath
import matplotlib.lines as mlines
import matplotlib.patches as mpatches
from time import sleep
import math
from collections import Counter
import pickle


#Class of the Problem Domain
class World:
    #class variables
    description = "This is the domain of the problem"
    author = "Daniel Cinalli"

    #FIX
    FIX_on_off   = 1 #on=1 off=0
    FIX_strategy = 2 #loop=0 random=1 dist=2 production=3 uniform=4

    #methods
    def __init__(self,m,n):

        #World dimension
        self.m = m
        self.n = n
        self.geo = [[1 for x in xrange(self.m)] for x in xrange(self.n)]

        #self.geo_dist = [[0]*(m*n) for x in xrange(m*n)]
        #self.GeoDist(self.geo_dist)

        #extraction areas
        self.areas = []

        #files for solutions
        #0=Pareto; 1=Best Hypercube; 2=Best Cover
        self.filePareto = "ParetoOptimalFront.txt"
        self.fileBestHypercube = "ParetoSolutionHypercube.txt"
        self.fileBestCover = "ParetoSolutionCover.txt"

        #reference points for hypervolume
        self.refpointF1 = 100 #cost os distance
        self.refpointF2 = 100 #production
        self.refpointF3 = 100 #number of units

        #quantities
        self.all_areas = 5
        self.all_units = 7
        self.all_units_type = 2

        #the production and cost impact the solution... they must have values according the world dimension.
        #so the different choices for units mean something in the solution
        #production
        self.PROD_unit_0 = 23
        self.PROD_unit_1 = 35
        #costs
        self.BRL_gateway = 5
        self.BRL_unit_0 = 13
        self.BRL_unit_1 = 17
        self.BRL_allgates = self.all_areas * self.BRL_gateway


    #define number of areas in the World
    def MaxArea(self, i):
        self.all_areas = i
        #also, set the max for units, because can not be larger than the areas
        self.all_units = i

    #define number of units in the World
    def MaxUnits(self, i):
        #also, set the max for units
        self.all_units = i

    #define costs for the world
    def Costs(self, gateway, unit0, unit1):

        ##############################
        # ATENCAO, type of units here#
        ##############################
        #costs
        self.BRL_gateway = gateway
        self.BRL_unit_0 = unit0
        self.BRL_unit_1 = unit1
        self.BRL_allgates = self.all_areas * self.BRL_gateway

    #define production for the world
    def Production(self, prod0, prod1):

        ##############################
        # ATENCAO, type of units here#
        ##############################
        #production
        self.PROD_unit_0 = prod0
        self.PROD_unit_1 = prod1

    #define reference point for Hypervolume measure.
    def RefPointHypervolume(self):

        ##############################
        # ATENCAO, type of units here#
        ##############################
        self.refpointF1= (max(self.m, self.n)*self.all_areas)+ self.BRL_allgates + (self.all_units*self.BRL_unit_1)
        #all units of type 2
        self.refpointF2= self.all_units*self.PROD_unit_1

        ###############################
        # ATENCAO, number of functions#
        ###############################
        return self.refpointF1, self.refpointF2


    #Plot the World in graphical mode
    def PlotWorld(self, gateways=None, units=None, ax=None):
        #matrix = [[1 for x in xrange(self.m)] for x in xrange(self.n)]
        ax = ax if ax is not None else plt.gca()

        #if not max_weight:
        #    max_weight = 2**np.ceil(np.log(np.abs(matrix).max())/np.log(2))

        #declare the background
        ax.patch.set_facecolor('gray')
        ax.set_aspect('equal', 'box')
        ax.xaxis.set_major_locator(plt.NullLocator())
        ax.yaxis.set_major_locator(plt.NullLocator())

        #loop on the World
        for (x,y),w in np.ndenumerate(self.geo):
            #regular square (rectangle)
            #color = 'white' if w > 0 else 'black'
            color = 'white'
            border = 'black'
            size = np.sqrt(np.abs(w))
            rect = plt.Rectangle([x - size / 2, y - size / 2], size, size,
                                 facecolor=color, edgecolor=border)
            ax.add_patch(rect)


        #draw production areas
        for w in self.areas:
            for i in range(w[0], w[0]+w[2]):
                for j in range(w[1], w[1]+w[2]):
                    #print i,j
                    rect = plt.Rectangle([i - size / 2, j - size / 2], size, size,
                                 facecolor='#A8A8A8', edgecolor=border)
                    ax.add_patch(rect)



        ax.autoscale_view()
        ax.invert_yaxis()

        #Font definition to the Plot Title
        # font = {'family' : 'serif',
        #         'color'  : 'black',
        #         'weight' : 'bold',
        #         'size'   : 16,
        #         }
        plt.title('Resource Placement and Assignment')

        #show World
        plt.show()

    #Plot the World in graphical mode with details
    def PlotWorldDetails(self, gateways=None, units=None, cost=None,ax=None):
        #matrix = [[1 for x in xrange(self.m)] for x in xrange(self.n)]
        ax = ax if ax is not None else plt.gca()

        #if not max_weight:
        #    max_weight = 2**np.ceil(np.log(np.abs(matrix).max())/np.log(2))

        #declare the background
        ax.patch.set_facecolor('gray')
        ax.set_aspect('equal', 'box')
        ax.xaxis.set_major_locator(plt.NullLocator())
        ax.yaxis.set_major_locator(plt.NullLocator())

        #loop on the World
        for (x,y),w in np.ndenumerate(self.geo):
            #regular square (rectangle)
            #color = 'white' if w > 0 else 'black'
            color = 'white'
            border = 'black'
            size = np.sqrt(np.abs(w))
            rect = plt.Rectangle([x - size / 2, y - size / 2], size, size,
                                 facecolor=color, edgecolor=border)
            ax.add_patch(rect)

        #draw production areas
        for w in self.areas:
            for i in range(w[0], w[0]+w[2]):
                for j in range(w[1], w[1]+w[2]):
                    #print i,j
                    rect = plt.Rectangle([i - size / 2, j - size / 2], size, size,
                                 facecolor='#A8A8A8', edgecolor=border)
                    ax.add_patch(rect)

        #draw gateways
        for w in gateways:
            rect = plt.Rectangle([w[0] - size / 2, w[1] - size / 2], size, size,
                                 facecolor='black', edgecolor=border)
            ax.add_patch(rect)
            u= units[w[2]-1]
            #line 2 x in the first parameter ... them 2 y on the second
            line = mlines.Line2D((w[0],u[0]) , (w[1],u[1]),color='#AC0000', linestyle='--', lw=2., alpha=.6)
            ax.add_line(line)


        #draw production units
        for w in units:
            if w[2]==0:
                circle = plt.Circle((w[0], w[1]), radius=0.4, fc='b')
            else:
                circle = plt.Circle((w[0], w[1]), radius=0.4, fc='#1BA015')
            ax.add_patch(circle)



        ax.autoscale_view()
        ax.invert_yaxis()

        plt.title('Resource Placement and Assignment    $%s | #%s' % (cost[0], cost[1]))
        #plt.text(10, 10, 'camarao')

        #plt.text(.25, .5, 'kk', horizontalalignment='center', verticalalignment='center', size='x-large')

        #show World
        #plt.show()

        #max window
        #mng = plt.get_current_fig_manager()
        #mng.window.state('zoomed') #works fine on Windows!        plt.show(block=False)

        plt.pause(4.0001)
        plt.clf()


    #build distance matrix with costs = 1
    def GeoDist(self, geo_dist):

        for i in range(0,self.m*self.n):
            for j in range(0,self.m*self.n):
                self.geo_dist[i][j] = abs(j-i)

        return

    #create areas of production
    def CreateArea(self, x, y, len):
        aux=[x, y, len]

        #?????
        #before append, check overlap

        self.areas.append(aux)

    #print Geo elements
    def PrintGeoElement(self,x,y):
        print self.geo[x][y]

    #print GeoDist elements
    def PrintGeoDistElement(self,x,y):
        print self.geo_dist[x][y]

    #print Geo Areas of production
    def PrintGeoAreas(self):
        print self.areas

    #create all the World's gateways
    def CreateGateways(self):
        glist=[]
        for i in self.areas:
            glist.append(self.CreateGate(i))
        return glist

    #create a unique gateway
    def CreateGate(self, the_area):
        x= randint(the_area[0],the_area[0]+the_area[2]-1)
        y= randint(the_area[1],the_area[1]+the_area[2]-1)
        z=-1 #this is the link... it will be done later
        return (x,y,z)

    #check gateways
    def CheckGateways(self, glist):
        if len(glist) != len(self.areas):
            print 'ERROR: different number of gateways and areas.'
            return 0

        #copy of areas
        cp_area = list(self.areas)

        #check each area
        for i in glist:
            #if the gateway is not inside any area: error!
            if self.CheckGate(i,cp_area)==0:
                return 0
        return 1


    #check a specific gateway is inside an area
    def CheckGate(self, gateway, remaining_areas):
        for i in remaining_areas:
            if i[0]<= gateway[0] <i[0]+i[2] and i[1]<= gateway[1] <i[1]+i[2]:
                remaining_areas.remove(i)
                return 1

        #did not find the gateway inside the area
        return 0

    #check a specific gateway is inside an area
    def CheckGate_onepass(self, gateway, remaining_areas):
        for i in remaining_areas:
            if i[0]<= gateway[0] <i[0]+i[2] and i[1]<= gateway[1] <i[1]+i[2]:
                return 1

        #did not find the gateway inside the area
        return 0

    #create all the World's production units
    def CreateUnits(self):
        ulist=[]
        num=randint(1,self.all_units)
        for i in range(0,num):
            ulist.append(self.CreateProdUnit(ulist))
        return ulist

    #create a unique production unit
    def CreateProdUnit(self, ulist):
        while True:
            #choose point and type of unit
            x= randint(0,self.m-1)
            y= randint(0,self.n-1)
            z= randint(0,self.all_units_type-1)

            #check if it is out of the gateway area
            if self.CheckGate_onepass((x,y,z),self.areas)==0 and (x,y,z) not in ulist:
                break

        return (x,y,z)

    #check production units
    def CheckUnits(self, ulist):
        if len(ulist) > self.all_units:
            print 'ERROR: superior number of units in domain.'
            return 0

        #check each unit
        for i in ulist:
            #if the unit is inside any area: error!
            if self.CheckGate_onepass(i,self.areas)!=0:
                return 0
        return 1

    #create links
    def CreateLinks(self, glist, ulist):

        #copy of units and gateways
        cp_unit = list(ulist)
        cp_gate = list(glist)
        r=0

        #for each gateway, link to one unit
        for i in cp_gate:
            #link gateway to unit. The flag "r" indicates when all units were linked, so the free rand choose can happen
            new_gateway = self.CreateLink2Unit(i,cp_unit, ulist,r)
            if len(cp_unit)==0:
                r=1
                cp_unit = list(ulist)

            #rearrange the list
            glist.remove(i)
            glist.append(new_gateway)


    #create specific link
    def CreateLink2Unit(self,gateway,units,read_units,r):

        #choose one of the units
        i=randint(1,len(units))
        #discover its index in the original units list
        j= read_units.index(units[i-1])

        #if there are gateways and units to be linked, remove this from the list
        if r!=1:
            del units[i-1]
        #return tuple
        return (gateway[0], gateway[1],j+1)





    #check links
    def CheckLinks(self,glist,units):

        cp_unit = list(units)

        #check each link inside gateway
        for i in glist:
            #if the gateway is not linked to some unit: error!
            #if self.CheckLink2Unit(i,units,cp_unit)==0:
            if 0< i[2]<=len(units):
                #remove the unit from aux list
                if units[i[2]-1] in cp_unit: cp_unit.remove(units[i[2]-1])
            else:
                return 0
        #if the aux list is not empty... ERROr
        if not cp_unit:
            return 1

        return 0

    #create all structure for a individual
    def CreateFullindividual(self):
        #create gateways
        gateways = self.CreateGateways()

        #create units
        units = self.CreateUnits()

        #create links
        self.CreateLinks(gateways,units)

        man = []

        #create list of individual
        #gateways
        for i in gateways:
            man.append(i[0])
            man.append(i[1])
            man.append(i[2])
        #units
        aux = 0
        for i in units:
            man.append(i[0])
            man.append(i[1])
            man.append(i[2])
            aux=aux+1
        for i in range(aux,self.all_units):
            man.append(-1000)
            man.append(-1000)
            man.append(-1000)

        return man


    #copy all structure for a individual
    def CopyFullindividual(self,gateways, units):


        man = []

        #create list of individual
        #gateways
        for i in gateways:
            man.append(i[0])
            man.append(i[1])
            man.append(i[2])
        #units
        aux = 0
        for i in units:
            man.append(i[0])
            man.append(i[1])
            man.append(i[2])
            aux=aux+1
        for i in range(aux,self.all_units):
            man.append(-1000)
            man.append(-1000)
            man.append(-1000)

        return man


    #get gateways from DNA
    def GetGates(self, dna):
        #gateways
        gates = []
        start = 0
        for i in range(0,self.all_areas):
            stop = start + 3
            #x=dna[start:stop]
            gates.append(tuple(dna[start:stop]))
            start = stop

        return gates

    #get units from DNA
    def GetUnits(self, dna):
        #units
        units = []
        start = self.all_areas*3
        for i in range(self.all_areas,self.all_areas+self.all_units):
            stop = start + 3
            #x=dna[start:stop]
            aux=dna[start:stop]
            if aux[0]>=0 or aux[1]>=0: #just in case I will force this 3 checks
                units.append(tuple(dna[start:stop]))
            start = stop

        return units

            # x= dna[i]
            # y= dna[i+1]
            # l= dna[i+2]
            # i=i+2
            # gates.append(x,y,l)
        #for i in range(0,self.all_areas-1):
        #    gates.append(dna[i:i+2])

    #get units from DNA
    def GetUnits_aftercross(self, dna):
        #units
        units = []
        out_units=[]
        #go to the UNIT position in dna
        start = self.all_areas*3

        #run from the first UNIT position to the end
        for i in range(self.all_areas,self.all_areas+self.all_units):
            #set the block of info
            stop = start + 3
            #x=dna[start:stop]
            aux=dna[start:stop]

            #if I get position X or Y valid, then I will consider this block as a unit that will be fixed later
            if aux[0]>=0 or aux[1]>=0:
                units.append(tuple(dna[start:stop]))
            start = stop

        #I might have units completely invalid between two valids units, so I must remove than and
        #return only the ones to be fixed
        for i in range(0,len(units)):

            #if X or Y are valid
            if units[i][0]>=0 or units[i][1]>=0:
                out_units.append(units[i])


        return out_units


    def CostLinks(self,glist,units):

        #get each link inside gateway
        total_cost=0
        for i in glist:

            #get right position in the world for Matrix Distance
            #dim = self.m * self.n
            #x= (self.n *(i[1]-1)) + (i[0]+1)
            #y= (self.n *(units[i[2]-1][1])) + (units[i[2]-1][0]+1)
            #total_cost = total_cost + self.geo_dist[x][y]
            c=np.sqrt( (units[i[2]-1][0] - i[0])**2 + (units[i[2]-1][1] - i[1])**2 )
            #c=( (units[i[2]-1][0] - i[0])**2 + (units[i[2]-1][1] - i[1])**2 )
            total_cost = total_cost + c


        return (total_cost)

    def PlotLines(self, line_best,line_std,line_avg):

        plt.figure(1)                # the first figure
        plt.subplot(311)             # the first subplot in the first figure
        plt.plot(line_avg)
        plt.ylabel('Average')

        plt.subplot(312)             # the second subplot in the first figure
        plt.plot(line_std)
        plt.ylabel('Standard Deviation')

        plt.subplot(313)             # the second subplot in the first figure
        plt.plot(line_best)
        plt.ylabel('The Best')


        plt.show()


    #organize Gateways again
    def CheckGatewaysFIX(self, glist):
        if len(glist) != len(self.areas):
            print 'ERROR: different number of gateways and areas.'
            return 0

        #copy of areas, all orphan areas will be here
        cp_area  = list(self.areas)
        #gates with problem
        lst_gate = []

        #check each gateway
        j=0
        for i in glist:
            #if the gateway is not inside any area: keep its index!
            if self.CheckGate(i,cp_area)==0:
                lst_gate.append(j)
            j=j+1

        #here: orphan areas and wrong gateways
        j=0
        for i in cp_area:
            glist[lst_gate[j]] = self.CreateGate(i)
            j=j+1

        return 1

     #organize Units again
    def CheckUnitsFIX(self, ulist):
        if len(ulist) > self.all_units:
            print 'ERROR: superior number of units in domain.'
            return 0

        #copy of units, all orphan units will be here
        #cp_units  = list(ulist)
        #units with problem
        lst_units = []


        #check each unit
        j=0
        for i in ulist:
            #check X
            if i[0]<0:
                i = (randint(0,self.m-1),i[1],i[2])
            #check Y
            if i[1]<0:
                i = (i[0], randint(0,self.n-1), i[2])

            #check Type
            ##############################
            # ATENCAO, type of units here#
            ##############################
            if not(0<=i[2]<=1):
                i = (i[0], i[1], randint(0,self.all_units_type-1))

            #if the unit is inside any area of extraction: keep its index!
            if self.CheckGate_onepass(i,self.areas)!=0:
                lst_units.append(j)
            else:
                #update i into ulist... because it might be changed
                ulist[j] = i

            j=j+1

        #here: wrong units (position) ... create new one to replace it
        j=0
        for i in lst_units:
            ulist[lst_units[j]] = self.CreateProdUnit(ulist)
            j=j+1

        return 1


    #organize Links again
    def CheckLinksFIX(self,glist,units, strategy):

        #cp_unit  = list(units)
        #cp_gates = list(glist)
        #list all not linked units
        lst_units=[]
        lst_units_final=[]
        #list the index of all the wrong gates
        lst_gates=[]
        units_dist =[]


        #check each link inside gateway
        j=0
        for i in glist:
            #if the gateway is not linked to some unit: get it on the list!
            if not (1<= i[2]<=len(units)):
                lst_gates.append(j)

            #if the unit is linked to some gateway, get it on the list
            else:
                if i[2] not in lst_units:
                    lst_units.append(i[2])

            j=j+1

        #list all not linked units
        for i in range(0,len(units)):
            if i+1 not in lst_units:
                lst_units_final.append(i+1)
                units_dist.append(units[i])

        #here, there are gates with mistake and non-orphan units in a list
        ##########################################################
        #STRATEGY: ? rand or closer gateways or better production#
        ##########################################################

        if strategy == 1:

            #wrong gates to all orphan units and not orphan
            #length orphan units
            len_unit = len(lst_units_final)
            #length wrong gates
            len_gates = len(lst_gates)
            for i in range(0,len_gates):
                #link gate to units available
                if len_unit > 1:
                    #rand a unit from the available
                    j = randint(0,len_unit-1)
                    glist[lst_gates[i]] = (glist[lst_gates[i]][0],glist[lst_gates[i]][1],lst_units_final[j])
                    del lst_units_final[j]
                    len_unit = len_unit-1
                else:
                    #link to the unique unit available
                    if len_unit == 1:
                        #links to the unique unit
                        glist[lst_gates[i]] = (glist[lst_gates[i]][0],glist[lst_gates[i]][1],lst_units_final[0])
                        del lst_units_final[0]
                        len_unit = len_unit-1
                    else:
                        # rand all units, because all units have been already linked
                        j = randint(1,len(units))
                        glist[lst_gates[i]] = (glist[lst_gates[i]][0],glist[lst_gates[i]][1],j)


            #NO wrong gates and orphan units
            #orphan unit
            len_unit = len(lst_units_final)
            #original lenght of gateways
            len_gates = len(glist)

            #for each orphans
            for i in range(0,len_unit):
                #i is the index of a orphan unit
                # choose duplicated units in gateway list... the first ocurrences
                #c = Counter(elem[2] for elem in glist)
                duplicate=[]
                for k in range(0,len_gates):
                    for v in range(0,len_gates):
                        if v != k:
                            if glist[v][2]==glist[k][2]:
                                duplicate.append(v)
                                if k not in duplicate:
                                    duplicate.append(k)

                    #if a duplicate is found, get out
                    if duplicate:
                        break


                if duplicate:
                    aux = randint(0,len(duplicate)-1) ########
                    glist[duplicate[aux]]= (glist[duplicate[aux]][0],glist[duplicate[aux]][1],lst_units_final[i])
                else:
                    print 'ERROR: no duplication in gateways'
        elif strategy == 2:

            #wrong gates to all orphan units and not orphan
            #length orphan units
            len_unit = len(lst_units_final)
            #length wrong gates
            len_gates = len(lst_gates)
            for i in range(0,len_gates):
                #link gate to units available
                if len_unit > 1:
                    #get the shortest unit from the available
                    j = self.GetClosest(glist[i], units_dist)
                    glist[lst_gates[i]] = (glist[lst_gates[i]][0],glist[lst_gates[i]][1],lst_units_final[j])
                    del lst_units_final[j]
                    del units_dist[j]
                    len_unit = len_unit-1
                else:
                    #link to the unique unit available
                    if len_unit == 1:
                        #links to the unique unit
                        glist[lst_gates[i]] = (glist[lst_gates[i]][0],glist[lst_gates[i]][1],lst_units_final[0])
                        del lst_units_final[0]
                        len_unit = len_unit-1
                    else:
                        # get the shortest unit of all units, because all units have been already linked
                        j = self.GetClosest(glist[lst_gates[i]], units)
                        glist[lst_gates[i]] = (glist[lst_gates[i]][0],glist[lst_gates[i]][1],j+1)


            #NO wrong gates and orphan units
            #orphan unit
            len_unit = len(lst_units_final)
            #original lenght of gateways
            len_gates = len(glist)

            #for each orphans
            for i in range(0,len_unit):
                #i is the index of a orphan unit
                # choose duplicated units in gateway list... all ocurrences
                duplicate=[]
                for k in range(0,len_gates):
                    for v in range(0,len_gates):
                        if v != k:
                            if glist[v][2]==glist[k][2]:
                                if glist[v] not in duplicate:
                                    duplicate.append(glist[v])
                                if glist[k] not in duplicate:
                                    duplicate.append(glist[k])

                    #if a duplicate is found, get out
                    #if duplicate:
                        #break


                if duplicate:
                    #check unit and closest gateway... returns the index of the closest gateway (duplicate[aux])
                    aux=self.GetClosest(units[lst_units_final[i]-1], duplicate)
                    #aux = randint(0,len(duplicate)-1) ########
                    glist[glist.index(duplicate[aux])]= (glist[glist.index(duplicate[aux])][0], glist[glist.index(duplicate[aux])][1], lst_units_final[i])
                else:
                    print 'ERROR: no duplication in gateways'

    #organize Links again
    def FIX_ind(self,inv_individual):

        for k in range(0,len(inv_individual)):

            gates=self.GetGates(inv_individual[k][0])

            #HERE, I can fix in two ways:
            #a) accept one valid X or Y and make all the unit valid
            #b) if X or Y or Type are invalid, I can consider everything invalid
            #
            #this pushes the answers to a scenario with more UNITS or less UNITS
            #
            #I decided to balance it using a rand fomr 0 to 1
            coin = randint(0,1)
            if coin:
                units=self.GetUnits(inv_individual[k][0])
            else:
                units=self.GetUnits_aftercross(inv_individual[k][0])

            self.CheckGatewaysFIX(gates)
            self.CheckUnitsFIX(units)
            self.CheckLinksFIX(gates, units, self.FIX_strategy)

            inv_individual[k][0] = self.CopyFullindividual(gates,units)

        teste=1


    #get the closest
    def GetClosest(self,a, points):

        closestpoint=[]
        dist = 1000000000
        #run all points
        aux=0
        for i in points:
            #if this distance is shorter ... keep it
            tmp = np.sqrt( (i[0] - a[0])**2 + (i[1] - a[1])**2 )
            if tmp < dist:
                #closestpoint = i
                closestpoint = aux
                dist = tmp
            aux = aux + 1


        return closestpoint
        #return points.index(closestpoint)

    #Write a solution to the file.  #0=Pareto; 1=Best Hypercube; 2=Best Cover
    def WriteFileSolution(self,solution, type):

        if type==0:
            with open(self.filePareto, 'w') as file_:
                pickle.dump(solution, file_)
        elif type==1:
            with open(self.fileBestHypercube, 'w') as file_:
                pickle.dump(solution, file_)
        elif type==2:
            with open(self.fileBestCover, 'w') as file_:
                pickle.dump(solution, file_)


    #Read a solution to the file.  #0=Pareto; 1=Best Hypercube; 2=Best Cover
    def GetFileSolution(self, type):

        if type==0:
            with open(self.filePareto, 'r') as file_:
                return pickle.load(file_)
        elif type==1:
            with open(self.fileBestHypercube, 'r') as file_:
                return pickle.load(file_)
        elif type==2:
            with open(self.fileBestCover, 'r') as file_:
                return pickle.load(file_)

    #Print solution front
    def PrintSolutions(self, solution):

        #print all non-dominated solutions WORLD
        plt.clf()
        for i in range(0,len(solution)):
            gates=self.GetGates(solution[i][0])
            units=self.GetUnits(solution[i][0])
            self.PlotWorldDetails(gates,units,solution[i].fitness.values)

    #print population
    def PrintPop(self,pop):

        text_file = open("population.txt", "a")

        for ind in pop:
            text_file.write("%s\n" % str(ind[0]))

        text_file.close()