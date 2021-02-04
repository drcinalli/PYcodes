from itertools import product
from math import sqrt
from copy import copy
import time
import random

# Randomly choose a Facility
def ChooseFac_rand(item):
    return random.randint(0, len(item)-1)


# choose a Facility with the lowest Shipping Cost
def ChooseFac_lowest_shipping(cp_shipping_cost, client, cp_facilities, facilities):

    aux = {}
    aux_key = ()

    for j in cp_facilities:

        # empty list for the Client
        if not aux:
            aux[(client, j)] = cp_shipping_cost.get((client, facilities.index(j)))
            aux_key = ((client, j))
            # print(aux)
            # print(aux_key)
        #if this FAC is cheaper for this client, replace the item
        elif aux[aux_key] > cp_shipping_cost.get((client, facilities.index(j))):
            # print("....")
            # print (i[0],j[0])
            # print(aux[aux_key])
            # print(shipping_cost.get((i[0],j[0])))
            # print ("  ")
            aux.pop(aux_key)
            aux[(client, j)] = cp_shipping_cost.get((client, facilities.index(j)))
            aux_key = ((client, j))


    return aux_key

# choose a Facility with the lowest Shipping Cost
def ChooseFac_greatest_shipping(cp_shipping_cost, client, cp_facilities, facilities):

    aux = {}
    aux_key = ()

    for j in cp_facilities:

        # empty list for the Client
        if not aux:
            aux[(client, j)] = cp_shipping_cost.get((client, facilities.index(j)))
            aux_key = ((client, j))
            # print(aux)
            # print(aux_key)
        #if this FAC is cheaper for this client, replace the item
        elif not aux[aux_key] > cp_shipping_cost.get((client, facilities.index(j))):
            # print("....")
            # print (i[0],j[0])
            # print(aux[aux_key])
            # print(shipping_cost.get((i[0],j[0])))
            # print ("  ")
            aux.pop(aux_key)
            aux[(client, j)] = cp_shipping_cost.get((client, facilities.index(j)))
            aux_key = ((client, j))


    return aux_key



# choose a Facility with the lowest Shipping Cost
def ChooseFac_lowest_shipping_fcost(cp_shipping_cost, client, cp_facilities, facilities, setup_cost):

    aux = {}
    aux_key = ()

    for j in cp_facilities:

        dedo = facilities.index(j)
        # empty list for the Client
        if not aux:
            aux[(client, j)] = cp_shipping_cost.get((client, dedo)) + setup_cost[dedo]
            aux_key = ((client, j))
            # print(aux)
            # print(aux_key)
        #if this FAC is cheaper for this client, replace the item
        elif aux[aux_key] > cp_shipping_cost.get((client, dedo)) + setup_cost[dedo]:
            # print("....")
            # print (i[0],j[0])
            # print(aux[aux_key])
            # print(shipping_cost.get((i[0],j[0])))
            # print ("  ")
            aux.pop(aux_key)
            aux[(client, j)] = cp_shipping_cost.get((client, dedo)) + setup_cost[dedo]
            aux_key = ((client, j))


    return aux_key




# choose a Facility with the lowest Shipping Cost
def ChooseFac_greatest_shipping_fcost(cp_shipping_cost, client, cp_facilities, facilities, setup_cost):

    aux = {}
    aux_key = ()

    for j in cp_facilities:

        dedo = facilities.index(j)
        # empty list for the Client
        if not aux:
            aux[(client, j)] = cp_shipping_cost.get((client, dedo)) + setup_cost[dedo]
            aux_key = ((client, j))
            # print(aux)
            # print(aux_key)
        #if this FAC is cheaper for this client, replace the item
        elif not aux[aux_key] > cp_shipping_cost.get((client, dedo)) + setup_cost[dedo]:
            # print("....")
            # print (i[0],j[0])
            # print(aux[aux_key])
            # print(shipping_cost.get((i[0],j[0])))
            # print ("  ")
            aux.pop(aux_key)
            aux[(client, j)] = cp_shipping_cost.get((client, dedo)) + setup_cost[dedo]
            aux_key = ((client, j))


    return aux_key



# choose a Facility with the lowest Shipping Cost
def ChooseFac_lowest_production(cp_shipping_cost, client, cp_facilities, facilities, cp_maxp):

    aux = {}
    aux_key = ()

    for j in cp_facilities:

        # empty list for the Client
        dedo = facilities.index(j)
        if cp_maxp[dedo] != 0:
            if not aux:
                aux[(client, j)] = cp_maxp[dedo] #cp_shipping_cost.get((client, facilities.index(j)))
                aux_key = ((client, j))
                # print(aux)
                # print(aux_key)
            #if this FAC is cheaper for this client, replace the item
            elif aux[aux_key] > cp_maxp[dedo]: #cp_shipping_cost.get((client, facilities.index(j))):
                # print("....")
                # print (i[0],j[0])
                # print(aux[aux_key])
                # print(shipping_cost.get((i[0],j[0])))
                # print ("  ")
                aux.pop(aux_key)
                aux[(client, j)] = cp_maxp[dedo] #cp_shipping_cost.get((client, facilities.index(j)))
                aux_key = ((client, j))


    return aux_key


# choose a Facility with the lowest Shipping Cost
def ChooseFac_greatest_production(cp_shipping_cost, client, cp_facilities, facilities, cp_maxp):

    aux = {}
    aux_key = ()

    for j in cp_facilities:

        # empty list for the Client
        dedo = facilities.index(j)
        if cp_maxp[dedo] != 0:
            if not aux:
                aux[(client, j)] = cp_maxp[dedo] #cp_shipping_cost.get((client, facilities.index(j)))
                aux_key = ((client, j))
                # print(aux)
                # print(aux_key)
            #if this FAC is cheaper for this client, replace the item
            elif not aux[aux_key] > cp_maxp[dedo]: #cp_shipping_cost.get((client, facilities.index(j))):
                # print("....")
                # print (i[0],j[0])
                # print(aux[aux_key])
                # print(shipping_cost.get((i[0],j[0])))
                # print ("  ")
                aux.pop(aux_key)
                aux[(client, j)] = cp_maxp[dedo] #cp_shipping_cost.get((client, facilities.index(j)))
                aux_key = ((client, j))


    return aux_key


# Calculate the Total Cost
def CalculateCost(customers, cp_facilities, cp_setup_cost, cost_per_mile, result):
    total_dist = 0
    total_setup = 0
    for client, fac, prods in result:
        #shipping cost of demand
        #total_dist += prods * cost_per_mile * compute_distance(customers[client], fac)
        total_dist += cost_per_mile * compute_distance(customers[client], fac)

        #cost of FAC setup
        if(fac in cp_facilities):
            dedo = cp_facilities.index(fac)
            total_setup += cp_setup_cost[dedo]
            #remove FAC from the list
            del cp_setup_cost[dedo]
            del cp_facilities[dedo]


    return total_dist+total_setup

def PrintResult(result, facs):

    #length Facs
    lenFacs = len(facs)

    #print clients and facs disposition
    for i in result:
        #print("Client " + str(i[0] + 1) + "receives part of its demand  from Warehouse" + str(i[1] + 1) + ".")
        print("Client " + str(i[0]+1) + " receives part of its demand  from Warehouse %s." % (i[1],))

        if (i[1] in facs):
            dedo= facs.index(i[1])
            del facs[dedo]

    lenFacs = lenFacs - len(facs)
    print("Number of Warehouses: " + str(lenFacs))



# Euclidean distance between a facility and customer sites
def compute_distance(loc1, loc2):
    dx = loc1[0] - loc2[0]
    dy = loc1[1] - loc2[1]
    return sqrt(dx*dx + dy*dy)


##### Sets and Indices #####
customers = [(0,1.5), (2.5,1.2)]
facilities = [(0,0), (0,1), (0,2), (1,0), (1,1), (1,2), (2,0), (2,1), (2,2)]
num_facilities = len(facilities)
num_customers = len(customers)

##### Parameters #####
#cost per mile
cost_per_mile = 1
#fixed cost
setup_cost = [3,2,3,1,3,3,4,3,2]
#demand of customer
dc = [600,700]
#max production
maxp= [500,500,500,500,500,500,500,500,500]
cartesian_prod = list(product(range(num_customers), range(num_facilities)))
# shipping costs
shipping_cost = {(c,f): cost_per_mile*compute_distance(customers[c], facilities[f]) for c, f in cartesian_prod}
shipping_demand={}
for k, v in shipping_cost.items():
    shipping_demand[k] = v * dc[k[0]]

result = []
#i = 0

start = time.time()


#copies of facilities, demands and production
#to calculate temp values
cp_facilities = copy(facilities)
cp_maxp= copy(maxp)
cp_dc = copy(dc)

##########
# Random #
##########

# choose the Facility for each customer
for idx,j in enumerate(customers):

    #while still demand to be covered
    while cp_dc[idx] > 0:
        fac = ChooseFac_rand(cp_facilities)

        #update TOTAL Facility Prod and Customer demand
        if(cp_dc[idx] - cp_maxp[fac]<=0): #facility produces more than needed
            cp_maxp[fac] -= cp_dc[idx]
            # append the result: Client | FAC | total of products
            result.append((idx, cp_facilities[fac], cp_dc[idx]))
            cp_dc[idx] = 0
        else:
            cp_dc[idx]  -= cp_maxp[fac]
            # append the result: Client | FAC | total of products
            result.append((idx, cp_facilities[fac], cp_maxp[fac]))
            cp_maxp[fac] = 0

            #remove facility and its production item
            del cp_facilities[fac]
            del cp_maxp[fac]


# calculate the setup_cost
cp_facilities = copy(facilities)
cp_setup_cost = copy(setup_cost)
total = CalculateCost(customers, cp_facilities, cp_setup_cost, cost_per_mile,  result)
print(total)

end = time.time()
print("TIME IS: ", end - start)

cp_facilities = copy(facilities)
PrintResult(result, cp_facilities)

###############################################
# Lowest Shipping Cost (per client) Heuristic #
###############################################

start = time.time()

#copies of facilities, demands and production
#to calculate temp values
result = []
cp_facilities = copy(facilities)
cp_maxp= copy(maxp)
cp_dc = copy(dc)
cp_setup_cost = copy(setup_cost)
cp_shipping_cost = copy(shipping_cost)

for idx,j in enumerate(customers):

    #while still demand to be covered
    while cp_dc[idx] > 0:
        fac = ChooseFac_lowest_shipping(cp_shipping_cost, idx, cp_facilities, facilities)
        #get index number of fac
        faci = facilities.index(fac[1])


        #update TOTAL Facility Prod and Customer demand
        if(cp_dc[idx] - cp_maxp[faci]<=0): #facility produces more than needed
            cp_maxp[faci] -= cp_dc[idx]
            # append the result: Client | FAC | total of products
            result.append((idx, facilities[faci], cp_dc[idx]))
            cp_dc[idx] = 0
        else:
            cp_dc[idx]  -= cp_maxp[faci]
            # append the result: Client | FAC | total of products
            result.append((idx, facilities[faci], cp_maxp[faci]))
            cp_maxp[faci] = 0

            #remove facility and its production item
            faci_cp = cp_facilities.index(fac[1])
            del cp_facilities[faci_cp]
            #del cp_maxp[faci]
            del cp_shipping_cost[(idx,faci)]

    #redo cp_facs
    cp_facilities = copy(facilities)


# calculate the setup_cost
#cp_facilities = copy(facilities)
#cp_setup_cost = copy(setup_cost)
total = CalculateCost(customers, cp_facilities, cp_setup_cost, cost_per_mile,  result)
print(total)

end = time.time()
print("TIME IS: ", end - start)

cp_facilities = copy(facilities)
PrintResult(result, cp_facilities)

#################################################
# Greatest Shipping Cost (per client) Heuristic #
#################################################

start = time.time()

#copies of facilities, demands and production
#to calculate temp values
result = []
cp_facilities = copy(facilities)
cp_maxp= copy(maxp)
cp_dc = copy(dc)
cp_setup_cost = copy(setup_cost)
cp_shipping_cost = copy(shipping_cost)

for idx,j in enumerate(customers):

    #while still demand to be covered
    while cp_dc[idx] > 0:
        fac = ChooseFac_greatest_shipping(cp_shipping_cost, idx, cp_facilities, facilities)
        #get index number of fac
        faci = facilities.index(fac[1])


        #update TOTAL Facility Prod and Customer demand
        if(cp_dc[idx] - cp_maxp[faci]<=0): #facility produces more than needed
            cp_maxp[faci] -= cp_dc[idx]
            # append the result: Client | FAC | total of products
            result.append((idx, facilities[faci], cp_dc[idx]))
            cp_dc[idx] = 0
        else:
            cp_dc[idx]  -= cp_maxp[faci]
            # append the result: Client | FAC | total of products
            result.append((idx, facilities[faci], cp_maxp[faci]))
            cp_maxp[faci] = 0

            #remove facility and its production item
            faci_cp = cp_facilities.index(fac[1])
            del cp_facilities[faci_cp]
            #del cp_maxp[faci]
            del cp_shipping_cost[(idx,faci)]

    #redo cp_facs
    cp_facilities = copy(facilities)


# calculate the setup_cost
#cp_facilities = copy(facilities)
#cp_setup_cost = copy(setup_cost)
total = CalculateCost(customers, cp_facilities, cp_setup_cost, cost_per_mile,  result)
print(total)

end = time.time()
print("TIME IS: ", end - start)

cp_facilities = copy(facilities)
PrintResult(result, cp_facilities)


###############################################################
# Lowest Shipping Cost and Fixed Costs (per client) Heuristic #
###############################################################

start = time.time()

#copies of facilities, demands and production
#to calculate temp values
result = []
cp_facilities = copy(facilities)
cp_maxp= copy(maxp)
cp_dc = copy(dc)
cp_setup_cost = copy(setup_cost)
cp_shipping_cost = copy(shipping_cost)

for idx,j in enumerate(customers):

    #while still demand to be covered
    while cp_dc[idx] > 0:
        fac = ChooseFac_lowest_shipping_fcost(cp_shipping_cost, idx, cp_facilities, facilities, setup_cost)
        #get index number of fac
        faci = facilities.index(fac[1])


        #update TOTAL Facility Prod and Customer demand
        if(cp_dc[idx] - cp_maxp[faci]<=0): #facility produces more than needed
            cp_maxp[faci] -= cp_dc[idx]
            # append the result: Client | FAC | total of products
            result.append((idx, facilities[faci], cp_dc[idx]))
            cp_dc[idx] = 0
        else:
            cp_dc[idx]  -= cp_maxp[faci]
            # append the result: Client | FAC | total of products
            result.append((idx, facilities[faci], cp_maxp[faci]))
            cp_maxp[faci] = 0

            #remove facility and its production item
            faci_cp = cp_facilities.index(fac[1])
            del cp_facilities[faci_cp]
            #del cp_maxp[faci]
            del cp_shipping_cost[(idx,faci)]

    #redo cp_facs
    cp_facilities = copy(facilities)


# calculate the setup_cost
#cp_facilities = copy(facilities)
#cp_setup_cost = copy(setup_cost)
total = CalculateCost(customers, cp_facilities, cp_setup_cost, cost_per_mile,  result)
print(total)

end = time.time()
print("TIME IS: ", end - start)

cp_facilities = copy(facilities)
PrintResult(result, cp_facilities)

#################################################################
# Greatest Shipping Cost and Fixed Costs (per client) Heuristic #
#################################################################

start = time.time()

#copies of facilities, demands and production
#to calculate temp values
result = []
cp_facilities = copy(facilities)
cp_maxp= copy(maxp)
cp_dc = copy(dc)
cp_setup_cost = copy(setup_cost)
cp_shipping_cost = copy(shipping_cost)

for idx,j in enumerate(customers):

    #while still demand to be covered
    while cp_dc[idx] > 0:
        fac = ChooseFac_greatest_shipping_fcost(cp_shipping_cost, idx, cp_facilities, facilities, setup_cost)
        #get index number of fac
        faci = facilities.index(fac[1])


        #update TOTAL Facility Prod and Customer demand
        if(cp_dc[idx] - cp_maxp[faci]<=0): #facility produces more than needed
            cp_maxp[faci] -= cp_dc[idx]
            # append the result: Client | FAC | total of products
            result.append((idx, facilities[faci], cp_dc[idx]))
            cp_dc[idx] = 0
        else:
            cp_dc[idx]  -= cp_maxp[faci]
            # append the result: Client | FAC | total of products
            result.append((idx, facilities[faci], cp_maxp[faci]))
            cp_maxp[faci] = 0

            #remove facility and its production item
            faci_cp = cp_facilities.index(fac[1])
            del cp_facilities[faci_cp]
            #del cp_maxp[faci]
            del cp_shipping_cost[(idx,faci)]

    #redo cp_facs
    cp_facilities = copy(facilities)


# calculate the setup_cost
#cp_facilities = copy(facilities)
#cp_setup_cost = copy(setup_cost)
total = CalculateCost(customers, cp_facilities, cp_setup_cost, cost_per_mile,  result)
print(total)

end = time.time()
print("TIME IS: ", end - start)

cp_facilities = copy(facilities)
PrintResult(result, cp_facilities)



###############################################
# Lowest Production (per Warehouse) Heuristic #
###############################################

start = time.time()

#copies of facilities, demands and production
#to calculate temp values
result = []
cp_facilities = copy(facilities)
cp_maxp= copy(maxp)
cp_dc = copy(dc)
cp_setup_cost = copy(setup_cost)
cp_shipping_cost = copy(shipping_cost)

for idx,j in enumerate(customers):

    #while still demand to be covered
    while cp_dc[idx] > 0:
        fac = ChooseFac_lowest_production(cp_shipping_cost, idx, cp_facilities, facilities, cp_maxp)
        #get index number of fac
        faci = facilities.index(fac[1])


        #update TOTAL Facility Prod and Customer demand
        if(cp_dc[idx] - cp_maxp[faci]<=0): #facility produces more than needed
            cp_maxp[faci] -= cp_dc[idx]
            # append the result: Client | FAC | total of products
            result.append((idx, facilities[faci], cp_dc[idx]))
            cp_dc[idx] = 0
        else:
            cp_dc[idx]  -= cp_maxp[faci]
            # append the result: Client | FAC | total of products
            result.append((idx, facilities[faci], cp_maxp[faci]))
            cp_maxp[faci] = 0

            #remove facility and its production item
            faci_cp = cp_facilities.index(fac[1])
            del cp_facilities[faci_cp]
            #del cp_maxp[faci]
            del cp_shipping_cost[(idx,faci)]

    #redo cp_facs
    cp_facilities = copy(facilities)


# calculate the setup_cost
#cp_facilities = copy(facilities)
#cp_setup_cost = copy(setup_cost)
total = CalculateCost(customers, cp_facilities, cp_setup_cost, cost_per_mile,  result)
print(total)

end = time.time()
print("TIME IS: ", end - start)

cp_facilities = copy(facilities)
PrintResult(result, cp_facilities)



#################################################
# Greatest Production (per Warehouse) Heuristic #
#################################################

start = time.time()

#copies of facilities, demands and production
#to calculate temp values
result = []
cp_facilities = copy(facilities)
cp_maxp= copy(maxp)
cp_dc = copy(dc)
cp_setup_cost = copy(setup_cost)
cp_shipping_cost = copy(shipping_cost)

for idx,j in enumerate(customers):

    #while still demand to be covered
    while cp_dc[idx] > 0:
        fac = ChooseFac_greatest_production(cp_shipping_cost, idx, cp_facilities, facilities, cp_maxp)
        #get index number of fac
        faci = facilities.index(fac[1])


        #update TOTAL Facility Prod and Customer demand
        if(cp_dc[idx] - cp_maxp[faci]<=0): #facility produces more than needed
            cp_maxp[faci] -= cp_dc[idx]
            # append the result: Client | FAC | total of products
            result.append((idx, facilities[faci], cp_dc[idx]))
            cp_dc[idx] = 0
        else:
            cp_dc[idx]  -= cp_maxp[faci]
            # append the result: Client | FAC | total of products
            result.append((idx, facilities[faci], cp_maxp[faci]))
            cp_maxp[faci] = 0

            #remove facility and its production item
            faci_cp = cp_facilities.index(fac[1])
            del cp_facilities[faci_cp]
            #del cp_maxp[faci]
            del cp_shipping_cost[(idx,faci)]

    #redo cp_facs
    cp_facilities = copy(facilities)


# calculate the setup_cost
#cp_facilities = copy(facilities)
#cp_setup_cost = copy(setup_cost)
total = CalculateCost(customers, cp_facilities, cp_setup_cost, cost_per_mile,  result)
print(total)

end = time.time()
print("TIME IS: ", end - start)

cp_facilities = copy(facilities)
PrintResult(result, cp_facilities)
