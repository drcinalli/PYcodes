import random


class Robot:
    #class variables
    description = "This is the Robot for Pairwise in Genetic Algorithm"
    author = "Daniel Cinalli"


    #methods
    def __init__(self):
        #robot decision strategy

        #not used
        self.pro_cost = .5
        self.anti_cost = .5

    #return pro_cost rate
    def PrintStyle(self):
        return self.pro_cost

    #take decision on what objective to look at: 0=first=cost=f1; 1=second=production=f2
    def TakeDecision(self, value, ind1, ind2):

        coin = round(random.uniform(0, 1.0), 2)
        if coin <= value:
            return self.GetBetterCost(ind1,ind2)
        else:
            return self.GetBetterProduction(ind1, ind2)

    #get the better individual on cost
    def GetBetterCost(self, ind1, ind2):
        if ind1[0]<=ind2[0]:
            return 0
        else:
            return 1

    #get the better individual on production
    def GetBetterProduction(self, ind1, ind2):
        if ind1[1]<=ind2[1]:
            return 0
        else:
            return 1






