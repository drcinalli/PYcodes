import random
import numpy as np
from itertools import chain
from operator import attrgetter, itemgetter

from deap import tools
from scipy.spatial import distance
from hv import HyperVolume


class SMSCOIN:
    #class variables

    #methods

    def __init__(self):
        """Constructor."""
        description = "This is the NSGA COIN for MOEA"
        author = "Daniel Cinalli"


    def selTournamentHYPER(self, individuals, k):
        """Tournament selection based on dominance (D) between two individuals, if
        the two individuals do not interdominate the selection is made
        based on crowding distance (CD). The *individuals* sequence length has to
        be a multiple of 4. Starting from the beginning of the selected
        individuals, two consecutive individuals will be different (assuming all
        individuals in the input list are unique). Each individual from the input
        list won't be selected more than twice.

        This selection requires the individuals to have a :attr:`crowding_dist`
        attribute, which can be set by the :func:`assignCrowdingDist` function.

        :param individuals: A list of individuals to select from.
        :param k: The number of individuals to select.
        :returns: A list of selected individuals.
        """
        def tourn(ind1, ind2):
            if ind1.fitness.dominates(ind2.fitness):
                return ind1
            elif ind2.fitness.dominates(ind1.fitness):
                return ind2

            if ind1.fitness.crowding_dist < ind2.fitness.crowding_dist:
                return ind1
            elif ind1.fitness.crowding_dist > ind2.fitness.crowding_dist:
                return ind2

            if random.random() <= 0.5:
                return ind1
            return ind2

        individuals_1 = random.sample(individuals, len(individuals))
        individuals_2 = random.sample(individuals, len(individuals))

        chosen = []
        for i in xrange(0, k, 4):
            chosen.append(tourn(individuals_1[i],   individuals_1[i+1]))
            chosen.append(tourn(individuals_1[i+2], individuals_1[i+3]))
            chosen.append(tourn(individuals_2[i],   individuals_2[i+1]))
            chosen.append(tourn(individuals_2[i+2], individuals_2[i+3]))

        return chosen



    def selSMS(self, individuals, k):
        """Apply NSGA-II
        :param individuals: A list of individuals to select from.
        :param k: The number of individuals to select.
        :param nd: Specify the non-dominated algorithm to use: 'standard' or 'log'.
        :returns: A list of selected individuals.

        .. [Deb2002] Deb, Pratab, Agarwal, and Meyarivan, "A fast elitist
           non-dominated sorting genetic algorithm for multi-objective
           optimization: NSGA-II", 2002.
        """
        # nd='standard'
        # if nd == 'standard':
        #     #print 'ssjsjsjsjsjs'
        #     pareto_fronts = self.sortNondominated(individuals, k)
        # # elif nd == 'log':
        # #     pareto_fronts = sortLogNondominated(individuals, k)
        # else:
        #     raise Exception('selNSGA2: The choice of non-dominated sorting '
        #                     'method "{0}" is invalid.'.format(nd))

        pareto_fronts = self.sortNondominated(individuals, k)

        #som=0
        for front in pareto_fronts:
            self.assignHyperContribution(front) ### baseado no HYPERVOLUME e nao na distancia
            #som += len(front)
        #print som

        chosen = list(chain(*pareto_fronts[:-1]))
        k = k - len(chosen)

        #here I have the last front in my hands... READY

        if k > 0:

            #HERE THE REDUCTION based on Hyper+COIN dist
            #sorted_front = sorted(pareto_fronts[-1], key=attrgetter("fitness.crowding_dist"), reverse=True)
            sorted_front = sorted(pareto_fronts[-1], key=attrgetter("fitness.crowding_dist"))
            chosen.extend(sorted_front[:k])

        #return chosen
        return chosen #, sorted_front[:k],sorted_front[k:]


    def sortNondominated(self, individuals, k, first_front_only=False):
        """Sort the first *k* *individuals* into different nondomination levels
        using the "Fast Nondominated Sorting Approach" proposed by Deb et al.,
        see [Deb2002]_. This algorithm has a time complexity of :math:`O(MN^2)`,
        where :math:`M` is the number of objectives and :math:`N` the number of
        individuals.

        :param individuals: A list of individuals to select from.
        :param k: The number of individuals to select.
        :param first_front_only: If :obj:`True` sort only the first front and
                                 exit.
        :returns: A list of Pareto fronts (lists), the first list includes
                  nondominated individuals.

        .. [Deb2002] Deb, Pratab, Agarwal, and Meyarivan, "A fast elitist
           non-dominated sorting genetic algorithm for multi-objective
           optimization: NSGA-II", 2002.
        """
        from collections import defaultdict


        if k == 0:
            return []

        map_fit_ind = defaultdict(list)
        for ind in individuals:
            map_fit_ind[ind.fitness].append(ind)
        fits = map_fit_ind.keys()

        current_front = []
        next_front = []
        dominating_fits = defaultdict(int)
        dominated_fits = defaultdict(list)

        # Rank first Pareto front
        for i, fit_i in enumerate(fits):
            for fit_j in fits[i+1:]:
                if fit_i.dominates(fit_j):
                    dominating_fits[fit_j] += 1
                    dominated_fits[fit_i].append(fit_j)
                elif fit_j.dominates(fit_i):
                    dominating_fits[fit_i] += 1
                    dominated_fits[fit_j].append(fit_i)
            if dominating_fits[fit_i] == 0:
                current_front.append(fit_i)

        fronts = [[]]
        for fit in current_front:
            fronts[-1].extend(map_fit_ind[fit])
        pareto_sorted = len(fronts[-1])

        # Rank the next front until all individuals are sorted or
        # the given number of individual are sorted.
        if not first_front_only:
            N = min(len(individuals), k)
            while pareto_sorted < N:
                fronts.append([])
                for fit_p in current_front:
                    for fit_d in dominated_fits[fit_p]:
                        dominating_fits[fit_d] -= 1
                        if dominating_fits[fit_d] == 0:
                            next_front.append(fit_d)
                            pareto_sorted += len(map_fit_ind[fit_d])
                            fronts[-1].extend(map_fit_ind[fit_d])
                current_front = next_front
                next_front = []

        return fronts


    #Assign HyperVolume Contribution
    def assignHyperContribution(self, front):
    # Must use wvalues * -1 since hypervolume use implicit minimization
    # And minimization in deap use max on -obj
        wobj = np.array([ind.fitness.wvalues for ind in front]) * -1
        ref = np.max(wobj, axis=0) + 1
        #print ref

        def contribution(i):
            # The contribution of point p_i in point set P
            # is the hypervolume of P without p_i
            #a = self.Hypervolume2D(front , ref)
            if len(ind.fitness.values) == 2:
                b = self.Hypervolume2D(front[:i] +front[i+1:] , ref)
            elif len(ind.fitness.values) == 3:
                b = self.Hypervolume3D(front[:i] +front[i+1:] , ref)

            return  b

        # Parallelization note: Cannot pickle local function
        contrib_values = map(contribution, range(len(front)))

        # Select the maximum hypervolume value (correspond to the minimum difference)
        #return np.argmax(contrib_values)
        for i, h in enumerate(contrib_values):
            front[i].fitness.crowding_dist = h



    #get hypervolume
    def Hypervolume2D(self, front, refpoint):

        #transform front fitness to a list of fitness
        local_fit=[]
        for i in front:
            local_fit.append((i.fitness.values[0],i.fitness.values[1]))


        #evaluate the hypervolume
        hyper=HyperVolume(refpoint)
        aux = hyper.compute(local_fit)
        return aux/(refpoint[0]*refpoint[1])

    #get hypervolume
    def Hypervolume3D(self, front, refpoint):

        #transform front fitness to a list of fitness
        local_fit=[]
        for i in front:
            local_fit.append((i.fitness.values[0],i.fitness.values[1], i.fitness.values[2]))


        #evaluate the hypervolume
        hyper=HyperVolume(refpoint)
        aux = hyper.compute(local_fit)
        return aux/(refpoint[0]*refpoint[1]*refpoint[2])



    def selSMSCOIN(self, individuals, k, world, type_selection='R', exp_type='B', best_gmm=None, kmm=None, ga=None):
        """Apply NSGA-II
        :param individuals: A list of individuals to select from.
        :param k: The number of individuals to select.
        :param nd: Specify the non-dominated algorithm to use: 'standard' or 'log'.
        :returns: A list of selected individuals.

        .. [Deb2002] Deb, Pratab, Agarwal, and Meyarivan, "A fast elitist
           non-dominated sorting genetic algorithm for multi-objective
           optimization: NSGA-II", 2002.
        """
        nd='standard'
        if nd == 'standard':
            #print 'ssjsjsjsjsjs'
            pareto_fronts = self.sortNondominated(individuals, k)
        # elif nd == 'log':
        #     pareto_fronts = sortLogNondominated(individuals, k)
        else:
            raise Exception('selNSGA2: The choice of non-dominated sorting '
                            'method "{0}" is invalid.'.format(nd))

        #som=0
        for front in pareto_fronts:
            self.assignHyperContributionCOIN(front, world) ### baseado no HYPERVOLUME e nao na distancia
            #som += len(front)
        #print som

        chosen = list(chain(*pareto_fronts[:-1]))
        k = k - len(chosen)

        #here I have the last front in my hands... READY

        if k > 0:

            #HERE THE REDUCTION based on Hyper+COIN dist
            #sorted_front = sorted(pareto_fronts[-1], key=attrgetter("fitness.crowding_dist"), reverse=True)
            sorted_front = sorted(pareto_fronts[-1], key=attrgetter("fitness.crowding_dist"))
            chosen.extend(sorted_front[:k])

        #return chosen
        return chosen , sorted_front[:k],sorted_front[k:]

        #Assign HyperVolume Contribution


    def assignHyperContributionCOIN(self, front, world):
    # Must use wvalues * -1 since hypervolume use implicit minimization
    # And minimization in deap use max on -obj
        wobj = np.array([ind.fitness.wvalues for ind in front]) * -1
        ref = np.max(wobj, axis=0) + 1
        #print ref

        def contribution(i):
            # The contribution of point p_i in point set P
            # is the hypervolume of P without p_i
            #a = self.Hypervolume2D(front , ref)
            if len(ind.fitness.values) == 2:
                b = self.Hypervolume2D(front[:i] +front[i+1:] , ref)
            elif len(ind.fitness.values) == 3:
                b = self.Hypervolume3D(front[:i] +front[i+1:] , ref)

            return  b

        # Parallelization note: Cannot pickle local function
        contrib_values = map(contribution, range(len(front)))

        # Select the maximum hypervolume value (correspond to the minimum difference)
        #return np.argmax(contrib_values)
        for i, h in enumerate(contrib_values):
            front[i].fitness.crowding_dist = h





        #'A', '1-D Gaussian Mixture'
        #'B', '2-D Gaussian Mixture'
        #'C', '2-D K-means

        #initialize distance vector
        distances = [0.0] * len(front)
        #all values item and its index)
        fit = [(ind.fitness.values, i) for i, ind in enumerate(front)]

        #if 1-D Gaussian
        if world.experimentTYPE == 'A':

            #set the distances for my 1D gaussian
            self.SetCOINdist_1D( fit, distances, world.mean1, world.mean2)


        #if 2-D Gaussiam
        elif world.experimentTYPE == 'B':

            self.SetCOINdist_2D( fit, distances, world.means, "mahalanobis")

        #if Kmeans
        elif world.experimentTYPE == 'C':

            self.SetCOINdist_2D( fit, distances, world.centroids, "euclidean")

        #if 3-D Gaussiam
        elif world.experimentTYPE == 'D':

            self.SetCOINdist_3D( fit, distances, world.means, "mahalanobis")

        elif world.experimentTYPE == 'E':

            self.SetCOINdist_3D( fit, distances, world.centroids, "euclidean")

        # nobj = len(individuals[0].fitness.values)
        #
        # for i in xrange(nobj):
        #     crowd.sort(key=lambda element: element[0][i])
        #     distances[crowd[0][1]] = float("inf")
        #     distances[crowd[-1][1]] = float("inf")
        #     if crowd[-1][0][i] == crowd[0][0][i]:
        #         continue
        #     norm = nobj * float(crowd[-1][0][i] - crowd[0][0][i])
        #     for prev, cur, next in zip(crowd[:-2], crowd[1:-1], crowd[2:]):
        #         distances[cur[1]] += (next[0][i] - prev[0][i]) / norm

        #here I put the distance back into the pop
        for i, dist in enumerate(distances):
            front[i].fitness.crowding_dist += dist

        #print distances
        #quit()



####TIRAR



    #I copied from DEAP implementation
    def assignCOINdist(self, individuals, world):
        """Assign a COIN distance to each individual's fitness. The
        COIN distance can be retrieve via the :attr:`crowding_dist`
        attribute of each individual's fitness.
        """
        #print "xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
        #print world.experimentTYPE
        #'A', '1-D Gaussian Mixture'
        #'B', '2-D Gaussian Mixture'
        #'C', '2-D K-means

        if len(individuals) == 0:
            return

        #initialize distance vector
        distances = [0.0] * len(individuals)
        #all values item and its index)
        fit = [(ind.fitness.values, i) for i, ind in enumerate(individuals)]
        #crowd = [(ind.fitness.values, i) for i, ind in enumerate(individuals)]
        # print crowd[0]
        # print "  x  "
        # print crowd[0][0]
        # print "  x  "
        # print crowd[0][1]
        # print "  x  "
        # print crowd[0][0][0]
        # print "  x  "
        #print distances
        #if 1-D Gaussian
        if world.experimentTYPE == 'A':

            #set the distances for my 1D gaussian
            self.SetCOINdist_1D( fit, distances, world.mean1, world.mean2)


        #if 2-D Gaussiam
        elif world.experimentTYPE == 'B':

            self.SetCOINdist_2D( fit, distances, world.means, "mahalanobis")

        #if Kmeans
        elif world.experimentTYPE == 'C':

            self.SetCOINdist_2D( fit, distances, world.centroids, "euclidean")

        #if 3-D Gaussiam
        elif world.experimentTYPE == 'D':

            self.SetCOINdist_3D( fit, distances, world.means, "mahalanobis")

        elif world.experimentTYPE == 'E':

            self.SetCOINdist_3D( fit, distances, world.centroids, "euclidean")

        # nobj = len(individuals[0].fitness.values)
        #
        # for i in xrange(nobj):
        #     crowd.sort(key=lambda element: element[0][i])
        #     distances[crowd[0][1]] = float("inf")
        #     distances[crowd[-1][1]] = float("inf")
        #     if crowd[-1][0][i] == crowd[0][0][i]:
        #         continue
        #     norm = nobj * float(crowd[-1][0][i] - crowd[0][0][i])
        #     for prev, cur, next in zip(crowd[:-2], crowd[1:-1], crowd[2:]):
        #         distances[cur[1]] += (next[0][i] - prev[0][i]) / norm

        #here I put the distance back into the pop
        for i, dist in enumerate(distances):
            individuals[i].fitness.crowding_dist = dist

        #print distances
        #quit()

    #set the distances for my 1D gaussian
    def SetCOINdist_1D(self, fit, distances, mean1, mean2):

        #loop population Fitness
        count = 0
        for f in fit:
            # print f
            # print "  x  "
            # print f[0][0]

            #distance 1: mean - cost
            a = abs(mean1 - f[0][0])
            #distance 2: mean - cost
            b = abs(mean2 - f[0][0])

            #the shortest distance to one of the points
            if a <= b:
                distances[count] = a
            else:
                distances[count] = b

            count +=1

    #set the distances for my 2D gaussian
    def SetCOINdist_2D(self, fit, distances, means, dist_type):

        #loop population Fitness
        count = 0

        #calculate covariance
        if dist_type == "mahalanobis":
            x = []
            y = []
            for i in fit:
                x.append(i[0][0])
                y.append(i[0][1])
            covar = np.linalg.pinv(np.cov(np.array(x),np.array(y)))
            #print " zzzzz "

        #print covar
        for f in fit:
            # print f
            # print "  x  "
            # print f[0][0]

            #distances to the mean
            shortest =  float("inf")
            #print means

            for i in means:
                #  1: mean - fitness
                if dist_type == "euclidean":
                    a = distance.euclidean(i,f[0])
                else:

                    # a = distance.euclidean(i,f[0])
                    # b = distance.mahalanobis(i,f[0],covar)
                    # print a
                    # print b
                    a = distance.mahalanobis(i,f[0],covar)
                    #print a


                if a<=shortest:
                    shortest = a
                # print a
                # print i
                # print f[0]
                # quit()
            #print shortest
            #quit()
            #the shortest distance to one of the points
            distances[count] = shortest

            count +=1

    #set the distances for my 3D gaussian
    def SetCOINdist_3D(self, fit, distances, means, dist_type):

        #loop population Fitness
        count = 0

        #calculate covariance
        if dist_type == "mahalanobis":
            x = []
            y = []
            z = []
            for i in fit:
                x.append(i[0][0])
                y.append(i[0][1])
                z.append(i[0][2])
            covar = np.linalg.pinv(np.cov(np.array(x),np.array(y),np.array(z)))
            #print " zzzzz "

        #print covar
        for f in fit:
            # print f
            # print "  x  "
            # print f[0][0]

            #distances to the mean
            shortest =  float("inf")
            #print means

            for i in means:
                #  1: mean - fitness
                #Zorro
                if dist_type == "euclidean":
                    a = distance.euclidean(i,f[0])
                else:

                    a = distance.euclidean(i,f[0])
                    # b = distance.mahalanobis(i,f[0],covar)
                    # print a
                    # print b
                    #a = distance.mahalanobis(i,f[0],covar)
                    #print a


                if a<=shortest:
                    shortest = a
                # print a
                # print i
                # print f[0]
                # quit()
            #print shortest
            #quit()
            #the shortest distance to one of the points
            distances[count] = shortest

            count +=1

    #
    # #I copied from DEAP implementation
    # def selNSGA2COIN(self, individuals, k, world, type_selection='R', exp_type='B', best_gmm=None, kmm=None, ga=None):
    #
    #     #the difference is that I use the COIN distance instead of Crowding Distance
    #
    #     """Apply NSGA-II selection operator on the *individuals*. Usually, the
    #     size of *individuals* will be larger than *k* because any individual
    #     present in *individuals* will appear in the returned list at most once.
    #     Having the size of *individuals* equals to *k* will have no effect other
    #     than sorting the population according to their front rank. The
    #     list returned contains references to the input *individuals*. For more
    #     details on the NSGA-II operator see [Deb2002]_.
    #
    #     :param individuals: A list of individuals to select from.
    #     :param k: The number of individuals to select.
    #     :param nd: Specify the non-dominated algorithm to use: 'standard' or 'log'.
    #     :returns: A list of selected individuals.
    #
    #     .. [Deb2002] Deb, Pratab, Agarwal, and Meyarivan, "A fast elitist
    #        non-dominated sorting genetic algorithm for multi-objective
    #        optimization: NSGA-II", 2002.
    #     """
    #     pareto_fronts = tools.sortNondominated(individuals, k)
    #     #print len(pareto_fronts)
    #
    #
    #     for front in pareto_fronts:
    #         #here is the difference
    #         self.assignCOINdist(front, world)
    #         ###self.assignCrowdingDist_PURE(front)
    #
    #         # herd=[]
    #         # for i in front:
    #         #     if i not in herd:
    #         #         herd.append(i)
    #
    #     #if only one front and Tour type is not Random (it is NSGA-II cluster diversity)
    #     # if len(pareto_fronts)==1:
    #     #     print "aqui"
    #     chosen=[]
    #     if len(pareto_fronts) == 1 and type_selection != 'R': #modified
    #         #quit()
    #         #keep some minimum diversity
    #
    #         #sort the unique front
    #         sorted_front = sorted(pareto_fronts[0], key=attrgetter("fitness.crowding_dist"), reverse=False)
    #
    #         #make fitness list
    #         fitnesses = ga.GetFitness(pareto_fronts[0])
    #         ga.AttachFitness(pareto_fronts[0],fitnesses)
    #         #create list of fitness
    #         front_fit=[]
    #         for i in pareto_fronts[0]:
    #             front_fit.append([i.fitness.values[0],i.fitness.values[1]])
    #
    #
    #         #if Gaussian, get the prediction
    #         if exp_type == 'B':
    #             #
    #             print "gaussian mix"
    #             result_clusters = best_gmm.predict(front_fit)
    #
    #         #if Kmeans
    #         elif exp_type == 'C':
    #             #
    #             print "kmeans"
    #             quit()
    #
    #
    #         #now, I get one list for each cluster... in order of the COIN dist
    #         #
    #         #
    #         #tenho o numero de clusters ja no gmm
    #         final_front=[]
    #         for i in xrange(best_gmm.n_components):
    #             final_front.append([])
    #         for i in xrange(len(front_fit)):
    #             final_front[result_clusters[i]].append(pareto_fronts[0][i])
    #
    #         #here final_front is clustered and ordered by COIN dist
    #
    #         if type_selection == 'C':
    #
    #             #divide between the clusters
    #             count=0
    #             i=0
    #             while i < k:
    #
    #                 #loop between clusters
    #                 for j in xrange(best_gmm.n_components):
    #
    #                     #still a place to fill
    #                     if i<k:
    #
    #                         #insert
    #                         #but check if it is possible
    #                         if len(final_front[j]) > count:
    #                             #insert and increment
    #                             chosen.append(final_front[j][count])
    #                             i += 1
    #
    #                     #else, go away and exit loop
    #                     else:
    #                         break
    #
    #                 count +=1
    #
    #         elif type_selection == 'A':
    #
    #             count = 0
    #             for j in xrange(best_gmm.n_components):
    #
    #                 #insert
    #                 if len(final_front[j]) > 0:
    #                     chosen.append(final_front[j][0])
    #                     count +=1
    #             k = k - count
    #             for i in xrange(k):
    #                 chosen.append(pareto_fronts[0][i])
    #
    #
    #
    #
    #         #chosen.extend(herd[:])
    #         #k = k - len(herd)
    #         #
    #         #
    #         #
    #         #
    #
    #     else:#regular
    #         chosen = list(chain(*pareto_fronts[:-1]))
    #         k = k - len(chosen)
    #         # print "..." +  str(k)
    #
    #         if k > 0:
    #             sorted_front = sorted(pareto_fronts[-1], key=attrgetter("fitness.crowding_dist"), reverse=False)
    #             ###sorted_front = sorted(pareto_fronts[-1], key=attrgetter("fitness.crowding_dist"), reverse=True)
    #             chosen.extend(sorted_front[:k])
    #
    #     return chosen, sorted_front[:k],sorted_front[k:]
    #
    # def assignCrowdingDist_PURE(self, individuals):
    #     """Assign a crowding distance to each individual's fitness. The
    #     crowding distance can be retrieve via the :attr:`crowding_dist`
    #     attribute of each individual's fitness.
    #     """
    #     if len(individuals) == 0:
    #         return
    #
    #     distances = [0.0] * len(individuals)
    #     crowd = [(ind.fitness.values, i) for i, ind in enumerate(individuals)]
    #
    #     nobj = len(individuals[0].fitness.values)
    #
    #     for i in xrange(nobj):
    #         crowd.sort(key=lambda element: element[0][i])
    #         distances[crowd[0][1]] = float("inf")
    #         distances[crowd[-1][1]] = float("inf")
    #         if crowd[-1][0][i] == crowd[0][0][i]:
    #             continue
    #         norm = nobj * float(crowd[-1][0][i] - crowd[0][0][i])
    #         for prev, cur, next in zip(crowd[:-2], crowd[1:-1], crowd[2:]):
    #             distances[cur[1]] += (next[0][i] - prev[0][i]) / norm
    #
    #     for i, dist in enumerate(distances):
    #         individuals[i].fitness.crowding_dist = dist
    #
    #
    # def assignCrowdingDist(self, individuals):
    #     """Assign a crowding distance to each individual's fitness. The
    #     crowding distance can be retrieve via the :attr:`crowding_dist`
    #     attribute of each individual's fitness.
    #     """
    #     if len(individuals) == 0:
    #         return
    #
    #     distances = [0.0] * len(individuals)
    #     crowd = [(ind.fitness.values, i) for i, ind in enumerate(individuals)]
    #
    #     nobj = len(individuals[0].fitness.values)
    #
    #     for i in xrange(nobj):
    #         crowd.sort(key=lambda element: element[0][i])
    #         distances[crowd[0][1]] = float("inf")
    #         distances[crowd[-1][1]] = float("inf")
    #         if crowd[-1][0][i] == crowd[0][0][i]:
    #             continue
    #         norm = nobj * float(crowd[-1][0][i] - crowd[0][0][i])
    #         for prev, cur, next in zip(crowd[:-2], crowd[1:-1], crowd[2:]):
    #             distances[cur[1]] += (next[0][i] - prev[0][i]) / norm
    #
    #     for i, dist in enumerate(distances):
    #         individuals[i].fitness.crowding_dist = dist
    #
    # def selNSGA2PURE(self, individuals, k, nd='standard'):
    #     """Apply NSGA-II selection operator on the *individuals*. Usually, the
    #     size of *individuals* will be larger than *k* because any individual
    #     present in *individuals* will appear in the returned list at most once.
    #     Having the size of *individuals* equals to *k* will have no effect other
    #     than sorting the population according to their front rank. The
    #     list returned contains references to the input *individuals*. For more
    #     details on the NSGA-II operator see [Deb2002]_.
    #
    #     :param individuals: A list of individuals to select from.
    #     :param k: The number of individuals to select.
    #     :param nd: Specify the non-dominated algorithm to use: 'standard' or 'log'.
    #     :returns: A list of selected individuals.
    #
    #     .. [Deb2002] Deb, Pratab, Agarwal, and Meyarivan, "A fast elitist
    #        non-dominated sorting genetic algorithm for multi-objective
    #        optimization: NSGA-II", 2002.
    #     """
    #     nd='standard'
    #     if nd == 'standard':
    #         #print 'ssjsjsjsjsjs'
    #         pareto_fronts = self.sortNondominated(individuals, k)
    #     # elif nd == 'log':
    #     #     pareto_fronts = sortLogNondominated(individuals, k)
    #     else:
    #         raise Exception('selNSGA2: The choice of non-dominated sorting '
    #                         'method "{0}" is invalid.'.format(nd))
    #
    #     for front in pareto_fronts:
    #         self.assignCrowdingDist(front)
    #
    #     chosen = list(chain(*pareto_fronts[:-1]))
    #     k = k - len(chosen)
    #     if k > 0:
    #         sorted_front = sorted(pareto_fronts[-1], key=attrgetter("fitness.crowding_dist"), reverse=True)
    #         chosen.extend(sorted_front[:k])
    #
    #     #return chosen
    #     return chosen, sorted_front[:k],sorted_front[k:]

