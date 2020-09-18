import random
import numpy as np
from itertools import chain
from operator import attrgetter, itemgetter
import math
from copy import deepcopy
from random import randint
from deap import tools
from scipy.spatial import distance



class SPEACOIN:
    #class variables

    #methods

    def __init__(self):
        """Constructor."""
        description = "This is the SPEA COIN for MOEA"
        author = "Daniel Cinalli"


    #I copied from DEAP implementation
    def selSPEA2COIN(self, individuals, k, world, type_selection='R', exp_type='B', best_gmm=None, kmm=None, ga=None, nadir=None):

        #My CHanges here are:

        """Apply SPEA-II selection operator on the *individuals*. Usually, the
        size of *individuals* will be larger than *n* because any individual
        present in *individuals* will appear in the returned list at most once.
        Having the size of *individuals* equals to *n* will have no effect other
        than sorting the population according to a strength Pareto scheme. The
        list returned contains references to the input *individuals*. For more
        details on the SPEA-II operator see [Zitzler2001]_.

        :param individuals: A list of individuals to select from.
        :param k: The number of individuals to select.
        :returns: A list of selected individuals.

        .. [Zitzler2001] Zitzler, Laumanns and Thiele, "SPEA 2: Improving the
           strength Pareto evolutionary algorithm", 2001.
        """
        N = len(individuals)
        L = len(individuals[0].fitness.values)
        K = math.sqrt(N)
        strength_fits = [0] * N
        fits = [0] * N
        dominating_inds = [list() for i in xrange(N)]

        #here calculates the strenght
        for i, ind_i in enumerate(individuals):
            for j, ind_j in enumerate(individuals[i+1:], i+1):
                if ind_i.fitness.dominates(ind_j.fitness):
                    strength_fits[i] += 1
                    dominating_inds[j].append(i)
                elif ind_j.fitness.dominates(ind_i.fitness):
                    strength_fits[j] += 1
                    dominating_inds[i].append(j)

        for i in xrange(N):
            for j in dominating_inds[i]:
                fits[i] += strength_fits[j]
        #######################

        # Choose all non-dominated individuals
        chosen_indices = [i for i in xrange(N) if fits[i] < 1]

        #ok, the  population to be tested!!
        ronda=[]
        for i in chosen_indices:
            ronda.append(individuals[i])
        fitnesses_ronda = ga.GetFitness(ronda)

        kk=[]
        for i, ind_i in enumerate(fits):
            kk.append(individuals[i])
        fitnesses_kkz = ga.GetFitness(kk)
        elements = []
        for i, f in enumerate(fits):
            elements.append((fitnesses_kkz[i],f))
            #print i

        # fits[] has all the fitness (strength) of all the points. INDEX is the same as individuals.
        # COIN_dist has the distance to the closest COIN_ref_point
        #COIN_dist=[float("inf")] * N
        # now I create the INDIVIDUAL CLUSTER array based on each cluster, so one array to each cluster.
        ind_cluster=[]
        #if Gaussian or Kmeans
        if exp_type == 'B' or exp_type == 'D' :
            #
            #print "gaussian mix"
            for i in best_gmm.means_:
                ind_cluster.append([])
            means = world.means
        #if Kmeans
        elif exp_type == 'C' or exp_type == 'E' :
            #
            #print "kmeans"
            for i in kmm.cluster_centers_:
                ind_cluster.append([])
            means = world.centroids
        #print means

        #OK... Strength is READY and separated in the clusters


        # The archive is too small and I need extra ind.
        if len(chosen_indices) < k:     # The archive is too small

            # print " SMALLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLL"

            # for i in xrange(N):
            #     distances = [0.0] * N
            #     for j in xrange(i + 1, N):
            #         dist = 0.0
            #         for l in xrange(L):
            #             val = individuals[i].fitness.values[l] - \
            #                   individuals[j].fitness.values[l]
            #             dist += val * val
            #         distances[j] = dist
            #     kth_dist = self._randomizedSelect(distances, 0, N - 1, K)
            #     density = 1.0 / (kth_dist + 2.0)
            #     fits[i] += density
            #
            # next_indices = [(fits[i], i) for i in xrange(N)
            #                 if not i in chosen_indices]
            # next_indices.sort()
            #
            # kk=[]
            # for i in next_indices:
            #     kk.append(individuals[i[1]])
            # fitnesses_kk2 = ga.GetFitness(kk)

            # #print next_indices
            # chosen_indices += [i for _, i in next_indices[:k - len(chosen_indices)]]

            # tentar pegr na ordem do fits e ver a diferenca


            #OLD HERE, sort asc
            self.CreateIndBYCluster(individuals, ind_cluster, means, fits, nadir)
            for i in ind_cluster:
               i.sort()

            comp=[]
            for ilist in ind_cluster:
                for i in ilist:
                    comp.append(i)
            comp.sort()
            kk=[]
            for i in comp:
                kk.append(individuals[i[1]])
            fitnesses_kk3 = ga.GetFitness(kk)
            # print "z"


            #ver aqui tbem pra descobrir e comparar tbem.




            #calculates the number of extra individuals
            extra = k - len(chosen_indices)
            #extra = k
            #chosen_indices = []

            #while there are spaces to be filled
            while extra:

                for i in comp:


                    #get one if it is not in the choosen already
                    if i[1] in chosen_indices:
                        #remove it from the list
                        del i
                    else:
                        chosen_indices.append(i[1])
                        #remove it from the list
                        del i
                        # remove 1 from extra
                        extra -=1
                    if extra == 0:
                        break

                    # chosen_indices.append(i[1])
                    # #remove it from the list
                    # del i
                    # # remove 1 from extra
                    # extra -=1
                    #
                    # if extra == 0:
                    #     break


                # #OLD insert by clusters
                # #loop clusters
                # for i in ind_cluster:
                #
                #     if extra == 0:
                #         break
                #
                #     if i:
                #         #get one if it is not in the choosen already
                #         if i[0][1] in chosen_indices:
                #             #remove it from the list
                #             del i[0]
                #         else:
                #             chosen_indices.append(i[0][1])
                #             #remove it from the list
                #             del i[0]
                #             # remove 1 from extra
                #             extra -=1



        # The archive is too large, I must cut some of them
        elif len(chosen_indices) > k:
            # print " BIGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGG"
            #create the individual indices by cluster
            #self.CreateIndBYCluster(individuals, ind_cluster, COIN_dist, means, fits, nadir)
            #self.CreateIndBYCluster(individuals, ind_cluster, means, fits, nadir)

            #aqui esta o erro .... ronda nao tem o index ao individuals...
            # ele tem apenas o individuals
            self.CreateIndBYCluster_in_chosen(ronda, ind_cluster, means, fits, nadir, chosen_indices)
            #HERE, sort asc
            for i in ind_cluster:
                i.sort()


            #print " BIGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGG"

            N = len(chosen_indices)

            #OLD PERCENTAGE option
            perc=[]

            #HERE, sort asc
            for i in ind_cluster:
                i.sort()

                #discover percentage
                x = len(i)
                if x == 0:
                    perc.append(0)

                else:
                    #guarantee at least 1
                    y = round(float(x)/len(chosen_indices)*k)
                    if y < 1:
                        y = 1.0
                    perc.append(y)

            #perc_aux = deepcopy(perc)
            #perc_aux.sort()
            z = k

            #get the choosen in percentual
            #because of the round... might be necessary another fix
            for i in perc:
                z -= i

            #here, z is the quantity to be managed
            while z>0:

                for i, j in enumerate(ind_cluster):
                    if len(j) > perc[i]:
                        #vai
                        perc[i] +=1
                        z -= 1
                        if z == 0:
                            break

            while z<0:


                for i, j in enumerate(ind_cluster):

                    if len(j) >= perc[i]:
                        #vai
                        if perc[i] !=1:
                            perc[i] -=1
                            z += 1
                            if z == 0:
                                break


            #JUST checking
            x=0
            for i in perc:
                x += i

            if x != k:
                print "NAOOOOOOOOOO"
                quit()


            new_chosen = []
            #loop clusters
            for i, cluster in enumerate(ind_cluster):

                for j in cluster[:int(perc[i])]:
                    new_chosen.append(j[1])

            chosen_indices = new_chosen

        #### over####




        kk=[]
        for i in chosen_indices:
            kk.append(individuals[i])
        fitnesses_kk = ga.GetFitness(kk)
        fitnesses_kk.sort(key=itemgetter(1), reverse=True)
        x=0


            #
            # new_chosen = []
            #
            # chosen_k=0
            # while chosen_k < k:
            #     #loop clusters
            #     for i, cluster in enumerate(ind_cluster):
            #
            #         if cluster:
            #             new_chosen.append(cluster[0][1])
            #             del cluster[0]
            #             chosen_k +=1
            #             if chosen_k >= k:
            #                 break
            #
            #
            # chosen_indices = new_chosen
            #
            # kk=[]
            # for i in chosen_indices:
            #     kk.append(individuals[i])
            # fitnesses_kk = ga.GetFitness(kk)
            # x=0

            #
            # #OLD method I created
            # #while there are individuals to remove
            # while N > k:
            #
            #     #HERE, sort asc
            #     for i in ind_cluster:
            #         i.sort(reverse=True)
            #
            #     #loop clusters
            #     for i in ind_cluster:
            #
            #         if N <= k:
            #             break
            #
            #         if i:
            #             #remove one if it is in the choosen already
            #             if i[0][1] in chosen_indices:
            #                 #remove it from the list
            #                 chosen_indices.remove(i[0][1])
            #                 del i[0]
            #
            #                 # remove 1 from the total
            #                 N -=1
            #             else:
            #                 #remove it from the list
            #                 del i[0]
            #
            #
            #
            #
            # N = len(chosen_indices)
            # distances = [[0.0] * N for i in xrange(N)]
            # sorted_indices = [[0] * N for i in xrange(N)]
            #
            # #get the SIMETRIC matrix of distances between points (fitness)
            # for i in xrange(N):
            #     for j in xrange(i + 1, N):
            #         dist = 0.0
            #         for l in xrange(L):
            #             val = individuals[chosen_indices[i]].fitness.values[l] - \
            #                   individuals[chosen_indices[j]].fitness.values[l]
            #             dist += val * val
            #         distances[i][j] = dist
            #         distances[j][i] = dist
            #     distances[i][i] = -1
            #
            # # Insert sort is faster than quick sort for short arrays
            # for i in xrange(N):
            #     for j in xrange(1, N):
            #         l = j
            #         while l > 0 and distances[i][j] < distances[i][sorted_indices[i][l - 1]]:
            #             sorted_indices[i][l] = sorted_indices[i][l - 1]
            #             l -= 1
            #         sorted_indices[i][l] = j
            #
            # size = N
            # to_remove = []
            # while size > k:
            #     # Search for minimal distance
            #     min_pos = 0
            #     for i in xrange(1, N):
            #         for j in xrange(1, size):
            #             dist_i_sorted_j = distances[i][sorted_indices[i][j]]
            #             dist_min_sorted_j = distances[min_pos][sorted_indices[min_pos][j]]
            #
            #             if dist_i_sorted_j < dist_min_sorted_j:
            #                 min_pos = i
            #                 break
            #             elif dist_i_sorted_j > dist_min_sorted_j:
            #                 break
            #
            #     #min_pos =
            #
            #
            #     # Remove minimal distance from sorted_indices
            #     for i in xrange(N):
            #         #disable this point in the dist
            #         distances[i][min_pos] = float("inf")
            #         distances[min_pos][i] = float("inf")
            #
            #         #remove dist
            #         for j in xrange(1, size - 1):
            #             if sorted_indices[i][j] == min_pos:
            #                 sorted_indices[i][j] = sorted_indices[i][j + 1]
            #                 sorted_indices[i][j + 1] = min_pos
            #
            #     # Remove corresponding individual from chosen_indices
            #     to_remove.append(min_pos)
            #     size -= 1
            #
            # for index in reversed(sorted(to_remove)):
            #     del chosen_indices[index]

        return [individuals[i] for i in chosen_indices]

    def CreateIndBYCluster(self, individuals, ind_cluster_index,  means, fits, nadir):

        #get fitness
        fit = [ind.fitness.values for i, ind in enumerate(individuals)]

        #loop individuals
        for i in xrange(len(individuals)):




            #inf dist
            shortest =  float("inf")

            #to each cluster point, calculates the shortest distance
            the_c = -1
            count_mean = 0
            for j in means:
                #  1: mean - fitness
                a = distance.euclidean(j,fit[i])

                if a<=shortest:
                    shortest = a
                    the_c = count_mean

                count_mean +=1
                # print a
                # print i
                # print f[0]
                # quit()
            #print shortest
            #quit()
            #SHORTEST is the shortest distance from point i to one of the COIN POINTS and J is the cluster
            #COIN_dist[count] = shortest

            #recalculate SPEA strength
            density = (shortest/ nadir)
            # add density to fits
            fits[i] += density

            #insert the individual index and its strength into the list
            #ind_cluster_index [the_c].append((fits[i],i))
            ind_cluster_index [the_c].append((shortest,i, fit[i]))

    def CreateIndBYCluster_in_chosen(self, individuals, ind_cluster_index,  means, fits, nadir, chosen):

        #get fitness
        fit = [ind.fitness.values for i, ind in enumerate(individuals)]

        #loop individuals
        for i in xrange(len(individuals)):




            #inf dist
            shortest =  float("inf")

            #to each cluster point, calculates the shortest distance
            the_c = -1
            count_mean = 0
            for j in means:
                #  1: mean - fitness
                a = distance.euclidean(j,fit[i])

                if a<=shortest:
                    shortest = a
                    the_c = count_mean

                count_mean +=1
                # print a
                # print i
                # print f[0]
                # quit()
            #print shortest
            #quit()
            #SHORTEST is the shortest distance from point i to one of the COIN POINTS and J is the cluster
            #COIN_dist[count] = shortest

            #recalculate SPEA strength
            #density = (shortest/ nadir)
            # add density to fits
            #fits[i] += density

            #insert the individual index and its strength into the list
            #ind_cluster_index [the_c].append((fits[i],i))
            ind_cluster_index [the_c].append((shortest,chosen[i], fit[i]))


    def _randomizedSelect(self, array, begin, end, i):
        """Allows to select the ith smallest element from array without sorting it.
        Runtime is expected to be O(n).
        """
        if begin == end:
            return array[begin]
        q = self._randomizedPartition(array, begin, end)
        k = q - begin + 1
        if i < k:
            return self._randomizedSelect(array, begin, q, i)
        else:
            return self._randomizedSelect(array, q + 1, end, i - k)

    def _randomizedPartition(self, array, begin, end):
        i = random.randint(begin, end)
        array[begin], array[i] = array[i], array[begin]
        return self._partition(array, begin, end)


    def _partition(self, array, begin, end):
        x = array[begin]
        i = begin - 1
        j = end + 1
        while True:
            j -= 1
            while array[j] > x:
                j -= 1
            i += 1
            while array[i] < x:
                i += 1
            if i < j:
                array[i], array[j] = array[j], array[i]
            else:
                return j


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


   #I copied from DEAP implementation

    #get distance from one of the means
    def GetPointDist(self, point, means):


        #inf dist
        shortest =  float("inf")

        #to each cluster point, calculates the shortest distance
        the_c = -1
        count_mean = 0
        for j in means:
            #  1: mean - fitness
            a = distance.euclidean(j,point)

            if a<=shortest:
                shortest = a
                the_c = count_mean

            count_mean +=1

        return shortest



    def selTournamentCOINd(self, individuals, k , means):
        """Tournament selection based on COIN distance between two individuals, if
        the two individuals do not interdominate the selection is made
        based on COIN distance.

        The *individuals* sequence length has to
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

            #COIN DIstance = is the distance to the means
            # if ind1.fitness.crowding_dist < ind2.fitness.crowding_dist:
            #     return ind1
            # elif ind1.fitness.crowding_dist > ind2.fitness.crowding_dist:
            #     return ind2

            ind1_d = self.GetPointDist(ind1.fitness.values, means)
            ind2_d = self.GetPointDist(ind2.fitness.values, means)
            if ind1_d < ind2_d:
                return ind1
            elif ind1_d > ind2_d:
                return ind2


            if random.random() <= 0.5:
                return ind1
            return ind2

        chosen = []

        y =  len(individuals)
        for i in xrange(k):
            index1 = randint(0,y-1)
            index2 = randint(0,y-1)
            chosen.append(tourn(individuals[index1],   individuals[index2]))


        # y =  len(individuals)
        # individuals_1 = random.sample(individuals, y)
        # individuals_2 = random.sample(individuals, y)

        # for i in xrange(0, k, 4):
        #
        #     index = randint(0,y-1)
        #     chosen.append(tourn(individuals_1[index],   individuals_1[index+1]))
        #     chosen.append(tourn(individuals_1[index+2], individuals_1[index+3]))
        #     chosen.append(tourn(individuals_2[index],   individuals_2[index+1]))
        #     chosen.append(tourn(individuals_2[index+2], individuals_2[index+3]))
        #
            # chosen.append(tourn(individuals_1[i],   individuals_1[i+1]))
            # chosen.append(tourn(individuals_1[i+2], individuals_1[i+3]))
            # chosen.append(tourn(individuals_2[i],   individuals_2[i+1]))
            # chosen.append(tourn(individuals_2[i+2], individuals_2[i+3]))


        return chosen

    def selSPEA2_PURE(self, individuals, k):
        """Apply SPEA-II selection operator on the *individuals*. Usually, the
        size of *individuals* will be larger than *n* because any individual
        present in *individuals* will appear in the returned list at most once.
        Having the size of *individuals* equals to *n* will have no effect other
        than sorting the population according to a strength Pareto scheme. The
        list returned contains references to the input *individuals*. For more
        details on the SPEA-II operator see [Zitzler2001]_.

        :param individuals: A list of individuals to select from.
        :param k: The number of individuals to select.
        :returns: A list of selected individuals.

        .. [Zitzler2001] Zitzler, Laumanns and Thiele, "SPEA 2: Improving the
           strength Pareto evolutionary algorithm", 2001.
        """
        N = len(individuals)
        L = len(individuals[0].fitness.values)
        K = math.sqrt(N)
        strength_fits = [0] * N
        fits = [0] * N
        dominating_inds = [list() for i in xrange(N)]

        for i, ind_i in enumerate(individuals):
            for j, ind_j in enumerate(individuals[i+1:], i+1):
                if ind_i.fitness.dominates(ind_j.fitness):
                    strength_fits[i] += 1
                    dominating_inds[j].append(i)
                elif ind_j.fitness.dominates(ind_i.fitness):
                    strength_fits[j] += 1
                    dominating_inds[i].append(j)

        for i in xrange(N):
            for j in dominating_inds[i]:
                fits[i] += strength_fits[j]

        # Choose all non-dominated individuals
        chosen_indices = [i for i in xrange(N) if fits[i] < 1]


        #ok, the  population to be tested!!
        kk=[]
        fitnesses_kkz=[]
        for i, ind_i in enumerate(fits):
            kk.append(individuals[i])
        for i in kk:
            fitnesses_kkz.append(i.fitness.values)
        elements = []
        for i, f in enumerate(fits):
            elements.append((fitnesses_kkz[i],f))
            #print i
        elements.sort(key=itemgetter(1))

        if len(chosen_indices) < k:     # The archive is too small
            #print " ORIGINAL  SMALLLLLLLLLLLLLLLL"

            for i in xrange(N):
                distances = [0.0] * N
                for j in xrange(i + 1, N):
                    dist = 0.0
                    for l in xrange(L):
                        val = individuals[i].fitness.values[l] - \
                              individuals[j].fitness.values[l]
                        dist += val * val
                    distances[j] = dist
                kth_dist = self._randomizedSelect(distances, 0, N - 1, K)
                density = 1.0 / (kth_dist + 2.0)
                fits[i] += density

            next_indices = [(fits[i], i) for i in xrange(N)
                            if not i in chosen_indices]
            next_indices.sort()
            #print next_indices
            chosen_indices += [i for _, i in next_indices[:k - len(chosen_indices)]]

        elif len(chosen_indices) > k:   # The archive is too large
            #print " ORIGINAL  BIGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGG"
            N = len(chosen_indices)
            distances = [[0.0] * N for i in xrange(N)]
            sorted_indices = [[0] * N for i in xrange(N)]
            for i in xrange(N):
                for j in xrange(i + 1, N):
                    dist = 0.0
                    for l in xrange(L):
                        val = individuals[chosen_indices[i]].fitness.values[l] - \
                              individuals[chosen_indices[j]].fitness.values[l]
                        dist += val * val
                    distances[i][j] = dist
                    distances[j][i] = dist
                distances[i][i] = -1

            # Insert sort is faster than quick sort for short arrays
            for i in xrange(N):
                for j in xrange(1, N):
                    l = j
                    while l > 0 and distances[i][j] < distances[i][sorted_indices[i][l - 1]]:
                        sorted_indices[i][l] = sorted_indices[i][l - 1]
                        l -= 1
                    sorted_indices[i][l] = j

            size = N
            to_remove = []
            while size > k:
                # Search for minimal distance
                min_pos = 0
                for i in xrange(1, N):
                    for j in xrange(1, size):
                        dist_i_sorted_j = distances[i][sorted_indices[i][j]]
                        dist_min_sorted_j = distances[min_pos][sorted_indices[min_pos][j]]

                        if dist_i_sorted_j < dist_min_sorted_j:
                            min_pos = i
                            break
                        elif dist_i_sorted_j > dist_min_sorted_j:
                            break

                # Remove minimal distance from sorted_indices
                for i in xrange(N):
                    distances[i][min_pos] = float("inf")
                    distances[min_pos][i] = float("inf")

                    for j in xrange(1, size - 1):
                        if sorted_indices[i][j] == min_pos:
                            sorted_indices[i][j] = sorted_indices[i][j + 1]
                            sorted_indices[i][j + 1] = min_pos

                # Remove corresponding individual from chosen_indices
                to_remove.append(min_pos)
                size -= 1

            for index in reversed(sorted(to_remove)):
                del chosen_indices[index]

        return [individuals[i] for i in chosen_indices]