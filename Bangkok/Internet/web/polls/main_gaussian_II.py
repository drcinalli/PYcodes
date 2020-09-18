import math, random, copy
import numpy as np
import sys

import matplotlib.pyplot as plt
import matplotlib.mlab as mlab

def expectation_maximization(t, nbclusters=2, nbiter=1, normalize=False,\
        epsilon=0.1, monotony=False, datasetinit=True):
    """
    Each row of t is an observation, each column is a feature
    'nbclusters' is the number of seeds and so of clusters
    'nbiter' is the number of iterations
    'epsilon' is the convergence bound/criterium

    """
    def pnorm(x, m, s):
        """
        Compute the multivariate normal distribution with values vector x,
        mean vector m, sigma (variances/covariances) matrix s
        """
        xmt = np.matrix(x-m).transpose()
        for i in xrange(len(s)):
            if s[i,i] <= sys.float_info[3]: # min float
                s[i,i] = sys.float_info[3]
        sinv = np.linalg.inv(s)
        xm = np.matrix(x-m)
        return (2.0*math.pi)**(-len(x)/2.0)*(1.0/math.sqrt(np.linalg.det(s)))\
                *math.exp(-0.5*(xm*sinv*xmt))

    def draw_params():
            if datasetinit:
                tmpmu = np.array([1.0*t[random.uniform(0,nbobs),:]],np.float64)
            else:
                tmpmu = np.array([random.uniform(min_max[f][0], min_max[f][1])\
                        for f in xrange(nbfeatures)], np.float64)
            return {'mu': tmpmu,\
                    'sigma': np.matrix(np.diag(\
                    [(min_max[f][1]-min_max[f][0])/2.0\
                    for f in xrange(nbfeatures)])),\
                    'proba': 1.0/nbclusters}

    nbobs = t.shape[0]
    nbfeatures = t.shape[1]
    min_max = []
    # find xranges for each features
    for f in xrange(nbfeatures):
        min_max.append((t[:,f].min(), t[:,f].max()))

    ### Normalization
    if normalize:
        for f in xrange(nbfeatures):
            t[:,f] -= min_max[f][0]
            t[:,f] /= (min_max[f][1]-min_max[f][0])
    min_max = []
    for f in xrange(nbfeatures):
        min_max.append((t[:,f].min(), t[:,f].max()))
    ### /Normalization

    result = {}
    quality = 0.0 # sum of the means of the distances to centroids
    random.seed()
    Pclust = np.ndarray([nbobs,nbclusters], np.float64) # P(clust|obs)
    Px = np.ndarray([nbobs,nbclusters], np.float64) # P(obs|clust)
    # iterate nbiter times searching for the best "quality" clustering
    for iteration in xrange(nbiter):
        ##############################################
        # Step 1: draw nbclusters sets of parameters #
        ##############################################
        params = [draw_params() for c in xrange(nbclusters)]
        old_log_estimate = sys.maxsize         # init, not true/real
        log_estimate = sys.maxint/2 + epsilon # init, not true/real
        estimation_round = 0
        # Iterate until convergence (EM is monotone) <=> < epsilon variation
        while (abs(log_estimate - old_log_estimate) > epsilon\
                and (not monotony or log_estimate < old_log_estimate)):
            restart = False
            old_log_estimate = log_estimate
            ########################################################
            # Step 2: compute P(Cluster|obs) for each observations #
            ########################################################
            for o in xrange(nbobs):
                for c in xrange(nbclusters):
                    # Px[o,c] = P(x|c)
                    Px[o,c] = pnorm(t[o,:],\
                            params[c]['mu'], params[c]['sigma'])
            #for o in xrange(nbobs):
            #    Px[o,:] /= math.fsum(Px[o,:])
            for o in xrange(nbobs):
                for c in xrange(nbclusters):
                    # Pclust[o,c] = P(c|x)
                    Pclust[o,c] = Px[o,c]*params[c]['proba']
            #    assert math.fsum(Px[o,:]) >= 0.99 and\
            #            math.fsum(Px[o,:]) <= 1.01
            for o in xrange(nbobs):
                tmpSum = 0.0
                for c in xrange(nbclusters):
                    tmpSum += params[c]['proba']*Px[o,c]
                Pclust[o,:] /= tmpSum
                #assert math.fsum(Pclust[:,c]) >= 0.99 and\
                #        math.fsum(Pclust[:,c]) <= 1.01
            ###########################################################
            # Step 3: update the parameters (sets {mu, sigma, proba}) #
            ###########################################################
            print "iter:", iteration, " estimation#:", estimation_round,\
                    " params:", params
            for c in xrange(nbclusters):
                tmpSum = math.fsum(Pclust[:,c])
                params[c]['proba'] = tmpSum/nbobs
                if params[c]['proba'] <= 1.0/nbobs:           # restart if all
                    restart = True                             # converges to
                    print "Restarting, p:",params[c]['proba'] # one cluster
                    break
                m = np.zeros(nbfeatures, np.float64)
                for o in xrange(nbobs):
                    m += t[o,:]*Pclust[o,c]
                params[c]['mu'] = m/tmpSum
                s = np.matrix(np.diag(np.zeros(nbfeatures, np.float64)))
                for o in xrange(nbobs):
                    s += Pclust[o,c]*(np.matrix(t[o,:]-params[c]['mu']).transpose()*\
                            np.matrix(t[o,:]-params[c]['mu']))
                    #print ">>>> ", t[o,:]-params[c]['mu']
                    #diag = Pclust[o,c]*((t[o,:]-params[c]['mu'])*\
                    #        (t[o,:]-params[c]['mu']).transpose())
                    #print ">>> ", diag
                    #for i in xrange(len(s)) :
                    #    s[i,i] += diag[i]
                params[c]['sigma'] = s/tmpSum
                print "------------------"
                print params[c]['sigma']

            ### Test bound conditions and restart consequently if needed
            if not restart:
                restart = True
                for c in xrange(1,nbclusters):
                    if not np.allclose(params[c]['mu'], params[c-1]['mu'])\
                    or not np.allclose(params[c]['sigma'], params[c-1]['sigma']):
                        restart = False
                        break
            if restart:                # restart if all converges to only
                old_log_estimate = sys.maxint          # init, not true/real
                log_estimate = sys.maxint/2 + epsilon # init, not true/real
                params = [draw_params() for c in xrange(nbclusters)]
                continue
            ### /Test bound conditions and restart

            ####################################
            # Step 4: compute the log estimate #
            ####################################
            log_estimate = math.fsum([math.log(math.fsum(\
                    [Px[o,c]*params[c]['proba'] for c in xrange(nbclusters)]))\
                    for o in xrange(nbobs)])
            print "(EM) old and new log estimate: ",\
                    old_log_estimate, log_estimate
            estimation_round += 1

        # Pick/save the best clustering as the final result
        quality = -log_estimate
        if not quality in result or quality > result['quality']:
            result['quality'] = quality
            result['params'] = copy.deepcopy(params)
            result['clusters'] = [[o for o in xrange(nbobs)\
                    if Px[o,c] == max(Px[o,:])]\
                    for c in xrange(nbclusters)]
    return result
#
# t =  np.random.randint(0, 100, (1000, 1))
# print t
# result = expectation_maximization(t, epsilon=1)
#
# k1_mu = result['params'][0]['mu']
# k1_p = result['params'][0]['proba']
# k1_sigma = result['params'][0]['sigma']
#
# k2_mu = result['params'][1]['mu']
# k2_p = result['params'][1]['proba']
# k2_sigma = result['params'][1]['sigma']
#
# mean = float(k1_mu)
# variance = float(k1_sigma)
# sigma = np.sqrt(variance)
# x = np.linspace(-50,150,100)
# plt.plot(x,mlab.normpdf(x,mean,sigma))
#
# mean = float(k2_mu)
# variance = float(k2_sigma)
# sigma = np.sqrt(variance)
# x = np.linspace(-50,150,100)
# plt.plot(x,mlab.normpdf(x,mean,sigma))
#
# plt.show()
#

#fake fitness
fakefit= [(73.868159279972645, -23.0), (77.431658819361502, -35.0), (78.923887369973485, -46.0), (78.923887369973485, -46.0),
          (82.901600471735435, -58.0), (83.053405777305727, -69.0), (85.9976776873049, -70.0), (87.053405777305727, -81.0),
          (91.053405777305727, -93.0), (95.053405777305727, -105.0), (99.064495102245985, -116.0), (103.06449510224598, -128.0),
          (107.06449510224598, -140.0), (114.41421356237309, -151.0), (114.41421356237309, -151.0), (118.41421356237309, -163.0),
          (118.41421356237309, -163.0), (118.41421356237309, -163.0), (118.41421356237309, -163.0), (122.41421356237309, -175.0),
          (130.82842712474618, -186.0), (130.82842712474618, -186.0), (134.82842712474618, -198.0), (134.82842712474618, -198.0),
          (138.82842712474618, -210.0), (138.82842712474618, -210.0)]


#get robots .85
fake = [[74, 15], [74, 11], [69, 16], [73, 21], [68, 28], [60, 27], [69, 33], [65, 27], [50, 40], [45, 35], [66, 46], [59, 46], [39, 41], [49, 55], [44, 52], [44, 39], [43, 64], [30, 45], [32, 63], [31, 63], [23, 69], [27, 69], [15, 76], [12, 77], [23, 61], [16, 81]]

fake_array = []
count=0
for i in fake:
    #how many votes on i
    for j in xrange(i[0]):
        fake_array.append([fakefit[count][0]])
    count += 1

t = np.array(fake_array, np.int32)
#t.reshape(len(t),2).shape
result = expectation_maximization(t, nbiter=3, epsilon=0.1)

k1_mu = result['params'][0]['mu']
k1_p = result['params'][0]['proba']
k1_sigma = result['params'][0]['sigma']

k2_mu = result['params'][1]['mu']
k2_p = result['params'][1]['proba']
k2_sigma = result['params'][1]['sigma']

mean1 = float(k1_mu)
variance = float(k1_sigma)
sigma = np.sqrt(variance)
x = np.linspace(50,180,100)

plt.subplot(2, 1, 1)
plt.plot(x,mlab.normpdf(x,mean1,sigma),'-o',color='b')

mean2 = float(k2_mu)
variance = float(k2_sigma)
sigma = np.sqrt(variance)
x = np.linspace(50,180,100)

plt.plot(x,mlab.normpdf(x,mean2,sigma),'-o', color='g')

if mean1<mean2:
    plt.hist(t[:len(result['clusters'][0])],histtype='stepfilled', bins=20, normed=True,color='#325ADB', label='Uniform')
    plt.hist(t[len(result['clusters'][0]):],histtype='stepfilled', bins=20, normed=True,color='#2FBA4F', label='Uniform')
else:
    plt.hist(t[:len(result['clusters'][0])], histtype='stepfilled',bins=20, normed=True,color='#2FBA4F', label='Uniform')
    plt.hist(t[len(result['clusters'][0]):], histtype='stepfilled',bins=20, normed=True,color='#325ADB', label='Uniform')


plt.title("PeloCano - Gaussian Mixture")
plt.xlabel("Value")
plt.ylabel("Frequency")
#plt.show()

mean_full=np.mean(t)
variance_full = np.std(t)
sigma_full = np.sqrt(variance_full)

plt.subplot(2, 1, 2)
plt.plot(x,mlab.normpdf(x,mean_full,sigma_full),'-o',color='r')
plt.plot(x,mlab.normpdf(x,mean2,sigma),'-o', color='g')
plt.plot(x,mlab.normpdf(x,mean1,sigma),'-o',color='b')



plt.title("PeloCano - Gaussian Mixture")
plt.xlabel("Value")
plt.ylabel("Frequency")
plt.show()


#####################################################

#get robots .95
fake = [[96, 3], [69, 10], [85, 13], [82, 7], [88, 18], [75, 21], [68, 28], [76, 25], [50, 40], [59, 37], [46, 33], [43, 37], [44, 39], [44, 49], [37, 60], [48, 58], [36, 72], [37, 50], [29, 61], [18, 81], [27, 74], [9, 81], [10, 77], [10, 73], [10, 77], [4, 76]]

fake_array = []
count=0
for i in fake:
    #how many votes on i
    for j in xrange(i[0]):
        fake_array.append([fakefit[count][0]])
    count += 1


t = np.array(fake_array, np.int32)
#t.reshape(len(t),2).shape
result = expectation_maximization(t, nbiter=3, epsilon=0.1)

k1_mu = result['params'][0]['mu']
k1_p = result['params'][0]['proba']
k1_sigma = result['params'][0]['sigma']

k2_mu = result['params'][1]['mu']
k2_p = result['params'][1]['proba']
k2_sigma = result['params'][1]['sigma']

mean1 = float(k1_mu)
variance = float(k1_sigma)
sigma = np.sqrt(variance)
x = np.linspace(50,180,100)
plt.plot(x,mlab.normpdf(x,mean1,sigma),'-o',color='b')

mean2 = float(k2_mu)
variance = float(k2_sigma)
sigma = np.sqrt(variance)
x = np.linspace(50,180,100)
plt.plot(x,mlab.normpdf(x,mean2,sigma),'-o', color='g')

if mean1<mean2:
    plt.hist(t[:len(result['clusters'][0])],histtype='stepfilled', bins=20, normed=True,color='#325ADB', label='Uniform')
    plt.hist(t[len(result['clusters'][0]):],histtype='stepfilled', bins=20, normed=True,color='#2FBA4F', label='Uniform')
else:
    plt.hist(t[:len(result['clusters'][0])], histtype='stepfilled',bins=20, normed=True,color='#2FBA4F', label='Uniform')
    plt.hist(t[len(result['clusters'][0]):], histtype='stepfilled',bins=20, normed=True,color='#325ADB', label='Uniform')


plt.title("PeloCano - Gaussian Mixture")
plt.xlabel("Value")
plt.ylabel("Frequency")
plt.show()


mean_full=np.mean(t)
variance_full = np.std(t)
sigma_full = np.sqrt(variance_full)
plt.plot(x,mlab.normpdf(x,mean_full,sigma_full),'-o',color='r')
plt.plot(x,mlab.normpdf(x,mean2,sigma),'-o', color='g')
plt.plot(x,mlab.normpdf(x,mean1,sigma),'-o',color='b')



plt.title("PeloCano - Gaussian Mixture")
plt.xlabel("Value")
plt.ylabel("Frequency")
plt.show()


#get robots .45
fake = [[37, 53], [39, 51], [41, 45], [45, 52], [44, 54], [50, 47], [39, 46], [45, 44], [43, 66], [46, 69], [46, 50], [44, 49], [54, 53], [35, 29], [40, 45], [59, 42], [40, 38], [46, 35], [50, 39], [56, 45], [53, 39], [50, 30], [58, 58], [46, 47], [41, 40], [53, 34]]

fake_array = []
count=0
for i in fake:
    #how many votes on i
    for j in xrange(i[0]):
        fake_array.append([fakefit[count][0]])
    count += 1


t = np.array(fake_array, np.int32)
#t.reshape(len(t),2).shape
result = expectation_maximization(t, nbiter=3, epsilon=0.1)

k1_mu = result['params'][0]['mu']
k1_p = result['params'][0]['proba']
k1_sigma = result['params'][0]['sigma']

k2_mu = result['params'][1]['mu']
k2_p = result['params'][1]['proba']
k2_sigma = result['params'][1]['sigma']

mean1 = float(k1_mu)
variance = float(k1_sigma)
sigma = np.sqrt(variance)
x = np.linspace(50,180,100)
plt.plot(x,mlab.normpdf(x,mean1,sigma),'-o',color='b')

mean2 = float(k2_mu)
variance = float(k2_sigma)
sigma = np.sqrt(variance)
x = np.linspace(50,180,100)
plt.plot(x,mlab.normpdf(x,mean2,sigma),'-o', color='g')

if mean1<mean2:
    plt.hist(t[:len(result['clusters'][0])],histtype='stepfilled', bins=20, normed=True,color='#325ADB', label='Uniform')
    plt.hist(t[len(result['clusters'][0]):],histtype='stepfilled', bins=20, normed=True,color='#2FBA4F', label='Uniform')
else:
    plt.hist(t[:len(result['clusters'][0])], histtype='stepfilled',bins=20, normed=True,color='#2FBA4F', label='Uniform')
    plt.hist(t[len(result['clusters'][0]):], histtype='stepfilled',bins=20, normed=True,color='#325ADB', label='Uniform')


plt.title("PeloCano - Gaussian Mixture")
plt.xlabel("Value")
plt.ylabel("Frequency")
plt.show()


mean_full=np.mean(t)
variance_full = np.std(t)
sigma_full = np.sqrt(variance_full)
plt.plot(x,mlab.normpdf(x,mean_full,sigma_full),'-o',color='r')
plt.plot(x,mlab.normpdf(x,mean2,sigma),'-o', color='g')
plt.plot(x,mlab.normpdf(x,mean1,sigma),'-o',color='b')



plt.title("PeloCano - Gaussian Mixture")
plt.xlabel("Value")
plt.ylabel("Frequency")
plt.show()
