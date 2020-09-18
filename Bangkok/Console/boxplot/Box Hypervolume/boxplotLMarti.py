import numpy as np
from numpy.random import randn
import pandas as pd
from scipy import stats
import matplotlib as mpl
import matplotlib.pyplot as plt
import pickle
from matplotlib.backends.backend_pdf import PdfPages
import itertools
import scipy.stats as stats
import math


from pylab import plot, show, savefig, xlim, figure, \
                hold, ylim, legend, boxplot, setp, axes


def conover_inman_procedure(data, alpha=0.05):
    num_runs = len(data)
    num_algos = len(data.columns)
    N = num_runs*num_algos

    _,p_value = stats.kruskal(*[data[col] for col in data.columns])

    ranked =  stats.rankdata(np.concatenate([data[col] for col in data.columns]))

    ranksums = []
    for i in range(num_algos):
        ranksums.append(np.sum(ranked[num_runs*i:num_runs*(i+1)]))

    S_sq = (np.sum(ranked**2) - N*((N+1)**2)/4)/(N-1)

    right_side = stats.t.cdf(1-(alpha/2), N-num_algos) * \
                 math.sqrt((S_sq*((N-1-p_value)/(N-1)))*2/num_runs)

    res = pd.DataFrame(columns=data.columns, index=data.columns)

    for i,j in itertools.combinations(np.arange(num_algos),2):
        res[res.columns[i]].ix[j] = abs(ranksums[i] - ranksums[j]/num_runs) > right_side
        res[res.columns[j]].ix[i] = abs(ranksums[i] - ranksums[j]/num_runs) > right_side
    return res


datax = []
data1=[]
with open('ZDT1g.txt', 'rb') as f:
    my_list1 = pickle.load(f)
data1.append(my_list1)
with open('ZDT1k.txt', 'rb') as f:
    my_list2 = pickle.load(f)
data1.append(my_list2)

datax.append(my_list1)
datax.append(my_list2)


data2=[]
with open('ZDT2g.txt', 'rb') as f:
    my_list1 = pickle.load(f)
data2.append(my_list1)
with open('ZDT2k.txt', 'rb') as f:
    my_list2 = pickle.load(f)
data2.append(my_list2)

datax.append(my_list1)
datax.append(my_list2)

data3=[]
with open('ZDT3g.txt', 'rb') as f:
    my_list1 = pickle.load(f)
data3.append(my_list1)
with open('ZDT3k.txt', 'rb') as f:
    my_list2 = pickle.load(f)
data3.append(my_list2)


datax.append(my_list1)
datax.append(my_list2)

restmp = zip(*datax)
res = pd.DataFrame(restmp)


tab_final= conover_inman_procedure(res)

print "fim"