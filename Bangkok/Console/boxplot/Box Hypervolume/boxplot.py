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


# function for setting the colors of the box plots pairs
def setBoxColors(bp):
    setp(bp['boxes'][0], color='black')
    setp(bp['caps'][0], color='black')
    setp(bp['caps'][1], color='black')
    setp(bp['whiskers'][0], color='black')
    setp(bp['whiskers'][1], color='black')
    #setp(bp['fliers'][0], color='black')
    #setp(bp['fliers'][1], color='black')
    setp(bp['medians'][0], color='black')

    setp(bp['boxes'][1], color='red')
    setp(bp['caps'][2], color='red')
    setp(bp['caps'][3], color='red')
    setp(bp['whiskers'][2], color='red')
    setp(bp['whiskers'][3], color='red')
    #setp(bp['fliers'][1], color='red')
    #setp(bp['fliers'][2], color='red')
    #setp(bp['fliers'][3], color='red')
    setp(bp['medians'][1], color='red')

datax = []
data1=[]
with open('ZDT1g.txt', 'rb') as f:
    my_list1 = pickle.load(f)
data1.append(my_list1)
with open('ZDT1k.txt', 'rb') as f:
    my_list2 = pickle.load(f)
data1.append(my_list2)
media1a = np.mean(my_list1)
std1a =  np.std(my_list1)
media1b = np.mean(my_list2)
std1b =  np.std(my_list2)

datax.append(my_list1)
datax.append(my_list2)


data2=[]
with open('ZDT2g.txt', 'rb') as f:
    my_list1 = pickle.load(f)
data2.append(my_list1)
with open('ZDT2k.txt', 'rb') as f:
    my_list2 = pickle.load(f)
data2.append(my_list2)

media2a = np.mean(my_list1)
std2a =  np.std(my_list1)
media2b = np.mean(my_list2)
std2b =  np.std(my_list2)

datax.append(my_list1)
datax.append(my_list2)

data3=[]
with open('ZDT3g.txt', 'rb') as f:
    my_list1 = pickle.load(f)
data3.append(my_list1)
with open('ZDT3k.txt', 'rb') as f:
    my_list2 = pickle.load(f)
data3.append(my_list2)

media3a = np.mean(my_list1)
std3a =  np.std(my_list1)
media3b = np.mean(my_list2)
std3b =  np.std(my_list2)

datax.append(my_list1)
datax.append(my_list2)


data4=[]
with open('ZDT4g.txt', 'rb') as f:
    my_list1 = pickle.load(f)
data4.append(my_list1)
with open('ZDT4k.txt', 'rb') as f:
    my_list2 = pickle.load(f)
data4.append(my_list2)

media4a = np.mean(my_list1)
std4a =  np.std(my_list1)
media4b = np.mean(my_list2)
std4b =  np.std(my_list2)

datax.append(my_list1)
datax.append(my_list2)

data6=[]
with open('ZDT6g.txt', 'rb') as f:
    my_list1 = pickle.load(f)
data6.append(my_list1)
with open('ZDT6k.txt', 'rb') as f:
    my_list2 = pickle.load(f)
data6.append(my_list2)

media6a = np.mean(my_list1)
std6a =  np.std(my_list1)
media6b = np.mean(my_list2)
std6b =  np.std(my_list2)

datax.append(my_list1)
datax.append(my_list2)

data7=[]
with open('DTLZ2g.txt', 'rb') as f:
    my_list1 = pickle.load(f)
data7.append(my_list1)
with open('DTLZ2k.txt', 'rb') as f:
    my_list2 = pickle.load(f)
data7.append(my_list2)

media7a = np.mean(my_list1)
std7a =  np.std(my_list1)
media7b = np.mean(my_list2)
std7b =  np.std(my_list2)

datax.append(my_list1)
datax.append(my_list2)


data8=[]
with open('DTLZ3g.txt', 'rb') as f:
    my_list1 = pickle.load(f)
data8.append(my_list1)
with open('DTLZ3k.txt', 'rb') as f:
    my_list2 = pickle.load(f)
data8.append(my_list2)

media8a = np.mean(my_list1)
std8a =  np.std(my_list1)
media8b = np.mean(my_list2)
std8b =  np.std(my_list2)

datax.append(my_list1)
datax.append(my_list2)


data9=[]
with open('DTLZ4g.txt', 'rb') as f:
    my_list1 = pickle.load(f)
data9.append(my_list1)
with open('DTLZ4k.txt', 'rb') as f:
    my_list2 = pickle.load(f)
data9.append(my_list2)

media9a = np.mean(my_list1)
std9a =  np.std(my_list1)
media9b = np.mean(my_list2)
std9b =  np.std(my_list2)

datax.append(my_list1)
datax.append(my_list2)

data10=[]
with open('DTLZ5g.txt', 'rb') as f:
    my_list1 = pickle.load(f)
data10.append(my_list1)
with open('DTLZ5k.txt', 'rb') as f:
    my_list2 = pickle.load(f)
data10.append(my_list2)

media10a = np.mean(my_list1)
std10a =  np.std(my_list1)
media10b = np.mean(my_list2)
std10b =  np.std(my_list2)

datax.append(my_list1)
datax.append(my_list2)

data11=[]
with open('DTLZ6g.txt', 'rb') as f:
    my_list1 = pickle.load(f)
data11.append(my_list1)
with open('DTLZ6k.txt', 'rb') as f:
    my_list2 = pickle.load(f)
data11.append(my_list2)

media11a = np.mean(my_list1)
std11a =  np.std(my_list1)
media11b = np.mean(my_list2)
std11b =  np.std(my_list2)

datax.append(my_list1)
datax.append(my_list2)


data12=[]
with open('DTLZ7gv2.txt', 'rb') as f:
    my_list1 = pickle.load(f)
data12.append(my_list1)
with open('DTLZ7kv2.txt', 'rb') as f:
    my_list2 = pickle.load(f)
data12.append(my_list2)

media12a = np.mean(my_list1)
std12a =  np.std(my_list1)
media12b = np.mean(my_list2)
std12b =  np.std(my_list2)

datax.append(my_list1)
datax.append(my_list2)



data13=[]
with open('DTLZ1g.txt', 'rb') as f:
    my_list1 = pickle.load(f)
data13.append(my_list1)
with open('DTLZ2k.txt', 'rb') as f:
    my_list2 = pickle.load(f)
data13.append(my_list2)

media13a = np.mean(my_list1)
std13a =  np.std(my_list1)
media13b = np.mean(my_list2)
std13b =  np.std(my_list2)

datax.append(my_list1)
datax.append(my_list2)


restmp = zip(*datax)
res = pd.DataFrame(restmp)

fig = figure()
ax = axes()
hold(True)

# first boxplot pair
bp = boxplot(data1, positions = [1, 2], widths = 0.6, notch=True)
setBoxColors(bp)

bp = boxplot(data2, positions = [4, 5], widths = 0.6, notch=True)
setBoxColors(bp)

bp = boxplot(data3, positions = [7, 8], widths = 0.6, notch=True)
setBoxColors(bp)

bp = boxplot(data4, positions = [10, 11], widths = 0.6, notch=True)
setBoxColors(bp)

bp = boxplot(data6, positions = [13, 14], widths = 0.6, notch=True)
setBoxColors(bp)

bp = boxplot(data13, positions = [16, 17], widths = 0.6, notch=True)
setBoxColors(bp)




bp = boxplot(data7, positions = [19, 20], widths = 0.6, notch=True)
setBoxColors(bp)

bp = boxplot(data8, positions = [22, 23], widths = 0.6, notch=True)
setBoxColors(bp)

bp = boxplot(data9, positions = [25, 26], widths = 0.6, notch=True)
setBoxColors(bp)

bp = boxplot(data10, positions = [28, 29], widths = 0.6, notch=True)
setBoxColors(bp)

bp = boxplot(data11, positions = [31, 32], widths = 0.6, notch=True)
setBoxColors(bp)

bp = boxplot(data12, positions = [34, 35], widths = 0.6, notch=True)
setBoxColors(bp)


# # second boxplot pair
# bp = boxplot(data2, positions = [4, 5], widths = 0.6, notch=True)
# setBoxColors(bp)
# #bp = boxplot(B, positions = [4, 5], widths = 0.6)
# #setBoxColors(bp)
#
# # thrid boxplot pair
# bp = boxplot(data3, positions = [7, 8], widths = 0.6, notch=True)
# setBoxColors(bp)
# #bp = boxplot(C, positions = [7, 8], widths = 0.6)
# #setBoxColors(bp)
#
#
# # thrid boxplot pair
# bp = boxplot(data4, positions = [10, 11], widths = 0.6, notch=True)
# setBoxColors(bp)
#
# # set axes limits and labels
# xlim(0,12)
# ylim(0.50,0.61)
#xlim(0,8)
ylim(0.71,1.015)

#ax.set_xticklabels([' ', 'ZDT1', 'ZDT2', 'ZDT3', 'ZDT4', 'ZDT6', ' '])
#ax.set_xticks([0.5, 1.5, 4.5, 7.5, 10.5, 13.5, 14.5])
ax.set_xticklabels([' ', 'ZDT1', 'ZDT2', 'ZDT3', 'ZDT4', 'ZDT6', 'DTLZ1', 'DTLZ2', 'DTLZ3', 'DTLZ4', 'DTLZ5', 'DTLZ6', 'DTLZ7', ' '])
ax.set_xticks([0.5, 1.5, 4.5, 7.5, 10.5, 13.5,16.5, 19.5, 22.5, 25.5, 28.5, 31.5, 34.5 ,  35.5, ])

# draw temporary red and blue lines and use them to create a legend
hB, = plot([1,1],'k-')
hR, = plot([1,1],'r-')
legend ((hB, hR),('Gaussian Mixture', 'K-means'),loc='upper center', bbox_to_anchor=(0.5, 1.09), ncol=3, fancybox=True, shadow=True)

# Put a legend below current axis
#ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))

hB.set_visible(False)
hR.set_visible(False)
savefig('boxcompare.png')
show()

pp = PdfPages('ZDT_results.pdf')
pp.savefig(fig)
pp.close()


tab_final= conover_inman_procedure(res)

print "fim"