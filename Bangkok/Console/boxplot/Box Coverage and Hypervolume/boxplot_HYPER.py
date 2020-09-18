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

    setp(bp['boxes'][2], color='green')
    setp(bp['caps'][4], color='green')
    setp(bp['caps'][5], color='green')
    setp(bp['whiskers'][4], color='green')
    setp(bp['whiskers'][5], color='green')
    #setp(bp['fliers'][1], color='red')
    #setp(bp['fliers'][2], color='red')
    #setp(bp['fliers'][3], color='red')
    setp(bp['medians'][2], color='green')

datax = []
data1=[]
with open('ZDT1 NSGA2 Hypervolume GM.txt', 'rb') as f:
    my_list1 = pickle.load(f)
data1.append(my_list1)
with open('ZDT1 SPEA2 Hypervolume GM.txt', 'rb') as f:
    my_list2 = pickle.load(f)
data1.append(my_list2)
with open('ZDT1 SMS2 Hypervolume GM.txt', 'rb') as f:
    my_list3 = pickle.load(f)
data1.append(my_list3)

media1a = np.mean(my_list1)
std1a =  np.std(my_list1)
media1b = np.mean(my_list2)
std1b =  np.std(my_list2)
media1c = np.mean(my_list3)
std1c =  np.std(my_list3)

datax.append(my_list1)
datax.append(my_list2)
datax.append(my_list3)

restmp = zip(*datax)
res = pd.DataFrame(restmp)
tab_final= conover_inman_procedure(res)

datax = []
data2=[]
with open('ZDT2 NSGA2 Hypervolume GM.txt', 'rb') as f:
    my_list1 = pickle.load(f)
data2.append(my_list1)
with open('ZDT2 SPEA2 Hypervolume GM.txt', 'rb') as f:
    my_list2 = pickle.load(f)
data2.append(my_list2)
with open('ZDT2 SMS2 Hypervolume GM.txt', 'rb') as f:
    my_list3 = pickle.load(f)
data2.append(my_list3)

media1a = np.mean(my_list1)
std1a =  np.std(my_list1)
media1b = np.mean(my_list2)
std1b =  np.std(my_list2)
media1c = np.mean(my_list3)
std1c =  np.std(my_list3)

datax.append(my_list1)
datax.append(my_list2)
datax.append(my_list3)


restmp = zip(*datax)
res = pd.DataFrame(restmp)
tab_final= conover_inman_procedure(res)

datax = []
data3=[]
with open('ZDT3 NSGA2 Hypervolume GM.txt', 'rb') as f:
    my_list1 = pickle.load(f)
data3.append(my_list1)
with open('ZDT3 SPEA2 Hypervolume GM.txt', 'rb') as f:
    my_list2 = pickle.load(f)
data3.append(my_list2)
with open('ZDT3 SMS2 Hypervolume GM.txt', 'rb') as f:
    my_list3 = pickle.load(f)
data3.append(my_list3)

media1a = np.mean(my_list1)
std1a =  np.std(my_list1)
media1b = np.mean(my_list2)
std1b =  np.std(my_list2)
media1c = np.mean(my_list3)
std1c =  np.std(my_list3)

datax.append(my_list1)
datax.append(my_list2)
datax.append(my_list3)

restmp = zip(*datax)
res = pd.DataFrame(restmp)
tab_final= conover_inman_procedure(res)


datax = []
data4=[]
with open('ZDT4 NSGA2 Hypervolume GM.txt', 'rb') as f:
    my_list1 = pickle.load(f)
data4.append(my_list1)
with open('ZDT4 SPEA2 Hypervolume GM.txt', 'rb') as f:
    my_list2 = pickle.load(f)
data4.append(my_list2)
with open('ZDT4 SMS2 Hypervolume GM.txt', 'rb') as f:
    my_list3 = pickle.load(f)
data4.append(my_list3)

media1a = np.mean(my_list1)
std1a =  np.std(my_list1)
media1b = np.mean(my_list2)
std1b =  np.std(my_list2)
media1c = np.mean(my_list3)
std1c =  np.std(my_list3)

datax.append(my_list1)
datax.append(my_list2)
datax.append(my_list3)

restmp = zip(*datax)
res = pd.DataFrame(restmp)
tab_final= conover_inman_procedure(res)

datax = []
data6=[]
with open('ZDT6 NSGA2 Hypervolume GM.txt', 'rb') as f:
    my_list1 = pickle.load(f)
data6.append(my_list1)
with open('ZDT6 SPEA2 Hypervolume GM.txt', 'rb') as f:
    my_list2 = pickle.load(f)
data6.append(my_list2)
with open('ZDT6 SMS2 Hypervolume GM.txt', 'rb') as f:
    my_list3 = pickle.load(f)
data6.append(my_list3)

media1a = np.mean(my_list1)
std1a =  np.std(my_list1)
media1b = np.mean(my_list2)
std1b =  np.std(my_list2)
media1c = np.mean(my_list3)
std1c =  np.std(my_list3)

datax.append(my_list1)
datax.append(my_list2)
datax.append(my_list3)

restmp = zip(*datax)
res = pd.DataFrame(restmp)
tab_final= conover_inman_procedure(res)



fig = figure()
ax = axes()
hold(True)

# first boxplot pair
bp = boxplot(data1, positions = [1, 2, 3], widths = 0.6, notch=True)
setBoxColors(bp)

bp = boxplot(data2, positions = [5, 6, 7], widths = 0.6, notch=True)
setBoxColors(bp)

bp = boxplot(data3, positions = [9, 10, 11], widths = 0.6, notch=True)
setBoxColors(bp)

bp = boxplot(data4, positions = [12, 13, 14], widths = 0.6, notch=True)
setBoxColors(bp)

bp = boxplot(data6, positions = [16, 17, 18], widths = 0.6, notch=True)
setBoxColors(bp)

#xlim(0,8)
ylim(0.82,1.05)

ax.set_xticklabels([' ', 'ZDT1', 'ZDT2', 'ZDT3', 'ZDT4', 'ZDT6', ' '])
ax.set_xticks([0.5, 2.5, 6.0, 9.5, 13.0, 16.5, 18.0])
#ax.set_xticklabels([' ', 'ZDT1', 'ZDT2', 'ZDT3', 'ZDT4', 'ZDT6', 'DTLZ1', 'DTLZ2', 'DTLZ3', 'DTLZ4', 'DTLZ5', 'DTLZ6', 'DTLZ7', ' '])
#ax.set_xticks([0.5, 1.5, 4.5, 7.5, 10.5, 13.5,16.5, 19.5, 22.5, 25.5, 28.5, 31.5, 34.5 ,  35.5, ])

# draw temporary red and blue lines and use them to create a legend
hB, = plot([1,1],'k-')
hR, = plot([1,1],'r-')
hG, = plot([1,1],'g-')
legend ((hB, hR, hG),('CI-NSGA-II', 'CI-SPEA2', 'CI-SMS-EMOA'),loc='upper center', bbox_to_anchor=(0.5, 1.09), ncol=3, fancybox=True, shadow=True)

# Put a legend below current axis
#ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))

hB.set_visible(False)
hR.set_visible(False)
hG.set_visible(False)
savefig('boxcompare.png')
show()

pp = PdfPages('ZDT_results.pdf')
pp.savefig(fig)
pp.close()





####### PARTE 2


datax = []
data1=[]
with open('DTLZ1 NSGA2 Hypervolume GM.txt', 'rb') as f:
    my_list1 = pickle.load(f)
data1.append(my_list1)
with open('DTLZ1 SPEA2 Hypervolume GM.txt', 'rb') as f:
    my_list2 = pickle.load(f)
data1.append(my_list2)
with open('DTLZ1 SMS2 Hypervolume GM.txt', 'rb') as f:
    my_list3 = pickle.load(f)
data1.append(my_list3)

media1a = np.mean(my_list1)
std1a =  np.std(my_list1)
media1b = np.mean(my_list2)
std1b =  np.std(my_list2)
media1c = np.mean(my_list3)
std1c =  np.std(my_list3)

datax.append(my_list1)
datax.append(my_list2)
datax.append(my_list3)


restmp = zip(*datax)
res = pd.DataFrame(restmp)
tab_final= conover_inman_procedure(res)




datax = []
data2=[]
with open('DTLZ2 NSGA2 Hypervolume GM.txt', 'rb') as f:
    my_list1 = pickle.load(f)
data2.append(my_list1)
with open('DTLZ2 SPEA2 Hypervolume GM.txt', 'rb') as f:
    my_list2 = pickle.load(f)
data2.append(my_list2)
with open('DTLZ2 SMS2 Hypervolume GM.txt', 'rb') as f:
    my_list3 = pickle.load(f)
data2.append(my_list3)

media1a = np.mean(my_list1)
std1a =  np.std(my_list1)
media1b = np.mean(my_list2)
std1b =  np.std(my_list2)
media1c = np.mean(my_list3)
std1c =  np.std(my_list3)

datax.append(my_list1)
datax.append(my_list2)
datax.append(my_list3)


restmp = zip(*datax)
res = pd.DataFrame(restmp)
tab_final= conover_inman_procedure(res)



datax = []
data3=[]
with open('DTLZ3 NSGA2 Hypervolume GM.txt', 'rb') as f:
    my_list1 = pickle.load(f)
data3.append(my_list1)
with open('DTLZ3 SPEA2 Hypervolume GM.txt', 'rb') as f:
    my_list2 = pickle.load(f)
data3.append(my_list2)
with open('DTLZ3 SMS2 Hypervolume GM.txt', 'rb') as f:
    my_list3 = pickle.load(f)
data3.append(my_list3)

media1a = np.mean(my_list1)
std1a =  np.std(my_list1)
media1b = np.mean(my_list2)
std1b =  np.std(my_list2)
media1c = np.mean(my_list3)
std1c =  np.std(my_list3)

datax.append(my_list1)
datax.append(my_list2)
datax.append(my_list3)


restmp = zip(*datax)
res = pd.DataFrame(restmp)
tab_final= conover_inman_procedure(res)


datax = []
data4=[]
with open('DTLZ4 NSGA2 Hypervolume GM.txt', 'rb') as f:
    my_list1 = pickle.load(f)
data4.append(my_list1)
with open('DTLZ4 SPEA2 Hypervolume GM.txt', 'rb') as f:
    my_list2 = pickle.load(f)
data4.append(my_list2)
with open('DTLZ4 SMS2 Hypervolume GM.txt', 'rb') as f:
    my_list3 = pickle.load(f)
data4.append(my_list3)

media1a = np.mean(my_list1)
std1a =  np.std(my_list1)
media1b = np.mean(my_list2)
std1b =  np.std(my_list2)
media1c = np.mean(my_list3)
std1c =  np.std(my_list3)

datax.append(my_list1)
datax.append(my_list2)
datax.append(my_list3)


restmp = zip(*datax)
res = pd.DataFrame(restmp)
tab_final= conover_inman_procedure(res)



datax = []
data5=[]
with open('DTLZ5 NSGA2 Hypervolume GM.txt', 'rb') as f:
    my_list1 = pickle.load(f)
data5.append(my_list1)
with open('DTLZ5 SPEA2 Hypervolume GM.txt', 'rb') as f:
    my_list2 = pickle.load(f)
data5.append(my_list2)
with open('DTLZ5 SMS2 Hypervolume GM.txt', 'rb') as f:
    my_list3 = pickle.load(f)
data5.append(my_list3)

media1a = np.mean(my_list1)
std1a =  np.std(my_list1)
media1b = np.mean(my_list2)
std1b =  np.std(my_list2)
media1c = np.mean(my_list3)
std1c =  np.std(my_list3)

datax.append(my_list1)
datax.append(my_list2)
datax.append(my_list3)


restmp = zip(*datax)
res = pd.DataFrame(restmp)
tab_final= conover_inman_procedure(res)



datax = []
data6=[]
with open('DTLZ6 NSGA2 Hypervolume GM.txt', 'rb') as f:
    my_list1 = pickle.load(f)
data6.append(my_list1)
with open('DTLZ6 SPEA2 Hypervolume GM.txt', 'rb') as f:
    my_list2 = pickle.load(f)
data6.append(my_list2)
with open('DTLZ6 SMS2 Hypervolume GM.txt', 'rb') as f:
    my_list3 = pickle.load(f)
data6.append(my_list3)

media1a = np.mean(my_list1)
std1a =  np.std(my_list1)
media1b = np.mean(my_list2)
std1b =  np.std(my_list2)
media1c = np.mean(my_list3)
std1c =  np.std(my_list3)

datax.append(my_list1)
datax.append(my_list2)
datax.append(my_list3)


restmp = zip(*datax)
res = pd.DataFrame(restmp)
tab_final= conover_inman_procedure(res)



datax = []
data7=[]
with open('DTLZ7 NSGA2 Hypervolume GM.txt', 'rb') as f:
    my_list1 = pickle.load(f)
data7.append(my_list1)
with open('DTLZ7 SPEA2 Hypervolume GM.txt', 'rb') as f:
    my_list2 = pickle.load(f)
data7.append(my_list2)
with open('DTLZ7 SMS2 Hypervolume GM.txt', 'rb') as f:
    my_list3 = pickle.load(f)
data7.append(my_list3)

media1a = np.mean(my_list1)
std1a =  np.std(my_list1)
media1b = np.mean(my_list2)
std1b =  np.std(my_list2)
media1c = np.mean(my_list3)
std1c =  np.std(my_list3)

datax.append(my_list1)
datax.append(my_list2)
datax.append(my_list3)


restmp = zip(*datax)
res = pd.DataFrame(restmp)
tab_final= conover_inman_procedure(res)



fig = figure()
ax = axes()
hold(True)

# first boxplot pair
bp = boxplot(data1, positions = [1, 2, 3], widths = 0.6, notch=True)
setBoxColors(bp)

bp = boxplot(data2, positions = [5, 6, 7], widths = 0.6, notch=True)
setBoxColors(bp)

bp = boxplot(data3, positions = [9, 10, 11], widths = 0.6, notch=True)
setBoxColors(bp)

bp = boxplot(data4, positions = [12, 13, 14], widths = 0.6, notch=True)
setBoxColors(bp)

bp = boxplot(data5, positions = [16, 17, 18], widths = 0.6, notch=True)
setBoxColors(bp)

bp = boxplot(data6, positions = [20, 21, 22], widths = 0.6, notch=True)
setBoxColors(bp)

bp = boxplot(data7, positions = [24, 25, 26], widths = 0.6, notch=True)
setBoxColors(bp)

#xlim(0,8)
ylim(0.5,1.02)

ax.set_xticklabels([' ', 'DTLZ1', 'DTLZ2', 'DTLZ3', 'DTLZ4', 'DTLZ5','DTLZ6', 'DTLZ7', ' '])
ax.set_xticks([0.5, 2.5, 6.0, 9.5, 13.0, 16.5, 20.5, 24.0, 25.5])
#ax.set_xticklabels([' ', 'ZDT1', 'ZDT2', 'ZDT3', 'ZDT4', 'ZDT6', 'DTLZ1', 'DTLZ2', 'DTLZ3', 'DTLZ4', 'DTLZ5', 'DTLZ6', 'DTLZ7', ' '])
#ax.set_xticks([0.5, 1.5, 4.5, 7.5, 10.5, 13.5,16.5, 19.5, 22.5, 25.5, 28.5, 31.5, 34.5 ,  35.5, ])

# draw temporary red and blue lines and use them to create a legend
hB, = plot([1,1],'k-')
hR, = plot([1,1],'r-')
hG, = plot([1,1],'g-')
legend ((hB, hR, hG),('CI-NSGA-II', 'CI-SPEA2', 'CI-SMS-EMOA'),loc='upper center', bbox_to_anchor=(0.5, 1.09), ncol=3, fancybox=True, shadow=True)

# Put a legend below current axis
#ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))

hB.set_visible(False)
hR.set_visible(False)
hG.set_visible(False)
savefig('boxcompare2.png')
show()

pp = PdfPages('dtlz_results.pdf')
pp.savefig(fig)
pp.close()





print "fim"