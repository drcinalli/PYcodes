
import itertools

import numpy as np
from scipy import linalg
import matplotlib.pyplot as plt
import matplotlib as mpl

from sklearn import mixture

# Number of samples per component
n_samples = 500

# Generate random sample, two components
np.random.seed(0)
C = np.array([[0., -0.1], [1.7, .4]])
X = np.r_[np.dot(np.random.randn(n_samples, 2), C),
          .7 * np.random.randn(n_samples, 2) + np.array([-6, 3])]


#fake = [[74, 15], [74, 11], [69, 16], [73, 21], [68, 28], [60, 27], [69, 33], [65, 27], [50, 40], [45, 35], [66, 46], [59, 46], [39, 41], [49, 55], [44, 52], [44, 39], [43, 64], [30, 45], [32, 63], [31, 63], [23, 69], [27, 69], [15, 76], [12, 77], [23, 61], [16, 81]]
fake = [[78.911204246880132, -23.0], [78.911204246880132, -23.0], [78.911204246880132, -23.0], [78.911204246880132, -23.0], [78.911204246880132, -23.0], [78.911204246880132, -23.0], [78.911204246880132, -23.0], [78.911204246880132, -23.0], [78.911204246880132, -23.0], [82.833563725395152, -46.0], [82.833563725395152, -46.0], [82.833563725395152, -46.0], [82.833563725395152, -46.0], [82.833563725395152, -46.0], [82.833563725395152, -46.0], [82.833563725395152, -46.0], [82.833563725395152, -46.0], [82.833563725395152, -46.0], [82.833563725395152, -46.0], [82.833563725395152, -46.0], [82.833563725395152, -46.0], [82.833563725395152, -46.0], [82.833563725395152, -46.0], [82.833563725395152, -46.0], [88.404743229436974, -58.0], [88.404743229436974, -58.0], [88.404743229436974, -58.0], [88.404743229436974, -58.0], [88.404743229436974, -58.0], [88.404743229436974, -58.0], [88.404743229436974, -58.0], [88.404743229436974, -58.0], [88.404743229436974, -58.0], [88.404743229436974, -58.0], [88.635650570838351, -69.0], [88.635650570838351, -69.0], [88.635650570838351, -69.0], [88.635650570838351, -69.0], [88.635650570838351, -69.0], [88.635650570838351, -69.0], [88.635650570838351, -69.0], [88.635650570838351, -69.0], [93.884715335422769, -81.0], [93.884715335422769, -81.0], [93.884715335422769, -81.0], [93.884715335422769, -81.0], [93.884715335422769, -81.0], [93.884715335422769, -81.0], [97.404743229436974, -93.0], [97.404743229436974, -93.0], [97.404743229436974, -93.0], [97.404743229436974, -93.0], [97.404743229436974, -93.0], [97.404743229436974, -93.0], [103.57772817821186, -104.0], [103.57772817821186, -104.0], [103.57772817821186, -104.0], [103.57772817821186, -104.0], [103.57772817821186, -104.0], [103.57772817821186, -104.0], [103.57772817821186, -104.0], [109.80326398495583, -128.0], [109.80326398495583, -128.0], [128.92636961057349, -163.0], [131.40224255080034, -175.0], [131.40224255080034, -175.0], [136.67004637770998, -186.0], [136.67004637770998, -186.0], [136.67004637770998, -186.0], [136.67004637770998, -186.0], [144.18033988749895, -198.0], [152.90038970424035, -210.0], [152.90038970424035, -210.0], [152.90038970424035, -210.0]]
t = np.array(fake, np.int32)
gmm = mixture.GMM(n_components=2, covariance_type='spherical')
gmm.fit(fake)






lowest_bic = np.infty
bic = []
n_components_range = range(1, 12)
cv_types = ['spherical', 'tied', 'diag', 'full']
#cv_types = ['spherical',]
for cv_type in cv_types:
    for n_components in n_components_range:
        # Fit a mixture of Gaussians with EM
        gmm = mixture.GMM(n_components=n_components, covariance_type=cv_type)
        gmm.fit(t)
        bic.append(gmm.bic(t))
        if bic[-1] < lowest_bic:
            lowest_bic = bic[-1]
            best_gmm = gmm

# bic = np.array(bic)
# color_iter = itertools.cycle(['k', 'r', 'g', 'b', 'c', 'm', 'y'])
# clf = best_gmm
# bars = []
# #
# #
# #######################
# # Plot the BIC scores
# spl = plt.subplot(2, 1, 1)
# for i, (cv_type, color) in enumerate(zip(cv_types, color_iter)):
#     xpos = np.array(n_components_range) + .2 * (i - 2)
#     bars.append(plt.bar(xpos, bic[i * len(n_components_range):
#                                   (i + 1) * len(n_components_range)],
#                         width=.2, color=color))
# plt.xticks(n_components_range)
# plt.ylim([bic.min() * 1.01 - .01 * bic.max(), bic.max()])
# plt.title('BIC score per model')
# xpos = np.mod(bic.argmin(), len(n_components_range)) + .65 +\
#     .2 * np.floor(bic.argmin() / len(n_components_range))
# plt.text(xpos, bic.min() * 0.97 + .03 * bic.max(), '*', fontsize=14)
# spl.set_xlabel('Number of components')
# spl.legend([b[0] for b in bars], cv_types)


# # Plot the winner
# splot = plt.subplot(2, 1, 2)
# Y_ = clf.predict(t)
# for i, (mean, covar, color) in enumerate(zip(clf.means_, clf.covars_,
#                                              color_iter)):
#     v, w = linalg.eigh(covar)
#     if not np.any(Y_ == i):
#         continue
#     plt.scatter(t[Y_ == i, 0], t[Y_ == i, 1], .8, color=color)
#
#     # Plot an ellipse to show the Gaussian component
#     angle = np.arctan2(w[0][1], w[0][0])
#     angle = 180 * angle / np.pi  # convert to degrees
#     v *= 4
#     ell = mpl.patches.Ellipse(mean, v[0], v[1], 180 + angle, color=color)
#     ell.set_clip_box(splot.bbox)
#     ell.set_alpha(.5)
#     splot.add_artist(ell)
#
# plt.xlim(-10, 10)
# plt.ylim(-3, 6)
plt.xticks(())
plt.yticks(())
plt.title('Selected GMM: full model, 2 components')
plt.subplots_adjust(hspace=.35, bottom=.02)
plt.show()