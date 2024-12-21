# Simple SVM Classification for XOR dataset using sklearn

import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm

# XOR dataset
X = np.meshgrid(np.arange(-1, 1, 0.1), np.arange(-1, 1, 0.1))
X = np.array(X).reshape(2, -1).T
y = np.logical_xor(X[:, 0] > 0, X[:, 1] > 0)

# SVM
clf = svm.SVC(kernel='linear')
clf.fit(X, y)

# Plot
fig, axs = plt.subplots(1, 1, figsize=(5, 5))
axs.scatter(X[:, 0], X[:, 1], c=y, s=30, cmap=plt.cm.Paired)
axs.set_xlabel('X1')
axs.set_ylabel('X2')
axs.set_xlim(-1, 1)
axs.set_ylim(-1, 1)

# Show the decision boundary
axs.autoscale(False)
XX, YY = np.meshgrid(np.arange(-1, 1, 0.01), np.arange(-1, 1, 0.01))
Z = clf.decision_function(np.c_[XX.ravel(), YY.ravel()])
Z = Z.reshape(XX.shape)
axs.contour(XX, YY, Z, colors=['k', 'k', 'k'], linestyles=['--', '-', '--'], levels=[-1, 0, 1])
plt.show()