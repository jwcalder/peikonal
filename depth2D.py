import numpy as np
import graphlearning as gl
import matplotlib.pyplot as plt
import sklearn.datasets as datasets
from utils import peikonal_depth
from utils import half_moon
import sys

size = 10
label_color = 'w'

frac=0.05
seed=0
alpha=2
n=1000
eps = 1

#dataset = 'gaussian'
#dataset = 'moon'
dataset = 'mixture'

np.random.seed(seed)
if dataset == 'gaussian':
    X = np.random.randn(n,2)
elif dataset == 'moon':
    X = half_moon(n)
elif dataset == 'mixture':
    m1 = int(5*n/6)
    m2 = n - m1
    X = np.random.randn(m1,2)
    Y = 0.5*np.random.randn(m2,2) + [2.5,2.5]
    X = np.vstack((X,Y))

knn_ind, knn_dist = gl.weightmatrix.knnsearch(X,50)
W = gl.weightmatrix.knn(X,10,knn_data=(knn_ind,knn_dist))
G = gl.graph(W)
if not G.isconnected():
    sys.exit('Graph is not connected')
d = np.max(knn_dist,axis=1)
kde = (d/d.max())**(-1)

medians = np.zeros(3).astype(int)
for k, alpha in enumerate([-1,0,1]):

    median, depth = peikonal_depth(G, kde, frac, alpha)
    depth = depth/np.max(depth)
    depth = 1-depth
    medians[k] = median

    plt.figure()
    plt.scatter(X[:,0],X[:,1], s=size, c=depth)
    plt.axis('off')
    plt.axis('square')
    plt.savefig('figures/depth_'+dataset+'_alpha%d.pdf'%alpha)

plt.figure()
plt.scatter(X[:,0],X[:,1], s=size, c=depth)
plt.scatter(X[medians[1],0],X[medians[1],1], c=label_color, marker='s', s=10*size, edgecolors='black', linewidth=2)
plt.scatter(X[medians[0],0],X[medians[0],1], c=label_color, marker='v', s=15*size, edgecolors='black', linewidth=2)
plt.scatter(X[medians[2],0],X[medians[2],1], c=label_color, marker='^', s=15*size, edgecolors='black', linewidth=2)
plt.axis('off')
plt.axis('square')
plt.savefig('figures/depth_'+dataset+'.pdf')


