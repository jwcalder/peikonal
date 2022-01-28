import numpy as np
import graphlearning as gl
import matplotlib.pyplot as plt
import sklearn.datasets as datasets
from utils import peikonal_depth
from utils import half_moon
import sys

size = 10
label_color = 'r'
markers = ['v','s','^']

frac=0.05
seed=0
alpha=2
n=1000
eps = 1

#dataset = 'gaussian'
dataset = 'moon'

np.random.seed(seed)
if dataset == 'gaussian':
    X = np.random.randn(n,2)
elif dataset == 'moon':
    X = half_moon(n)

knn_ind, knn_dist = gl.weightmatrix.knnsearch(X,50)
W = gl.weightmatrix.knn(X,10,knn_data=(knn_ind,knn_dist))
G = gl.graph(W)
if not G.isconnected():
    sys.exit('Graph is not connected')
d = np.max(knn_dist,axis=1)
kde = (d/d.max())**(-1)


plt.figure(1)
plt.scatter(X[:,0],X[:,1], s=size)
for k, alpha in enumerate([-1,0,1]):

    median, depth = peikonal_depth(G, kde, frac, alpha)
    depth = depth/np.max(depth)
    depth = 1-depth
    plt.figure(1)
    plt.scatter(X[median,0],X[median,1], c=label_color, marker=markers[k], s=5*size, edgecolors='black')

    plt.figure()
    plt.scatter(X[:,0],X[:,1], s=size, c=depth)
    plt.axis('off')
    plt.axis('square')
    plt.savefig('figures/depth_'+dataset+'_alpha%d.pdf'%alpha)

plt.figure(1)
plt.axis('off')
plt.axis('square')
plt.savefig('figures/depth_'+dataset+'.pdf')


