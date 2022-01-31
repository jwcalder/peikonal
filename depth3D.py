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
alpha=1
n=1000
eps = 1

dataset = 'sphere'
X = utils.half_sphere(n)

knn_ind, knn_dist = gl.weightmatrix.knnsearch(X,50)
W = gl.weightmatrix.knn(X,10,knn_data=(knn_ind,knn_dist))
G = gl.graph(W)
if not G.isconnected():
    sys.exit('Graph is not connected')
d = np.max(knn_dist,axis=1)
kde = (d/d.max())**(-1)

k = 2
alpha = 1
median, depth = peikonal_depth(G, kde, frac, alpha)
depth = depth/np.max(depth)
depth = 1-depth


fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.scatter(X[:,0], X[:,1], X[:,2], s=0.1*size)
ax.scatter(X[median,0], X[median,1], X[median,2], c=label_color, marker=markers[k], s=5*size, edgecolors='black')
plt.axis('off')
plt.savefig('figures/depth_'+dataset+'_alpha%d.pdf'%alpha)
# plt.savefig('depth_'+dataset+'_alpha %d.pdf'%alpha)


fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.scatter(X[:,0], X[:,1], X[:,2], s=size, c=depth)
plt.axis('off')
plt.savefig('figures/depth_'+dataset+'.pdf')
# plt.savefig('depth_'+dataset+'.pdf')
