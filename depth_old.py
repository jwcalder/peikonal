import numpy as np
import graphlearning as gl
import matplotlib.pyplot as plt
import sklearn.datasets as datasets
from utils import peikonal_depth
import sys

size = 20
vmin = 0
vmax = 1
#cmap = 'Paired'
#cmap = 'Dark2'
#cmap = 'copper'
cmap = 'winter'
label_color = 'r'
marker = "^"
markers = ['v','s','^']

frac=0.05
seed=0
alpha=2
n=500
eps = 1
np.random.seed(seed)
X = np.random.randn(n,2)
W = gl.weightmatrix.epsilon_ball(X,eps)
G = gl.graph(W)
if not G.isconnected():
    sys.exit('Graph is not connected')
d = G.degree_vector()
kde = d/d.max()


plt.figure(1)
plt.scatter(X[:,0],X[:,1], s=size)
for k, alpha in enumerate([-1,0,1]):

    median, depth = peikonal_depth(G, kde, frac, alpha)
    depth = depth/np.max(depth)
    depth = 1-depth**(3/4)
    plt.figure(1)
    plt.scatter(X[median,0],X[median,1], c=label_color, marker=markers[k], s=5*size, edgecolors='black')

    plt.figure()
    plt.scatter(X[:,0],X[:,1], s=size, c=depth)
    plt.savefig('figures/depth_alpha%d.pdf'%alpha)

plt.figure(1)
plt.savefig('figures/depth.pdf')


