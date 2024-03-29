import numpy as np
import graphlearning as gl
import matplotlib.pyplot as plt
import sklearn.datasets as datasets
from utils import peikonal_depth
from utils import half_sphere, half_helix
import sys


size = 2
label_color = 'r'
markers = ['v','s','^']

frac=0.05
seed=0
alpha=1
eps = 1

dataset = 'sphere'
#dataset = 'helix'
#dataset = 'swissroll'

np.random.seed(seed)
if dataset == 'helix':
    n=1000
    X = half_helix(n)
elif dataset == 'sphere':
    n=5000
    X = half_sphere(n)
else:
    n=5000
    X,_ = datasets.make_swiss_roll(n_samples=n)

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

dp = 0
idx = np.argmin(depth)
path = [idx]
while dp < 1:
    xnId = W[idx,:].nonzero()[1]
    wnId = depth[xnId]
    wnMx = np.argmax(wnId)
    idx = xnId[wnMx]
    path += [idx]
    dp = depth[idx]

fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.scatter(X[:,0], X[:,1], X[:,2], s=size, c=depth,zorder=1)
if dataset=='swissroll':
    ax.view_init(4,-77)

if dataset != 'helix':
    for i in range(len(path)-1):
        x1 = X[path[i],:]
        x2 = X[path[i+1],:]
        ax.plot3D([x1[0],x2[0]],[x1[1],x2[1]],[x1[2],x2[2]],c='r',zorder=2,linewidth=2)
     
plt.axis('off')
if dataset == 'sphere':
    ax.view_init(29,-160)

plt.savefig('figures/depth_'+dataset+'.pdf')
plt.savefig('figures/depth_'+dataset+'.png',dpi=300)







