import numpy as np
import graphlearning as gl
import matplotlib.pyplot as plt
from utils import peikonal_depth
import sys

size = 3
mult = 20
label_color = 'w'

marker = 's'
frac=0.05
seed=0
alpha=2
n=5000
eps = 0.05
delta = 0.1
p = 1

np.random.seed(seed)
X = gl.utils.rand_ball(n,2)
X[0,:] = [0,0]
Wc = gl.weightmatrix.epsilon_ball(X,eps,kernel='uniform')
if not gl.graph(Wc).isconnected():
    sys.exit('Graph is not connected')


Wc = Wc.tolil()
for num_corr in [0,1,5,10,50]:
    
    W = Wc.copy()
    if num_corr > 0:
        points = np.random.choice(n,size=num_corr)
        for j in points: 
            W[0,j]=delta; W[j,0]=delta

    G = gl.graph(W)
    deg = G.degree_vector()
    depth = G.peikonal([0],f=deg,p=p)
    depth = depth/np.max(depth); depth = 1-depth
    plt.figure()
    plt.scatter(X[:,0],X[:,1], s=size, c=depth)
    plt.axis('off')
    plt.axis('square')
    if num_corr > 0:
        for j in points:
            plt.scatter(X[j,0],X[j,1], c=label_color, marker=marker, s=mult*size, edgecolors='black', linewidth=2)
    plt.savefig('figures/robustness_peikonal_%d.pdf'%num_corr)

    depth = G.dijkstra([0],f=1)
    depth = depth/np.max(depth); depth = 1-depth
    plt.figure()
    plt.scatter(X[:,0],X[:,1], s=size, c=depth)
    plt.axis('off')
    plt.axis('square')
    if num_corr > 0:
        for j in points:
            plt.scatter(X[j,0],X[j,1], c=label_color, marker=marker, s=mult*size, edgecolors='black', linewidth=2)
    plt.savefig('figures/robustness_dijkstra_%d.pdf'%num_corr)


#plt.figure()
#plt.scatter(X[:,0],X[:,1], s=size, c=depth)
#plt.scatter(X[medians[1],0],X[medians[1],1], c=label_color, marker='s', s=10*size, edgecolors='black', linewidth=2)
#plt.scatter(X[medians[0],0],X[medians[0],1], c=label_color, marker='v', s=15*size, edgecolors='black', linewidth=2)
#plt.scatter(X[medians[2],0],X[medians[2],1], c=label_color, marker='^', s=15*size, edgecolors='black', linewidth=2)
#plt.axis('off')
#plt.axis('square')
#plt.savefig('figures/depth_'+dataset+'.pdf')


