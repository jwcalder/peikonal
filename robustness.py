import numpy as np
import graphlearning as gl
import matplotlib.pyplot as plt
from utils import peikonal_depth
import sys
from mayavi import mlab

size = 10
mult = 20
label_color = 'w'

marker = 's'
frac=0.05
seed=0
alpha=2
n=20000
eps = 0.05
p = 1

np.random.seed(seed)
X = gl.utils.rand_ball(n,2)
Tri = gl.utils.mesh(X)
Wc = gl.weightmatrix.epsilon_ball(X,eps,kernel='uniform')
if not gl.graph(Wc).isconnected():
    sys.exit('Graph is not connected')

bdy = np.linalg.norm(X,axis=1) > 1-eps

Wc = Wc.tolil()
for num_corr in [0,10,20,50,100,200,500,1000]:
    
    W = Wc.copy()
    if num_corr > 0:
        p1 = np.random.choice(n,size=num_corr)
        p2 = np.random.choice(n,size=num_corr)
        for j in range(len(p1)): 
            W[p1[j],p2[j]]+=1; W[p2[j],p1[j]]+=1
    G = gl.graph(W)

    #deg = G.degree_vector()
    depth = G.peikonal(bdy,p=p)
    plt.figure()
    plt.scatter(X[:,0],X[:,1], s=size, c=depth)
    plt.scatter(X[bdy,0],X[bdy,1], s=size, c='red')
    plt.axis('off')
    plt.axis('square')
    plt.savefig('figures/robustness_peikonal_%d.pdf'%num_corr)
    plt.savefig('figures/robustness_peikonal_%d.png'%num_corr, dpi=300)

    mlab.figure(bgcolor=(1,1,1))
    mlab.triangular_mesh(X[:,0],X[:,1],depth*5/6,Tri)
    mlab.savefig('figures/robustness_peikonal_%d_mlab.png'%num_corr,size=(100,100))
    mlab.close()

    W = Wc.copy()
    if num_corr > 0:
        for j in range(len(p1)): 
            W[p1[j],p2[j]]+=1; W[p2[j],p1[j]]+=1
    G = gl.graph(W)

    depth = G.dijkstra(bdy,f=1)
    plt.figure()
    plt.scatter(X[:,0],X[:,1], s=size, c=depth)
    plt.scatter(X[bdy,0],X[bdy,1], s=size, c='red')
    plt.axis('off')
    plt.axis('square')
    plt.savefig('figures/robustness_dijkstra_%d.pdf'%num_corr)
    plt.savefig('figures/robustness_dijkstra_%d.png'%num_corr, dpi=300)

    mlab.figure(bgcolor=(1,1,1))
    mlab.triangular_mesh(X[:,0],X[:,1],depth/13,Tri)
    mlab.savefig('figures/robustness_dijkstra_%d_mlab.png'%num_corr,size=(100,100))
    mlab.close()


