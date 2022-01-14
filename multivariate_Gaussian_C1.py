import numpy as np
import graphlearning as gl
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator
import matplotlib.gridspec as gridspec
from mpl_toolkits import mplot3d
import scipy.sparse as sparse
import sys
import random
import time


# non-unoform sampling: superposition of two Gaussioans
def nonUnifSample(n):
    
    X = np.zeros((n,2))
    ng = int(round(n/2))

    m1 = [0, 0]            # mean vector for G1
    c1 = [[1, 0], [0, 1]]  # diagonal covariance for G1
    
    m2 = [2, 2]               # mean vector for G2
    c2 = [[0.4, 0], [0, .2]]  # diagonal covariance for G2
    
    ng = int(n/2)
    X[0:ng,0], X[0:ng,1] = np.random.multivariate_normal(m1, c1, ng).T
    X[ng:n,0], X[ng:n,1] = np.random.multivariate_normal(m2, c2, n-ng).T

    return X


# calculate residual 
def residualCal(W,ind,u,f,p):
    n = W.shape[0]
    I,J,V = sparse.find(W)
    G = sparse.coo_matrix((V*np.maximum(u[I]-u[J],0)**p, (I,J)),shape=(n,n))
    F = f - G@np.ones((n,))
    F[ind] = 0
    return np.max(np.absolute(F))


n = 5000
X = nonUnifSample(n)

eps = 1
W = gl.eps_weight_matrix(X,eps)
if not gl.isconnected(W):
    print("Graph not connected")
I,J,D = sparse.find(W)  #Indices of nonzero entries

p = 1
alpha = 3
k = 1
n = W.shape[0]
u = np.zeros((n,k))
f = np.ones(n)
d = gl.degrees(W) 
f = (d/np.max(d))**alpha

fig = plt.figure(figsize=(5,5))
ax = fig.add_subplot(1, 1, 1)
plt.scatter(X[0:n,0], X[0:n,1], color='black',s=0.5, marker='.')
ax.legend(['data set'])
plt.axis('equal')
plt.show()


randomSetSize = int(np.round(0.1*n))
print("--- %s random set size ---\n" % randomSetSize)

peikonal_median_sum = []
peikonal_median_max = []
peikonal_median_med = []

ind_boundary_points = []

start_time = time.time()
for ind in range(randomSetSize):
    ind_boundary = random.sample(range(n), 1)
    ind_boundary_points.append(ind_boundary[0])
    gb = np.zeros((len(ind_boundary),))
    u = gl.cpeikonal(W, ind_boundary, p=p, f=f, g=gb, max_num_it=1e5, converg_tol=1e-5, num_bisection_it=30, prog=False)
    
    peikonal_median_sum.append(np.sum(u))
    peikonal_median_max.append(np.max(u))
    peikonal_median_med.append(np.median(u))
    
    x = ind % 100
    if (x == 0):
        print("--- %s seconds ---\n" % (time.time() - start_time))


index_median_sum = np.argmin(peikonal_media_sum)
index_median_max = np.argmin(peikonal_media_max)
index_median_med = np.argmin(peikonal_media_med)


fig = plt.figure(figsize=(5,5))
ax = fig.add_subplot(1, 1, 1)
plt.scatter(X[0:n,0], X[0:n,1], color='black',s=0.5, marker='.')
plt.scatter(X[ind_boundary_points,0], X[ind_boundary_points,1], color='red',s=1, marker='.')
plt.scatter(X[index_median_sum,0], X[index_median_sum,1], color='blue',s=500, marker='.')
plt.scatter(X[index_median_max,0], X[index_median_max,1], color='darkgreen',s=500, marker='.')
plt.scatter(X[index_median_med,0], X[index_median_med,1], color='maroon',s=500, marker='.')
ax.legend(['data points', 'random points', 'sum criteria','max criteria','median criteria'])
plt.axis('equal')
plt.show()
