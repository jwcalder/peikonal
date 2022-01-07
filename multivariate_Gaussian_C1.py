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



# non-unoform sampling: multivariate Gaussioans
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



n = 10000
X = nonUnifSample(n)

ind_boundary = random.sample(range(n), 1) # select one boundary point randomly


eps = 1
W = gl.eps_weight_matrix(X,eps)
if not gl.isconnected(W):
    print("Graph not connected")
I,J,V = sparse.find(W)  #Indices of nonzero entries

p = 1
alpha = 3
k = 1
n = W.shape[0]
u = np.zeros((n,k))
f = np.ones(n)
d = gl.degrees(W) 
f = (d/np.max(d))**alpha


# cpeikonal
gb = np.zeros((len(ind_boundary),))
start_time = time.time()
u = gl.cpeikonal(W, ind_boundary, p=p, f=f, g=gb, max_num_it=1e5, converg_tol=1e-6, num_bisection_it=30, prog=False)
res = residualCal(W,ind_boundary,u,f,p)
print('Residual for cpeikonal: ',res)
print("--- %s seconds ---\n" % (time.time() - start_time))



plt.scatter(X[0:n,0], X[0:n,1], color='black',s=0.5, marker='.')
plt.scatter(X[ind_boundary,0], X[ind_boundary,1], color='red',s=100, marker='.')
plt.axis('equal')
plt.show()

fig = plt.figure(figsize=(5,5))
ax = fig.add_subplot(1, 1, 1)
ax.scatter(X[:,0],X[:,1], c=u,s=1)
plt.show()



plt.ion()
plt.close('all')
ecolor = 0.4
linewidth = 0
aa = False
my_cmap = plt.get_cmap('jet')

#Plots
Tri = gl.mesh(X)
fig = plt.figure(figsize=(5,5))
ax = fig.add_subplot(1,1,1,projection='3d')
ax.plot_trisurf(X[:,0],X[:,1],u,triangles = Tri,cmap=my_cmap,edgecolors=(ecolor,ecolor,ecolor),linewidth=0.1,antialiased=aa)
ax.plot_trisurf(X[:,0],X[:,1],u,triangles = Tri,cmap=my_cmap,edgecolors=(ecolor,ecolor,ecolor),linewidth=linewidth,antialiased=aa)
# ax.view_init(elev=-160,azim=-45)
plt.axis('off')
plt.tight_layout()
plt.savefig('multivariate_Gaussian.eps')
plt.show()






