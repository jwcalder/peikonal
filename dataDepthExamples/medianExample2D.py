import numpy as np
import graphlearning as gl
import matplotlib.pyplot as plt
import sklearn.datasets as datasets
import random
import time


# non-unoform sampling
def nonUnifSample(n,dataID):
    X = np.zeros((n,2))
    m1 = [0, 0]            
    c1 = [[1, 0], [0, 1]] 
    if (dataID == 1):
        X[0:n,0], X[0:n,1] = np.random.multivariate_normal(m1, c1, n).T
    if (dataID == 2):
        ng = int(n/2)
        m2 = [2, 2]               
        c2 = [[0.2, 0], [0, .2]] 
        X[0:ng,0], X[0:ng,1] = np.random.multivariate_normal(m1, c1, ng).T
        X[ng:n,0], X[ng:n,1] = np.random.multivariate_normal(m2, c2, n-ng).T
    if (dataID == 3):
        X,labels = datasets.make_moons(n_samples=n,noise=0.1)
    return X







p = 1
alpha = -0.5

dataID = 1 
            # if numGroup = 1 then multivariate Gaussian 
            # if numGroup = 2 then mixture of multivariate Gausian 
            # if numGroup = 3 then two moon-data set 
n = 1000





dataset = nonUnifSample(n,dataID)
k = 10
W = gl.weightmatrix.knn(dataset,k)
# W = gl.weightmatrix.epsilon_ball(dataset,epsilon=0.1)
G = gl.graph(W)
D = gl.weightmatrix.knn(dataset, k, kernel='distance')



n = W.shape[0]
u = np.zeros(n)
f = np.ones(n)
d = D.max(axis=1).toarray().flatten() #distance to furtherest neighbor
f = (d/np.max(d))**alpha
randomSetSize = int(np.round(0.01*n))
randomSetSize = 10
print("--- %s random set size ---\n" % randomSetSize)


peikonal_median_sum = []
peikonal_median_max = []
ind_boundary_points = []
UT = np.zeros((randomSetSize,n))
start_time = time.time()
for ind in range(randomSetSize):
    ind_boundary = random.sample(range(n), 1)
    bdy_set = []
    x = dataset[:,0]
    bdy_set = (x == dataset[ind_boundary,0])
    ind_boundary_points.append(ind_boundary[0])
    u = G.peikonal(bdy_set,p=p,f=f)
    UT[ind,:] = u
    peikonal_median_sum.append(np.sum(u))
    peikonal_median_max.append(np.max(u))
    x = ind % 5
    if (x == 0):
        print("--- %s seconds ---\n" % (time.time() - start_time))
        
index_median_sum = np.argmin(peikonal_median_sum)
index_median_max = np.argmin(peikonal_median_max)
fig = plt.figure(figsize=(5,5))
ax = fig.add_subplot(1, 1, 1)
plt.scatter(dataset[:,0], dataset[:,1], color='black',s=0.5, marker='.')
plt.scatter(dataset[ind_boundary_points,0], dataset[ind_boundary_points,1], color='red',s=20, marker='.')
kms = ind_boundary_points[index_median_sum]
plt.scatter(dataset[kms,0], dataset[kms,1], color='blue',s=100, marker='*')
kmm = ind_boundary_points[index_median_max]
plt.scatter(dataset[kmm,0], dataset[kmm,1], color='green',s=200, marker='.')
ax.legend(['data points', 'random (boundary) points', 'median (sum criteria)','median (max criteria)'])
plt.axis('equal')
plt.show()




