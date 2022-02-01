import numpy as np
import graphlearning as gl
import matplotlib.pyplot as plt
import sklearn.datasets as datasets
from utils import peikonal_depth
import sys


dataset = 'mnist'
k = 10

frac=0.05
seed=0
alpha=1
eps = 1

X, labels = gl.datasets.load(dataset)

#Plotting
numw = 16
numh = 10
f_bdy, axarr_bdy = plt.subplots(numh,numw,gridspec_kw={'wspace':0.1,'hspace':0.1})
f_peikonal_median, axarr_peikonal = plt.subplots(numh,numw,gridspec_kw={'wspace':0.1,'hspace':0.1})
f_bdy.suptitle('Boundary images')
f_peikonal_median.suptitle('p-Eikonal Median images')



for label in range(10):
    print("Digit %d..."%label)

    #Subset labels
    X_sub = X[labels==label,:]
    num = X_sub.shape[0]

    #KNN search
    knn_ind, knn_dist = gl.weightmatrix.knnsearch(X_sub,20*k)
    W = gl.weightmatrix.knn(X_sub,k,knn_data=(knn_ind,knn_dist))
    G = gl.graph(W)
    if not G.isconnected():
        sys.exit('Graph is not connected')
    d = np.max(knn_dist,axis=1)
    kde = (d/d.max())**(-1)
    
    alpha = 1
    median, depth = peikonal_depth(G, kde, frac, alpha)
    depth = depth/np.max(depth)
    depth = 1-depth
    
    ind_boundary = np.argsort(+depth)
    ind_peikonal = np.argsort(-depth)
    
    
    #Visualization
    for j in range(numw):
        
        img = X_sub[ind_boundary[j],:]
        m = int(np.sqrt(img.shape[0]))
        img = np.reshape(img,(m,m))
        if dataset.lower() == 'mnist':
            img = np.transpose(img)
        axarr_bdy[label,j].imshow(img,cmap='gray')
        axarr_bdy[label,j].axis('off')
        axarr_bdy[label,j].set_aspect('equal')

        img = X_sub[ind_peikonal[j],:]
        m = int(np.sqrt(img.shape[0]))
        img = np.reshape(img,(m,m))
        if dataset.lower() == 'mnist':
            img = np.transpose(img)
        axarr_peikonal[label,j].imshow(img,cmap='gray')
        axarr_peikonal[label,j].axis('off')
        axarr_peikonal[label,j].set_aspect('equal')




f_bdy.savefig(dataset+'_boundary.pdf')
f_peikonal_median.savefig(dataset+'_peikonal_median.pdf')
plt.show()
