import numpy as np
import graphlearning as gl
import matplotlib.pyplot as plt
import sklearn.datasets as datasets
from utils import peikonal_depth
import sys

k = 20
frac=0.05
alpha=2

#Plotting
numw = 16
numh = 10


for dataset in ['mnist','fashionmnist']:

    f_bdy, axarr_bdy = plt.subplots(numh,numw,gridspec_kw={'wspace':0.1,'hspace':0.1})
    f_peikonal_median, axarr_peikonal = plt.subplots(numh,numw,gridspec_kw={'wspace':0.1,'hspace':0.1})
    f_bdy.suptitle('Boundary images')
    f_peikonal_median.suptitle('peikonal Median images')


    X, labels = gl.datasets.load(dataset)
    pathID = np.zeros((10,200))

    for label in range(10):
        print("Digit %d..."%label)

        #Subset labels
        X_sub = X[labels==label,:]
        num = X_sub.shape[0]

        #KNN search
        knn_ind, knn_dist = gl.weightmatrix.knnsearch(X_sub,30)
        W = gl.weightmatrix.knn(X_sub,k,knn_data=(knn_ind,knn_dist))
        G = gl.graph(W)
        if not G.isconnected():
            sys.exit('Graph is not connected')
        d = np.max(knn_dist,axis=1)
        kde = (d/d.max())**(-1)
        
        median, depth = peikonal_depth(G, kde, frac, alpha)

        depth = depth/np.max(depth)
        depth = 1-depth
        
        ind_boundary = np.argsort(+depth)
        ind_peikonal = np.argsort(-depth)
        
        b_indx = ind_boundary[0]
        m_indx = ind_peikonal[0] 
        
        
        W = W.tocsr()
        neigh_num = 20
        b_indx_up = b_indx
        pathID[label,0] = b_indx
        maxItt = 1e2
        dp = 0
        cnt = 0
        while (dp < 1) and (cnt < maxItt):
            cnt += 1
            #xnId = knn_ind[b_indx_up,1:neigh_num]
            xnId = W[b_indx_up,:].nonzero()[1]
            wnId = depth[xnId]
            wnMx = np.argmax(wnId)
            b_indx_up = xnId[wnMx]
            pathID[label,cnt] = b_indx_up
            dp = depth[b_indx_up]

        print(dp)
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



    f_bdy.savefig('figures/'+dataset+'_boundary.png')
    f_peikonal_median.savefig('figures/'+dataset+'_peikonal_median.png')


    # path from boundary to median plots
    columns = 1
    for i in range(10):
        x = pathID[i,:]
        indx = np.nonzero(x)
        digitIndx = indx[0]
        lp = len(digitIndx)
        if (lp > columns):
            columns = lp
            
    #Plotting
    numw = columns
    numh = 10

    f_peikonal_path, axarr_peikonal_path = plt.subplots(numh,numw,gridspec_kw={'wspace':0.1,'hspace':0.1})
    f_peikonal_path.suptitle('peikonal boundary to median images')


    img = X_sub[0,:]
    lm = img.shape[0]
    for label in range(10):
        x = pathID[label,:]
        indx = np.nonzero(x)
        digitIndx = indx[0]
        lp = len(digitIndx)
        path = pathID[label,digitIndx]
        
        X_sub = X[labels==label,:]

        #Visualization
        for j in range(numw):
            if (j < lp):
                i = int(path[j])
                img = X_sub[i,:]
                m = int(np.sqrt(img.shape[0]))
                img = np.reshape(img,(m,m))
                if dataset.lower() == 'mnist':
                    img = np.transpose(img)
                axarr_peikonal_path[label,j].imshow(img,cmap='gray')
            else:
                img = np.ones(lm)
                m = int(np.sqrt(img.shape[0]))
                img = np.reshape(img,(m,m))
                axarr_peikonal_path[label,j].imshow(img,cmap='binary')
            axarr_peikonal_path[label,j].axis('off')
            axarr_peikonal_path[label,j].set_aspect('equal')
            
    f_peikonal_path.savefig('figures/'+dataset+'_peikonal_path.png')

plt.show()


