import numpy as np
import graphlearning as gl
import matplotlib.pyplot as plt
import sklearn.datasets as datasets

size = 30
vmin = 0
vmax = 1.5
#cmap = 'Paired'
#cmap = 'Dark2'
#cmap = 'copper'
#cmap = 'winter'
cmap = 'viridis'
label_color = 'r'
marker = "^"

X,labels = datasets.make_moons(n_samples=500,noise=0.1,random_state=0)
W = gl.weightmatrix.epsilon_ball(X,0.5)
class_priors = gl.utils.class_priors(labels)

for alpha in [-1,0,1]:

    model = gl.ssl.peikonal(W, class_priors=class_priors, eps_ball_graph=True, alpha=alpha)
    for seed in [3,15]:
        train_ind = gl.trainsets.generate(labels, rate=1, seed=seed) 
        train_labels = labels[train_ind]

        pred_labels = model.fit_predict(train_ind, train_labels)
        pred_labels_wo = model.predict(ignore_class_priors=True)

        accuracy = gl.ssl.ssl_accuracy(pred_labels, labels, len(train_ind))   
        print("Accuracy With Priors: %.2f%%"%accuracy)
        accuracy_wo = gl.ssl.ssl_accuracy(pred_labels_wo, labels, len(train_ind))   
        print("Accuracy Without Priors: %.2f%%"%accuracy_wo)

        plt.figure()
        plt.scatter(X[:,0],X[:,1], c=pred_labels, s=size, cmap=cmap, vmin=vmin, vmax=vmax)
        plt.scatter(X[train_ind,0],X[train_ind,1], c=label_color, s=10*size, marker=marker, edgecolors='black')
        plt.axis('off')
        plt.savefig('figures/twomoons_priors_%.2f_%d_%d.pdf'%(accuracy,seed,alpha))

        plt.figure()
        plt.scatter(X[:,0],X[:,1], c=pred_labels_wo, s=size, cmap=cmap, vmin=vmin, vmax=vmax)
        plt.scatter(X[train_ind,0],X[train_ind,1], c=label_color, s=10*size, marker=marker, edgecolors='black')
        plt.axis('off')
        plt.savefig('figures/twomoons_nopriors%.2f_%d_%d.pdf'%(accuracy_wo,seed,alpha))
