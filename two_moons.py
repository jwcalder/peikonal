import numpy as np
import graphlearning as gl
import matplotlib.pyplot as plt
import sklearn.datasets as datasets

X,labels = datasets.make_moons(n_samples=500,noise=0.1,random_state=0)
W = gl.weightmatrix.epsilon_ball(X,0.5)

for seed in [3,15]:
    train_ind = gl.trainsets.generate(labels, rate=1, seed=seed) 
    train_labels = labels[train_ind]

    class_priors = gl.utils.class_priors(labels)
    model = gl.ssl.peikonal(W, class_priors=class_priors)
    pred_labels = model.fit_predict(train_ind, train_labels)
    pred_labels_wo = model.predict(ignore_class_priors=True)

    accuracy = gl.ssl.ssl_accuracy(pred_labels, labels, len(train_ind))   
    print("Accuracy With Priors: %.2f%%"%accuracy)
    accuracy_wo = gl.ssl.ssl_accuracy(pred_labels_wo, labels, len(train_ind))   
    print("Accuracy Without Priors: %.2f%%"%accuracy_wo)

    plt.figure()
    plt.scatter(X[:,0],X[:,1], c=pred_labels)
    plt.scatter(X[train_ind,0],X[train_ind,1], c='r')
    plt.axis('off')
    plt.savefig('figures/twomoons_priors_%.2f_%d.pdf'%(accuracy,seed))

    plt.figure()
    plt.scatter(X[:,0],X[:,1], c=pred_labels_wo)
    plt.scatter(X[train_ind,0],X[train_ind,1], c='r')
    plt.axis('off')
    plt.savefig('figures/twomoons_nopriors%.2f_%d.pdf'%(accuracy_wo,seed))
