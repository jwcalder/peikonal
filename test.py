import graphlearning as gl

W = gl.weightmatrix.knn('mnist',20,metric='vae')
D = gl.weightmatrix.knn('mnist', 30, metric='vae', kernel='distance', symmetrize=False)
labels = gl.datasets.load('mnist',labels_only=True) 
priors = gl.utils.class_priors(labels)

train_ind = gl.trainsets.load('mnist')[0]
train_labels = labels[train_ind]

#pred_labels = gl.ssl.poisson(W).fit_predict(train_ind,train_labels)
pred_labels = gl.ssl.peikonal(W, D=D, p=0.25, alpha=3, class_priors=priors).fit_predict(train_ind,train_labels)
#WD = gl.weightmatrix.knn('mnist', 20, metric='vae', kernel='distance')
#pred_labels = gl.ssl.graph_nearest_neighbor(W, D=D, alpha=3, class_priors=priors).fit_predict(train_ind,train_labels)

acc = gl.ssl.ssl_accuracy(pred_labels,labels,len(train_ind))
print(acc)

