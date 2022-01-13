import graphlearningv2 as gl

dataset = 'cifar'
metric = 'aet' 
k = 10 #Number of neighbors

#Accuracy filename tags
tag = dataset + '_' + metric + '_k%d'%k

#Build a 10 nearest neighbor graph on the dataset
W = gl.weightmatrix.knn(dataset, k, metric=metric)
D = gl.weightmatrix.knn(dataset, 30, metric=metric, kernel='distance', symmetrize=False)

#Load data and labels
labels = gl.datasets.load(dataset, metric=metric, labels_only=True)

#Load existing trainsets
trainsets = gl.trainsets.load(dataset)

#List of models
model_list = [gl.ssl.laplace(W),
              gl.ssl.poisson(W),
              gl.ssl.peikonal(W, D=D, p=1, alpha=0, class_priors=gl.utils.class_priors(labels)),
              gl.ssl.peikonal(W, D=D, p=1, alpha=1, class_priors=gl.utils.class_priors(labels)),
              gl.ssl.peikonal(W, D=D, p=1, alpha=2, class_priors=gl.utils.class_priors(labels)),
              gl.ssl.peikonal(W, D=D, p=1, alpha=3, class_priors=gl.utils.class_priors(labels)),
              gl.ssl.peikonal(W, D=D, p=1, alpha=0.5, class_priors=gl.utils.class_priors(labels)),
              gl.ssl.graph_nearest_neighbor(W, D=D, alpha=0, class_priors=gl.utils.class_priors(labels)),
              gl.ssl.graph_nearest_neighbor(W, D=D, alpha=0.5, class_priors=gl.utils.class_priors(labels)),
              gl.ssl.graph_nearest_neighbor(W, D=D, alpha=1, class_priors=gl.utils.class_priors(labels)),
              gl.ssl.graph_nearest_neighbor(W, D=D, alpha=2, class_priors=gl.utils.class_priors(labels)),
              gl.ssl.graph_nearest_neighbor(W, D=D, alpha=3, class_priors=gl.utils.class_priors(labels))]


#Run experiments 
for model in model_list:
    model.ssl_trials(trainsets, labels, num_cores=20, tag=tag, num_trials=60)

gl.ssl.accuracy_table(model_list, tag=tag, savefile='tables/SSL_'+dataset+'.tex', title="SSL Comparison: "+dataset,append=False)


