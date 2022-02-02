import graphlearning as gl

metric = 'vae' 
k = 10 #Number of neighbors

for dataset in ['mnist', 'fashionmnist', 'cifar']:

    if dataset == 'cifar':
        metric = 'aet'

    #Accuracy filename tags
    tag = dataset + '_' + metric + '_k%d'%k

    #Build a 10 nearest neighbor graph on the dataset
    W = gl.weightmatrix.knn(dataset, k, metric=metric)
    D = gl.weightmatrix.knn(dataset, 30, metric=metric, kernel='distance', symmetrize=False)

    #Load data and labels
    labels = gl.datasets.load(dataset, metric=metric, labels_only=True)
    priors = gl.utils.class_priors(labels)

    #Load existing trainsets
    trainsets = gl.trainsets.load(dataset)

    #List of models
    model_list = [gl.ssl.laplace(W),gl.ssl.poisson(W)]

    for alpha in range(4):
        model_list += [gl.ssl.peikonal(W, D=D, p=1, alpha=alpha, class_priors=priors)]

    for alpha in range(4):
        model_list += [gl.ssl.graph_nearest_neighbor(W, D=D, alpha=alpha, class_priors=priors)]

    #Run experiments 
    for model in model_list:
        model.ssl_trials(trainsets, labels, num_cores=20, tag=tag, num_trials=60)

    gl.ssl.accuracy_table(model_list, tag=tag, savefile='tables/SSL_'+dataset+'.tex', title="SSL Comparison: "+dataset,append=False)


