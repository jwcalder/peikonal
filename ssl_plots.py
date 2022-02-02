import graphlearning as gl
import matplotlib.pyplot as plt
import matplotlib
import numpy as np

#General plot settings
legend_fontsize = 12
label_fontsize = 16
fontsize = 16
matplotlib.rcParams.update({'font.size': fontsize})
styles = ['^b-','or-','dg-','sk-','pm-','xc-','*y-']

colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728','#9467bd', '#8c564b', '#e377c2', '#7f7f7f','#bcbd22', '#17becf']
markers = ['^','o','d','s','p','x','*']

metric = 'vae' 
k = 10 #Number of neighbors

acc_alpha = {}

for dataset in ['mnist','fashionmnist','cifar']:

    plt.figure()

    if dataset == 'cifar':
        metric = 'aet'

    #Accuracy filename tags
    tag = dataset + '_' + metric + '_k%d'%k

    num_train,acc_mean,acc_stddev,num_trials = gl.ssl.poisson().trials_statistics(tag=tag)
    plt.plot(num_train,acc_mean,c=colors[0],marker=markers[0],label='Poisson Learning')

    c=1
    acc_alpha[dataset] = np.zeros((2,4))
    for alpha in range(4):
        num_train,acc_mean,acc_stddev,num_trials = gl.ssl.peikonal(p=1, alpha=alpha, class_priors=np.ones(10)).trials_statistics(tag=tag)
        acc_alpha[dataset][0,alpha] = acc_mean[0,0]
        acc_alpha[dataset][1,alpha] = acc_mean[0,1]
        if alpha in [0,3]:
            plt.plot(num_train,acc_mean[:,0],c=colors[c],marker=markers[c],linestyle='-',label='$p$-eikonal w/o priors ($p=1$,$\\alpha=%d$)'%alpha)
            plt.plot(num_train,acc_mean[:,1],c=colors[c],marker=markers[c],linestyle='--',label='p-eikonal with priors ($p=1$,$\\alpha=%d$)'%alpha)
            c += 1

    for alpha in [3]:
        num_train,acc_mean,acc_stddev,num_trials = gl.ssl.graph_nearest_neighbor(alpha=alpha, class_priors=np.ones(10)).trials_statistics(tag=tag)
    #    plt.plot(num_train,acc_mean[:,0],c=colors[c],marker=markers[c],linestyle='-',label='GNN w/o priors ($p=1$,$\\alpha=%d$)'%alpha)
        plt.plot(num_train,acc_mean[:,1],c=colors[c],marker=markers[c],linestyle='--',label='GNN with priors ($p=1$,$\\alpha=%d$)'%alpha)
        c += 1


    #for alpha in range(4):
    #    model_list += [gl.ssl.graph_nearest_neighbor(W, D=D, alpha=alpha, class_priors=priors)]

    if dataset == 'mnist':
        plt.ylim((50,100))
    if dataset == 'fashionmnist':
        plt.ylim((40,72))
    if dataset == 'cifar':
        plt.ylim((20,55))

    plt.xlabel('Number of labels',fontsize=label_fontsize)
    plt.ylabel('Accuracy (%)',fontsize=label_fontsize)
    plt.legend(loc='lower right',fontsize=legend_fontsize)
    plt.tight_layout()
    plt.grid(True)

    #Save figures
    plt.savefig('./figures/ssl_' + dataset + '.pdf')


plt.figure()
alpha = np.arange(4)
c=0
for dataset in ['mnist','fashionmnist','cifar']:
    name = dataset
    name = name.replace('mnist','MNIST')
    name = name.replace('fashion','Fashion')
    name = name.replace('cifar','CIFAR-10')
    plt.plot(alpha,acc_alpha[dataset][0,:] - acc_alpha[dataset][0,0],c=colors[c],marker=markers[c],linestyle='-',label=name+' w/o priors')
    plt.plot(alpha,acc_alpha[dataset][1,:] - acc_alpha[dataset][1,0],c=colors[c],marker=markers[c],linestyle='--',label=name+' with priors')
    c += 1

plt.xlabel('$\\alpha$',fontsize=label_fontsize)
plt.ylabel('Change in accuracy (%)',fontsize=label_fontsize)
plt.legend(loc='lower left',fontsize=legend_fontsize)
plt.tight_layout()
plt.grid(True)

plt.savefig('./figures/ssl_alphacomp.pdf')




















