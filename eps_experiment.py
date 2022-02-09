import numpy as np
import graphlearning as gl
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import sys

ecolor = 0.4
linewidth = 0
n=20000
delta = 0.1
X = gl.utils.rand_ball(n,2)
X[0,:] = [0,0]

for p in [1,2,4]:
    for eps in [0.03,0.06,0.09]:
        W = gl.weightmatrix.epsilon_ball(X,eps,kernel='uniform')
        G = gl.graph(W)
        d = G.degree_vector()
        print(p,eps,np.mean(d))
        if not G.isconnected():
            sys.exit('Graph is not connected')

        depth = G.peikonal([0],p=p)

        #Plots
        Tri = gl.utils.mesh(X)
        fig = plt.figure()
        ax = fig.add_subplot(1,1,1,projection='3d')
        ax.plot_trisurf(X[:,0],X[:,1],depth,triangles=Tri,color='gray',edgecolors=(ecolor,ecolor,ecolor),linewidth=0.1,antialiased=True)
        ax.view_init(elev=-160,azim=-45)
        plt.axis('off')
        plt.tight_layout()
        plt.savefig('figures/cone_eps_%.2f_p_%d.png'%(eps,p),dpi=300)


