import graphlearning as gl
import numpy as np

def half_sphere(n):

    X = gl.utils.rand_ball(3*n,3)
    X = X/np.linalg.norm(X,axis=1)[:,None]
    X = X[X[:,2]>0]
    X = X[:n,:]
    return X

def half_moon(n, shift=0.4):

    X = np.zeros((n,2))
    num_pts = 0

    while num_pts < n:
        Z = gl.utils.rand_ball(n,2)
        d = np.linalg.norm(Z-[shift,0],axis=1)
        Y = Z[d>1 + shift/5,:]
        num = min(Y.shape[0],n-num_pts)
        X[num_pts:num_pts+num,:] = Y[:num,:]
        num_pts += num

    return X

def half_helix(n, noiseID=0):
    X = np.zeros((n,3))
    noiseVal = np.zeros((3,n))
    if (noiseID == 1):
        noiseVal[0,:] = np.random.normal(0, 0.05, n)
        noiseVal[1,:] = np.random.normal(0, 0.05, n)
        noiseVal[2,:] = np.random.normal(0, 0.05, n)
    t = 0
    dt = 2*np.pi/n
    num_pts = 0
    while num_pts < n:
        X[num_pts,0] = np.sin(t) + noiseVal[0,num_pts]
        X[num_pts,1] = np.cos(t) + noiseVal[1,num_pts]
        X[num_pts,2] = t + noiseVal[2,num_pts]
        t += dt
        num_pts += 1 
    return X


def peikonal_depth(G, kde, frac, alpha):

    n = G.num_nodes
    num_cand = int(frac*n)
    p = np.random.choice(n,size=num_cand)


    f = kde**(-alpha)
    score = np.zeros(num_cand)
    for i,ind in enumerate(p):
        u = G.peikonal([ind],f=f)
        score[i] = np.mean(u)

    j = np.argmin(score)
    median = p[j]
    depth = G.peikonal([median],f=f)

    return median, depth
