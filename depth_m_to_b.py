import numpy as np
import graphlearning as gl
import matplotlib.pyplot as plt
import sklearn.datasets as datasets
from utils import peikonal_depth
import sys

from collections import defaultdict
class Graph():
    def __init__(self):
        """
        self.edges is a dict of all possible next nodes
        e.g. {'X': ['A', 'B', 'C', 'E'], ...}
        self.weights has all the weights between two nodes,
        with the two nodes as a tuple as the key
        e.g. {('X', 'A'): 7, ('X', 'B'): 2, ...}
        """
        self.edges = defaultdict(list)
        self.weights = {}
    
    def add_edge(self, from_node, to_node, weight):
        # Note: assumes edges are bi-directional
        self.edges[from_node].append(to_node)
        self.edges[to_node].append(from_node)
        self.weights[(from_node, to_node)] = weight
        self.weights[(to_node, from_node)] = weight


def dijsktra(graph, initial, end):
    # shortest paths is a dict of nodes
    # whose value is a tuple of (previous node, weight)
    shortest_paths = {initial: (None, 0)}
    current_node = initial
    visited = set()
    
    while current_node != end:
        visited.add(current_node)
        destinations = graph.edges[current_node]
        weight_to_current_node = shortest_paths[current_node][1]

        for next_node in destinations:
            weight = graph.weights[(current_node, next_node)] + weight_to_current_node
            if next_node not in shortest_paths:
                shortest_paths[next_node] = (current_node, weight)
            else:
                current_shortest_weight = shortest_paths[next_node][1]
                if current_shortest_weight > weight:
                    shortest_paths[next_node] = (current_node, weight)
        
        next_destinations = {node: shortest_paths[node] for node in shortest_paths if node not in visited}
        if not next_destinations:
            return "Route Not Possible"
        # next node is the destination with the lowest weight
        current_node = min(next_destinations, key=lambda k: next_destinations[k][1])
    
    # Work back through destinations in shortest path
    path = []
    while current_node is not None:
        path.append(current_node)
        next_node = shortest_paths[current_node][0]
        current_node = next_node
    # Reverse path
    path = path[::-1]
    return path



dataset = 'mnist'
k = 10

frac=0.05
seed=0
alpha=1
eps = 1

X, labels = gl.datasets.load(dataset)

label = 4

print("Digit %d..."%label)

#Subset labels
X_sub = X[labels==label,:]
num = X_sub.shape[0]

#KNN search
knn_ind, knn_dist = gl.weightmatrix.knnsearch(X_sub,20*k)
W = gl.weightmatrix.knn(X_sub,10,knn_data=(knn_ind,knn_dist))
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

g = Graph()
neigh_num = 5
for i in range(num):
    for j in range(neigh_num):
        g.add_edge(i, knn_ind[i,j+1], knn_dist[i,j+1]);

pathID = dijsktra(g, m_indx, b_indx)


fig = plt.figure(figsize=(10, 10))

columns = len(pathID)
rows = 1

for j in range(1, columns*rows +1):
    i = pathID[j-1]
    img = X_sub[i,:]
    m = int(np.sqrt(img.shape[0]))
    img = np.reshape(img,(m,m))
    if dataset.lower() == 'mnist':
        img = np.transpose(img)
    fig.add_subplot(rows, columns, j)
    plt.imshow(img,cmap='gray')
    plt.axis('off')

plt.savefig(dataset+'_peikonal_path.pdf')
plt.show()
    