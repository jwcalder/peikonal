import utils
import matplotlib.pyplot as plt

X = utils.half_sphere(1000)
fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.scatter(X[:,0], X[:,1], X[:,2], marker='o')
plt.axis('off')

plt.show()
