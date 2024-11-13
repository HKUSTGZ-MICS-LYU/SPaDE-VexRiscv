from utils import *


n_inits = 20
dim = 2

init_points = np.array(latin_hypercube(n_inits, dim)).T
uniform = [np.random.uniform(0, 1, n_inits), np.random.uniform(0, 1, n_inits)]

# print(init_points.T)

import matplotlib.pyplot as plt

plt.scatter(init_points[0], init_points[1])
plt.scatter(uniform[0], uniform[1])
plt.hlines(0.5, 0, 1)
plt.vlines(0.5, 0, 1)
plt.show()