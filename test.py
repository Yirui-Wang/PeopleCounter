from scipy.spatial import distance as dist
import numpy as np

a = np.array([[4.1], [6.3], [-1], [0.5]])
b = np.array([[3.8], [9.0], [7.7], [2.3]])

D = dist.cdist(a, b)
print(D)
rows = D.min(axis=1).argsort()
print(rows)
cols = D.argmin(axis=1)[rows]
print(cols)
