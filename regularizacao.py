import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import matrix_rank, inv

XTX = np.array([
    [1,2,15],
    [2,4,5],
    [3,6,17],
])

print(matrix_rank(XTX))
I = np.eye(XTX.shape[0])
lbd = [0.1,0.2,.7,.9,1]
print(inv(XTX+lbd*I))