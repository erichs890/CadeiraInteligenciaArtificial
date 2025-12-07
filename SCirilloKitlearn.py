import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import inv, pinv, lstsq
class LinearRegression:
    def __init__(self, X_train, y_train, lbd = 0):
        self.X_train = X_train
        self.y_train = y_train
        self.lbd = lbd
        self.beta_hat = None
        self.MSE = None
        self.SSE = None
        self.N, self.p = X_train.shape
        self.X_train = np.hstack((
            np.ones((self.N,1)),X_train
        ))
    
    def fit(self):
        I = np.eye(self.p+1)

        self.beta_hat = inv(self.X_train.T@self.X_train + self.lbd*I)@self.X_train.T@self.y_train

    