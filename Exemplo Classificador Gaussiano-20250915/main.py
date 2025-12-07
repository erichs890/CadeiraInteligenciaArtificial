import numpy as np
import matplotlib.pyplot as plt
from gaussian_classifiers import GaussianClassifier

data = np.loadtxt("EMG3Classes.csv",delimiter=',')

X, y = data[:,:-1], data[:,-1:]

X_train = X[:int(.8*X.shape[0]),:]
y_train = y[:int(.8*X.shape[0]),:]

x_test = X[int(.8*X.shape[0]),:].reshape(1,2)
y_test = y[int(.8*X.shape[0]),:]

gc = GaussianClassifier(X_train.T,y_train.T)
gc.fit()

y_pred = gc.predict(x_test.T)
bp = 1