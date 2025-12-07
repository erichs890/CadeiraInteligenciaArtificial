import numpy as np
import matplotlib.pyplot as plt
from MultiLayerPerceptron import MultilayerPerceptron

data = np.loadtxt("spiral_d.csv",delimiter=',')

X_treino = data[:,:-1].T
X_treino = 2*(X_treino-np.min(X_treino))/(np.max(X_treino)-np.min(X_treino))-1

y_treino = data[:,-1]
# plt.scatter(X_treino[0,y_treino[:]==1],X_treino[1,y_treino[:]==1],edgecolors='k')
# plt.scatter(X_treino[0,y_treino[:]==-1],X_treino[1,y_treino[:]==-1],edgecolors='k')
# plt.show()
n1 = np.sum(y_treino[:]==1)
n2 = np.sum(y_treino[:]==-1)

Y_treino = np.tile(
    np.array([1,-1]).reshape(2,1),(1,n1)
)
Y_treino = np.hstack((
    Y_treino, 
    np.tile(
    np.array([-1,1]).reshape(2,1),(1,n2)
)
))

mlp = MultilayerPerceptron(X_treino,Y_treino,[1000,1000,1000,1000,500,250,50])
mlp.fit()
bp = 1