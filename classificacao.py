import numpy as np
import matplotlib.pyplot as plt

data = np.loadtxt("EMGsDataset.csv", delimiter=",")

data = data.T

classes = np.unique(data[:,-1])
C = len(classes)
N,p = data.shape
p = 2
C = 3
X = np.empty((0,p))
nome_classes = ['Neutro', 'Sorriso', 'Sobrancelhas Levantadas', 'Surpreso', 'Rabugento']
cores = ['gray', 'red', 'g', 'blue', 'pink']

for i, classe in enumerate(classes):
    X_classe = data[data[:-1]==classe,:-1]
    x = np.vstack((
        X,X_classe
    ))
    rotulo = -np.ones((1,C))
    rotulo[0,i] = 1
    Y_classe = np.tile(rotulo,(X_classe.shape[0],1))
    plt.scatter(X_classe[:,0],X_classe[:,1],
                c = cores[i],
                label = nome_classes[i],
                edgecolor = 'k',)

N = X.shape[0]
x = np.hstack((np.ones((N,1)),X))
#Treinamento
W_hat = np.linalg.inv(x.T @ x) @ x.T @ Y_classe
x1 = np.linspace(-500,4500)
x2 = -W_hat[0,0]/W_hat[2,0] - W_hat[1,0]/W_hat[2,0]*x1
plt.plot(x1,x2,c="k")
x2 = -W_hat[0,1]/W_hat[2,1] - W_hat[1,1]/W_hat[2,1]*x1
plt.plot(x1,x2,c="k")
x2 = -W_hat[0,2]/W_hat[2,2] - W_hat[1,2]/W_hat[2,2]*x1
plt.plot(x1,x2,c="k")

#predicao
x_novo = np.array([[1,2,2005]])
plt.scatter(x_novo[0,1],x_novo[0,2],c="yellow",s=200,edgecolor='k',marker="*")
plt.legend()
plt.xlabel("Sensor 1 (Corrugador do supercilio)")
plt.ylabel("Sensor 2 (Zigomatico maior)")
plt.title("Classificação de expressões faciais")
plt.show()  