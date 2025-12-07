import numpy as np
import matplotlib.pyplot as plt

# ou
data = np.loadtxt(r'.\av2\spiral_d.csv', delimiter=',')

X = data[:,:-1] 
rotulos = data[:,-1]

fig = plt.figure(1)
ax = fig.add_subplot(2,2,1)
ax.grid(True)
ax.set_title('Dados originais')
ax.scatter(X[rotulos[:]==1,0], X[rotulos[:]==1,1], color='blue', label='Classe 1',edgecolors='k')
ax.scatter(X[rotulos[:]==-1,0],X[rotulos[:]==-1,1], color='red', label='Classe -1', edgecolors='k')

#Nomralização (norma) local
n1 = np.linalg.norm(X[:,0])
n2 = np.linalg.norm(X[:,1])
X_normalizado = np.copy(X)
X_normalizado[:,0] = X[:,0]/n1
X_normalizado[:,1] = X[:,1]/n2

fig = plt.figure(1)
ax = fig.add_subplot(2,2,2)
ax.grid(True)
ax.set_title('Dados normalizados')
ax.scatter(X_normalizado[rotulos[:]==1,0], X[rotulos[:]==1,1], color='blue', label='Classe 1',edgecolors='k')
ax.scatter(X_normalizado[rotulos[:]==-1,0],X[rotulos[:]==-1,1], color='red', label='Classe -1', edgecolors='k')


#Padronização
mu = np.mean(X, axis=0)
sigma = np.std(X, axis=0)
X_normalizado = (X-mu)/sigma
fig = plt.figure(1)
ax = fig.add_subplot(2,2,3)
ax.grid(True)
ax.set_title('Dados padronizados')
ax.scatter(X_normalizado[rotulos[:]==1,0], X[rotulos[:]==1,1], color='blue', label='Classe 1',edgecolors='k')
ax.scatter(X_normalizado[rotulos[:]==-1,0],X[rotulos[:]==-1,1], color='red', label='Classe -1', edgecolors='k')

#Nomrmalização (min-max)
X_normalizado = 2*((X-np.min(X))/(np.max(X)-np.min(X)))-1
fig = plt.figure(1)
ax = fig.add_subplot(2,2,4)
ax.grid(True)
ax.set_title('Dados noramalização min-max')
ax.scatter(X_normalizado[rotulos[:]==1,0], X[rotulos[:]==1,1], color='blue', label='Classe 1',edgecolors='k')
ax.scatter(X_normalizado[rotulos[:]==-1,0],X[rotulos[:]==-1,1], color='red', label='Classe -1', edgecolors='k')



plt.show()