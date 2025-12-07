import numpy as np
from numpy.linalg import pinv
import matplotlib.pyplot as plt

#Carregar os dados
data = np.loadtxt("Solubilidade.csv", delimiter=",")
X = data[:,:-1]
N,p = X.shape
y = data[:,-1:]

fig = plt.figure(1)
ax = fig.add_subplot(projection='3d')
ax.scatter(X[:,0],X[:,1],y[:,0],color = "pink", edgecolor = "k")
ax.set_xlabel("Quantidade de carbolo")
ax.set_ylabel("Peso Molecular")
ax.set_zlabel("Nivel Solubilidade")

X = np.hstack((
    np.ones((N,1)),X
))
#Desempenho 
mediaSSE = []
mediaMSE = []
MQOSSE = []
MQOMSE = []
#Definição da quantidade de rodadas
rodadas = 500
controle_plot = True

for r in range(rodadas):
    #Embaralhar o conjudo de dados
    idx = np.random.permutation(N)
    Xr = X[idx,:]
    yr = y[idx,:]
    #Particionamento do conjunto de dados (80% treino, 20% teste)
    X_treino = Xr[:int(0.8*N),:]
    y_treino = yr[:int(0.8*N),:]
    
    X_teste = Xr[int(0.8*N):,:]
    y_teste = yr[int(0.8*N):,:]
    #Treino (Modelo baseado na media, Modelo baseado no MQO/OLS)
    #Média:
    beta_media = np.array([
        [np.mean(y_treino)],
        [0],
        [0]
    ])
    #MQO/OLS
    beta_hat = pinv(X_treino.T@X_treino)@X_treino.T@y_treino

    if controle_plot:
        fig = plt.figure(2)
        ax = fig.add_subplot(projection='3d')
        ax.scatter(X_treino[:,1],X_treino[:,2],y_treino[:,0],color = "pink", edgecolor = "k")
        ax.set_xlabel("Quantidade de carbolo")
        ax.set_ylabel("Peso Molecular")
        ax.set_zlabel("Nivel Solubilidade")

        x1 = np.linspace(0,35)
        x2 = np.linspace(13,700)
        X3d,Y3d = np.meshgrid(x1,x2)
        Z3d = beta_media[0,0] + beta_media[1,0]*X3d + beta_media[2,0]*Y3d
        ax.plot_surface(X3d,Y3d,Z3d)

        Z3d = beta_hat[0,0] + beta_hat[1,0]*X3d + beta_hat[2,0]*Y3d
        ax.plot_surface(X3d,Y3d,Z3d)


    controle_plot = False


    #Teste (SSE,MSE,R^2)
    #Modelo Média:
    y_hat_teste = X_teste @ beta_media
    y_hat_treino = X_treino @ beta_media
    print("-----------Média-----------")
    SSE = np.sum((y_teste - y_hat_teste)**2)    
    print(F"{SSE:.5f}")
    SSE = np.sum((y_treino - y_hat_treino)**2)    
    MSE = np.mean((y_teste - y_hat_teste)**2)
    print(F"{MSE:.5f}")
    SST = np.sum((y_teste - np.mean(y_teste))**2)
    R2 = 1 - SSE/SST
    print(F"{R2:.5f}")
    mediaSSE.append(SSE)
    mediaMSE.append(MSE)

    #Modelo MQO/OLS:
    print("-----------MQO/OLS-----------")
    y_hat_teste = X_teste@ beta_hat
    y_hat_treino = X_treino @ beta_hat
    SSE = np.sum((y_teste - y_hat_teste)**2)
    print(F"{SSE:.5f}")
    MSE = np.mean((y_teste - y_hat_teste)**2)
    print(F"{MSE:.5f}")
    SST = np.sum((y_teste - np.mean(y_teste))**2)
    R2 = 1 - SSE/SST
    print(F"{R2:.5f}")
    MQOSSE.append(SSE)
    MQOMSE.append(MSE)

mediaSSE = np.mean(mediaSSE)
mediaMSE = np.mean(mediaMSE)
MQOSSE = np.mean(MQOSSE)
MQOMSE = np.mean(MQOMSE)
print("Média dos SSE: ",mediaSSE)
print("Média dos MSE: ",mediaMSE)
print("Média dos SSE MQO: ",MQOSSE)
print("Média dos MSE MQO: ",MQOMSE)

plt.show()