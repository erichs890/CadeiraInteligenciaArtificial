import numpy as np
import matplotlib.pyplot as plt

def f(X,Y):
    return X**2*np.sin(4*np.pi*X) - Y*np.sin(4*np.pi*Y + np.pi) + 1

x = np.linspace(-1,2,1000)
X,Y = np.meshgrid(x,x)

fig = plt.figure(1)
ax = fig.add_subplot(projection='3d')
ax.plot_surface(X,Y,f(X,Y),rstride=50,cstride=50,alpha=.1,edgecolor='k')


#Têmpera simulada
it_max = 10000
T = 100
sigma = .2

#Inicialização
x_opt = np.random.uniform(-1,2,size=(2,))
f_opt = f(*x_opt) # Avalia o ponto inicial
hist = []
ax.scatter(*x_opt,f_opt,c='r')

i = 0
while i<it_max:
    #Geração de um novo candidato
    n = np.random.normal(loc=0,scale=sigma)
    
    x_cand = x_opt + n # Novo candidato
    x_cand = np.clip(x_cand,-1,2) # Mantém dentro dos limites
    f_cand = f(*x_cand) # Avalia o novo candidato
    P_ij = np.exp( -(f_cand - f_opt)/T )# Probabilidade de aceitação

    if f_cand < f_opt or P_ij >= np.random.uniform(0,1):# se o candidato é melhor ou é aceito, entao se torna o novo ponto
        f_opt = f_cand
        x_opt = x_cand
        plt.pause(.00000001)
        ax.scatter(*x_opt,f_opt,c='r')
    i+=1
    hist.append(f_opt)
    T = T*.99
    print(f"I = {i}, T = {T:.3f}, f_opt = {f_opt:.5f}")

ax.scatter(*x_opt,f_opt,c='g',s=300,marker='x',lw=3)
plt.figure(2)
plt.plot(hist)

plt.show()