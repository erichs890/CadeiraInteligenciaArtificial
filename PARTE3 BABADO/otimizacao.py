import numpy as np
import matplotlib.pyplot as plt

def f(X,Y):
    return np.exp(-(X**2+Y**2)) + 2*np.exp(-((X-2)**2+(Y-2)**2))

lim_inf = -2
lim_sup = 4

x_axis = np.linspace(lim_inf, lim_sup, 500)

X, Y = np.meshgrid(x_axis, x_axis)

fig = plt.figure(1)
ax = fig.add_subplot(projection='3d')
ax.plot_surface(X, Y, f(X,Y),rstride=5, cstride=5, cmap='viridis', edgecolor='none', alpha=0.2)


#Busca pela subida de encosta (hill climbing)
epsilon = .1 #Tamanho da vizinhança(epsilon)
max_it = 1000
max_vizinhos = 20

x_opt = np.random.uniform(low=lim_inf, high=lim_sup, size=2)
f_opt = f(*x_opt)
historico = [f_opt]
ax.scatter(*x_opt, f_opt, color='r', s=50)

it = 0
melhoria = True
while it < max_it and melhoria:
    melhoria = False
    for j in range(max_vizinhos):
        #Perturbação do ótimo
        x_cand = np.random.uniform(low=x_opt-epsilon, high=x_opt+epsilon)
        for i,x in enumerate(x_cand):
            if x<lim_inf:
                x_cand[i] = lim_inf
            if x > lim_sup:
                x_cand[i] = lim_sup 
        f_cand = f(*x_cand)
        historico.append(f_cand)
        if f_cand > f_opt:
            x_opt = x_cand
            f_opt = f_cand
        
            melhoria = True
            plt.pause(.1)
            ax.scatter(*x_opt, f_opt, color='r', s=50)
            break
    it+=1
ax.scatter(*x_opt, f_opt, color='g',marker = "*", s=250)
plt.figure(2)
plt.plot(historico)
plt.grid()



plt.show()