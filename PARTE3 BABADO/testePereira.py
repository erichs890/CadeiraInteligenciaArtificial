import numpy as np
import matplotlib.pyplot as plt


def f(X,Y):
    return np.exp(-(X**2+Y**2)) + 2*np.exp(-((X-2)**2+(Y-2)**2))

lim_inf = -2
lim_sup = 4

x_axis = np.linspace(lim_inf,lim_sup,500)

X,Y = np.meshgrid(x_axis,x_axis)

fig = plt.figure(1)
ax = fig.add_subplot(projection='3d')
ax.plot_surface(X,Y, f(X,Y),rstride=20,cstride=20,alpha=.3,edgecolor='k')
#Têmpera simulada
it_max = 10000
T = 100
sigma = .2

#Inicialização
x_opt = np.random.uniform(-1,2,size=(2,))
f_opt = f(*x_opt)
hist = []
ax.scatter(*x_opt,f_opt,c='r')

i = 0
while i<it_max:
    n = np.random.normal(loc=0,scale=sigma,size=(2,))
    x_cand = x_opt + n
    x_cand = np.clip(x_cand,-1,2)
    f_cand = f(*x_cand)
    P_ij = np.exp( -(f_cand - f_opt)/T )

    if f_cand < f_opt or P_ij >= np.random.uniform(0,1):
        f_opt = f_cand
        x_opt = x_cand
        plt.pause(.00000001)
        ax.scatter(*x_opt,f_opt,c='r')
    i+=1
    hist.append(f_opt)
    T = T*.98
    print(f"I = {i}, T = {T:.3f}, f_opt = {f_opt:.5f}")

ax.scatter(*x_opt,f_opt,c='g',s=300,marker='x',lw=3)
plt.figure(2)
plt.plot(hist)

plt.show()