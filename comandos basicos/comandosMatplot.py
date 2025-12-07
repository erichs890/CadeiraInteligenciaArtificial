import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gs
def tridimensional(X,Y):
    return np.exp(-(X**2+Y**2)) + 2*np.exp(-((X-1.7)**2 + (Y-1.7)**2))
def seno(x):
    return np.sin(2*np.pi*x)
def cosseno(x):
    return np.cos(2*np.pi*x)

def normal(x,mu=0,sigma_2=1):
    return 1/(np.sqrt(2*np.pi*sigma_2))*np.exp(-((x-mu)**2/(2*sigma_2)))

x_axis = np.linspace(-5,5,1000)
plt.figure(1,facecolor='k')
plt.plot(x_axis, seno(x_axis),label='Seno x')
plt.plot(x_axis, cosseno(x_axis),label="Cosseno x")
plt.title("Funções matemáticas")
plt.legend(loc='upper right')
plt.grid()
plt.gca().set_facecolor("k")
plt.tight_layout()


x_axis = np.linspace(-5,5,50)
fig = plt.figure(2)

ax = fig.add_subplot(2,2,1)
n = normal(x_axis,mu=1.5)
ax.plot(x_axis,n,c='pink',lw=5,ls='dotted')
ax.set_title("Primeiro")

ax = fig.add_subplot(2,2,2)
ax.stem(x_axis, normal(x_axis))
ax.set_title("Segundo")

ax = fig.add_subplot(2,2,3)
ax.scatter(x_axis,normal(x_axis),c='r',marker=r'$😊$',s=500)
ax.set_title("Terceiro")

normais1 = np.random.normal(0,1,(500,))
normais2 = np.random.normal(3,0.1,(500,))
ax = fig.add_subplot(2,2,4)
ax.set_title("Quarto")
ax.boxplot([normais1,normais2],positions=[1,2])

g = gs.GridSpec(2,2)
fig = plt.figure(3)
inteiros = np.random.normal(0,2,(500,)).astype(int)


ax = fig.add_subplot(g[0,0])
ax.hist(inteiros,color='teal',edgecolor='k')
ax = fig.add_subplot(g[0,1])
contadores = [np.sum(inteiros == i) for i in np.unique(inteiros)]
ax.pie(contadores,labels=np.unique(inteiros))
ax = fig.add_subplot(g[1,:])
ax.violinplot(inteiros,vert=False)



fig = plt.figure(4)
x_axis = np.linspace(-10,10,1000)
ax = fig.add_subplot(projection='3d')

ax.plot(seno(x_axis)*np.exp(x_axis*.1),cosseno(x_axis)*np.exp(x_axis*.1),x_axis)


fig = plt.figure(5)
x_axis = np.linspace(-2,4,1000)
ax = fig.add_subplot(projection='3d')
X,Y = np.meshgrid(x_axis,x_axis)
Z = tridimensional(X,Y)
ax.plot_surface(X,Y,-Z,cmap='gray',alpha=.1,edgecolor='k',
                rstride=60, cstride=60)
ax.scatter(0,0,-tridimensional(0,0),s=120)
# ax.contour(X,Y,Z,offset=0)

fig = plt.figure(6)
c= plt.contour(X,Y,Z)
plt.colorbar(c)


dados = np.loadtxt("spiral.csv",delimiter=',')

plt.figure(7)
plt.scatter(dados[:,0],dados[:,1])
plt.figure(8)
plt.scatter(dados[dados[:,-1]==1,0],dados[dados[:,-1]==1,1],c='cyan',edgecolors='k')
plt.scatter(dados[dados[:,-1]==-1,0],dados[dados[:,-1]==-1,1],c='magenta',edgecolors='k')



dados = np.loadtxt("EMGsDataset.csv",delimiter=',')
dados = dados.T



plt.figure(9)
cores = ['pink','purple','blue','orange','brown']
[plt.scatter(dados[dados[:,-1]==rotulos,0],
             dados[dados[:,-1]==rotulos,1],
             color = cores[i],
             edgecolors='k') for i,rotulos in enumerate(np.unique(dados[:,-1]))]

plt.show()


bp=1