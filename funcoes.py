import numpy as np
import matplotlib.pyplot as plt

def logistica(u):
    return 1/(1+np.exp(-u))

def tanh(u):
    return (1-np.exp(-u))/(1+np.exp(-u))

def logistica_d(u):
    return np.exp(-u)/(1+np.exp(-u))**2

def tanh_d(u):
    return 2 * np.exp(-u) / (1 + np.exp(-u))**2

u = np.linspace(-5,5,1000)

fig = plt.figure(1)
ax = fig.add_subplot(2,2,1)
ax.set_title("Logistica")
y = logistica(u)
ax.plot(u,y)

ax = fig.add_subplot(2,2,2)
ax.set_title("Tangente Hiperbólica")
y = tanh(u)
ax.plot(u,y)

ax = fig.add_subplot(2,2,3)
ax.set_title("logistica derivada")
y = logistica_d(u)
ax.plot(u,y)

ax = fig.add_subplot(2,2,4)
ax.set_title("Tangente derivada")
y = tanh_d(u)
ax.plot(u,y)

plt.show()
