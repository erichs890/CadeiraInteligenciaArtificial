import numpy as np
import matplotlib.pyplot as plt

x1 = np.array()
y = np.array()

X = np.hstack((
    np.ones((x1.shape[0],1)),x1
))

#TREINAMENTO DO MODELO
beta_hat = np.linalg.inv(X.T @ X) @ X.T @ y
print(beta_hat)

x_teste = np.linspace(-5, 2000)
y_hat= beta_hat[0,0] + beta_hat[1,0]*x_teste

plt.plot(x_teste, y_hat, c ='r')
plt.scatter(x1,y)
plt.xlim((100,1200))
plt.ylim((0,500))
plt.show()