import numpy as np
import matplotlib.pyplot as plt



x_0 = 1
x_1 = np.linspace(-5,5,100)
plt.figure(1)

for i in range(50):
    b_0 = np.random.uniform(-5,7)
    b_1 = np.random.uniform(-3,8)
    y = b_0*x_0 + b_1*x_1
    plt.plot(x_1,y)


plt.figure(2)

for i in range(50):
    b_0 = np.random.uniform(-5,7)
    b_1 = np.random.uniform(-5,7)
    y = b_0*x_0 + b_1*x_1
    plt.plot(x_1,y)

plt.show()
bp=1

"""
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope Process
.\ambinte_ia\Scripts\activate

Intercepto eh quando o 

"""