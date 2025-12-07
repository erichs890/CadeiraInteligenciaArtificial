import numpy as np


# Declaração.
x = np.array([1,2,3,4]).reshape(4,1)
print(x.shape)
print(x)


#Geradores aleatórios.
X = np.random.uniform(low=-3, high=10, size=(10,3))
X = np.random.randint(low=10,high=100,size=(5,7))
X = np.random.normal(loc=0,scale=5,size=(10,2)).astype(int)
idx = np.random.permutation(X.shape[0])

print(idx)

#geradores determinísticos
x = np.linspace(-3,10)
x = np.arange(10,0,-1)
print(x)

o = np.ones((3,7))
print(o)
Z = np.zeros((5,6))
print(Z)

I = np.eye(7)
print(I)

X1 = np.copy(X)

#operadores ariméticos e da álgebra linear
A = np.random.randint(size =(10,1), low=1, high=10)
B = np.random.randint(size =(10,1), low=0, high=2)
print(A)
print()
print(B)
print()
P = A@B.T
print(P)

A = np.random.randint(size =(5,5), low=1, high=10)
print(np.linalg.det(A))
print(np.linalg.inv(A))
print(np.linalg.matrix_rank(A))


# auxiliares:
A = np.random.randint(size =(10,4), low=1, high=10)

Z = np.hstack((
    np.ones((A.shape[0],1)),
    A
))
print(Z)
Z = np.vstack((
    A,
    np.ones((1,A.shape[1]))
))
print(Z)

A = np.random.randint(size =(10,4), low=1, high=10)
Z = np.tile(A,(2,5))
print(Z)

x = np.linspace(0,5,100)
y = np.linspace(0,50,100)
X,Y = np.meshgrid(x,y)


#indexadores

A = np.random.randint(size =(10,4), low=10, high=100)
print(A)
print(A[1:2,-3:])
A = np.hstack((
    A,
    np.random.randint(0,2,(A.shape[0],1))
))


print(A)
print(A[A[:,-1]==0,:-1])

print("------")
A = np.random.randint(size =(10,4), low=10, high=100)
idx = np.random.permutation(A.shape[0])
idx = np.random.randint(low=0,high=4,size=(20,))
print(A)
print(idx)
embaralhado = A[idx,:]
print(embaralhado)