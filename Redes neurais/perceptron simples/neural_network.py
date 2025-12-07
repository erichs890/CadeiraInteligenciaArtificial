import numpy as np
import matplotlib.pyplot as plt


class Perceptron:
    def __init__(self,X_train, y_train,learning_rate=1e-3,plot=True):
        
        self.X_train = X_train
        self.p, self.N = X_train.shape
        self.X_train = np.vstack((
            -np.ones((1,self.N)), self.X_train
        ))
        
        self.w = np.zeros((self.p+1,1))
        self.w = np.random.random_sample((self.p+1,1))-.5
        self.y_train = y_train
        self.lr = learning_rate        
        self.plot = plot

        if plot:
            self.fig = plt.figure(1)
            self.ax = self.fig.add_subplot()
            self.ax.scatter(self.X_train[1,self.y_train[:]==1],
                            self.X_train[2,self.y_train[:]==1], marker='x', s=130)
            self.ax.scatter(self.X_train[1,self.y_train[:]==-1],
                            self.X_train[2,self.y_train[:]==-1], marker='s', s=130)
            self.ax.set_xlim(-1,7)
            self.ax.set_ylim(-1,7)
            self.x1 = np.linspace(-5,10)
            self.draw_line()
           
    def draw_line(self,c='k',alpha=1,lw=2):
        x2 = -self.w[1,0]/self.w[2,0]*self.x1 + self.w[0,0]/self.w[2,0]
        x2 = np.nan_to_num(x2)
        self.ax.plot(self.x1,x2,c=c,alpha=alpha,lw=lw)
    
    def activation_function(self, u):
        return 1 if u>=0 else -1
    
    def fit(self):
        error = True
        epochs = 0
        while error:
            error = False
            for k in range(self.N):
                x_k = self.X_train[:,k].reshape(self.p+1,1)
                u_k = (self.w.T@x_k)[0,0]
                y_k = self.activation_function(u_k)
                d_k = self.y_train[k]
                e_k = d_k - y_k
                if e_k != 0:
                    error = True
                self.w = self.w + self.lr*e_k*x_k
            plt.pause(.2)
            self.draw_line(c='r',alpha=.5)
            epochs+=1
            print(f'Epochs: {epochs}', end='\r')
            
        plt.pause(.4)
        self.draw_line(c='g',alpha=1,lw=4)
        plt.show()


bp=1