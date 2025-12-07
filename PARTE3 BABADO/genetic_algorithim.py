import numpy as np
import matplotlib.pyplot as plt

class GeneticAlgorithm:
    def __init__(self,max_gen = 100 ,N=50,nbits = 15, p=2, nst = 3 , pc = .85, pm = .02, lower_bound = -1, upper_bound = 15):
        self.lb = lower_bound        
        self.ub = upper_bound
        self.nst = nst
        self.pc = pc
        self.pm = pm
        self.max_gen = max_gen
        self.nbits = nbits
        self.p = p
        self.N = N
        
     
        self.population = np.random.uniform(0,2,size=(N,nbits*p)).astype(int)
        self.population_real = [np.split(self.population[i,:],p) for i in range(N)]
        self.population_real = np.array([[self.phi(individual[0]),self.phi(individual[1])] for individual in self.population_real])
        self.fitness = np.array([self.f(*individual) for individual in self.population_real])
        
        
        
        fig = plt.figure(1)
        self.ax = fig.add_subplot(projection='3d')
        x = np.linspace(-1,15,1000)
        x,y = np.meshgrid(x,x)
        self.point = self.ax.scatter(self.population_real[:,0],
                   self.population_real[:,1],
                   self.fitness[:],c='r',s=45)
        self.ax.plot_surface(x,y,self.f(x,y),rstride=30,cstride=30,alpha=.1,edgecolors='k', lw=.1)
        
        
    
    def phi(self,x):
        s = 0.0
        n = len(x)
        for i in range(n):
            s += 2**i*x[n-1-i]
        return self.lb + (self.ub - self.lb)/(2**n-1)*s
    
    
    def f(self, x,y):
        return (x*y*(np.sin(y*np.pi)/4))
    
    
    def selection(self):
        self.selection_group = np.empty((0, self.nbits * self.p))
        for i in range(self.N):
            indexes = np.random.permutation(self.N)[:self.nst]
            winner_index = np.argmin(self.fitness[indexes])
            winner = self.population[indexes[winner_index]]
            self.selection_group = np.vstack((
                self.selection_group,
                winner.reshape(1,self.p * self.nbits)
            ))
    def crossover(self):
        for i in range(0,self.N,2):
            r = np.random.uniform(0,1)
            if r < self.pc:
                parent1 = np.array(np.split(self.selection_group[i,:],self.p))
                parent2 = np.array(np.split(self.selection_group[i+1,:],self.p))
                
                idx = np.random.randint(1,self.nbits-1)
                print(idx)
                
                child1 = np.copy(parent1)
                child2 = np.copy(parent2)
                
                for j in range(self.p):
                    child1[j,idx:] = parent2[j,idx:]
                    child2[j,idx:] = parent1[j,idx:]
                self.selection_group[i,:] = child1.reshape(1,self.p*self.nbits)
                self.selection_group[i+1,:] = child2.reshape(1,self.p*self.nbits)
        self.offspring = np.copy(self.selection_group)
              
            
    def mutation(self):
        for i in range(self.N):
            for j in range(self.nbits * self.p):
                r = np.random.uniform(0,1)
                if r < self.pm:
                    self.offspring[i,j] = (~(self.offspring[i,j].astype(bool))).astype(int)
                    bp=1
    
    def evolve(self):
        gen = 0
        hist_most_fitness = [np.min(self.fitness)]
        hist_least_fitness = [np.max(self.fitness)]
        hist_average_fitness = [np.mean(self.fitness)]
        
        while gen < self.max_gen:
            self.selection()
            self.crossover()
            self.mutation()
            plt.pause(.2)
            self.point.remove()
            self.population = np.copy(self.offspring)
            self.population_real = [np.split(self.population[i,:],self.p) for i in range(self.N)]
            self.population_real = np.array([[self.phi(individual[0]),self.phi(individual[1])] for individual in self.population_real])
            self.fitness = np.array([self.f(*individual) for individual in self.population_real])
            self.point = self.ax.scatter(self.population_real[:,0],
                   self.population_real[:,1],
                   self.fitness[:],c='r',s=45)
            hist_average_fitness.append(np.mean(self.fitness))
            hist_least_fitness.append(np.max(self.fitness))
            hist_most_fitness.append(np.min(self.fitness))
            gen +=1
        
        plt.figure(2)
        plt.plot(hist_most_fitness,label='Most Fitness')
        plt.plot(hist_least_fitness,label='Least Fitness')
        plt.plot(hist_average_fitness,label='Average Fitness')
        plt.legend()
        plt.grid()
        plt.show()

ga = GeneticAlgorithm()
ga.evolve()