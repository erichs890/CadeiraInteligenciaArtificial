import numpy as np
import matplotlib.pyplot as plt
import os

# ========================================
# Leitura do arquivo CSV (apenas numpy)
# ========================================
def load_cities(filename):
    """
    Carrega coordenadas de um arquivo CSV com 2 ou 3 colunas.
    Assume que o arquivo NÃO tem cabeçalho.
    """
    if not os.path.exists(filename):
        # Caso o arquivo não exista, gera um exemplo aleatório com 40 cidades
        print(f"Arquivo '{filename}' não encontrado. Gerando 40 cidades aleatórias (2D)...")
        np.random.seed(42)
        return np.random.uniform(0, 100, size=(40, 2))
    
    data = np.loadtxt(filename, delimiter=',')
    if data.ndim == 1:
        data = data.reshape(1, -1)
    print(f"✅ Arquivo carregado: {data.shape[0]} cidades com {data.shape[1]} dimensões.")
    return data

# ========================================
# Função de distância total (função objetivo)
# ========================================
def total_distance(route, cities):
    """Calcula a distância total de uma rota (permutação de índices)."""
    coords = cities[route]
    # Distância entre pontos consecutivos + retorno ao início
    diffs = np.diff(np.vstack([coords, coords[0]]), axis=0)
    return np.sum(np.sqrt(np.sum(diffs**2, axis=1)))

# ========================================
# Algoritmo Genético para TSP
# ========================================
class TSP_GeneticAlgorithm:
    def __init__(self, cities, pop_size=100, max_gen=500, tournament_size=5,
                 pc=0.9, pm=0.01, elite_size=5, plot_interval=10):
        self.cities = np.array(cities)
        self.num_cities = len(cities)
        self.pop_size = pop_size
        self.max_gen = max_gen
        self.tournament_size = tournament_size
        self.pc = pc
        self.pm = pm
        self.elite_size = elite_size
        self.plot_interval = plot_interval
        self.dim = cities.shape[1]  # 2D ou 3D

        # Inicializar população: cada indivíduo é uma permutação de [0, 1, ..., n-1]
        self.population = np.array([np.random.permutation(self.num_cities) for _ in range(pop_size)])
        self.fitness = np.array([total_distance(ind, self.cities) for ind in self.population])

        # Melhor solução inicial
        best_idx = np.argmin(self.fitness)
        self.best_route = self.population[best_idx].copy()
        self.best_distance = self.fitness[best_idx]

        # Histórico
        self.history = {'best': [self.best_distance], 'avg': [np.mean(self.fitness)]}

        # Configurar plot
        self.setup_plot()

    def setup_plot(self):
        plt.ion()
        if self.dim == 2:
            self.fig, self.ax = plt.subplots(figsize=(8, 8))
        else:
            self.fig = plt.figure(figsize=(8, 8))
            self.ax = self.fig.add_subplot(111, projection='3d')
        self.update_plot()

    def update_plot(self):
        self.ax.clear()
        if self.dim == 2:
            self.ax.scatter(self.cities[:, 0], self.cities[:, 1], c='blue', s=50, zorder=5)
            route_coords = self.cities[self.best_route]
            self.ax.plot(route_coords[:, 0], route_coords[:, 1], 'r-', linewidth=1.5, zorder=4)
            self.ax.plot([route_coords[-1, 0], route_coords[0, 0]],
                         [route_coords[-1, 1], route_coords[0, 1]], 'r-', linewidth=1.5, zorder=4)
            self.ax.set_title(f"Melhor distância: {self.best_distance:.2f}")
        else:
            self.ax.scatter(self.cities[:, 0], self.cities[:, 1], self.cities[:, 2], c='blue', s=50)
            route_coords = self.cities[self.best_route]
            self.ax.plot(route_coords[:, 0], route_coords[:, 1], route_coords[:, 2], 'r-', linewidth=1.5)
            self.ax.plot([route_coords[-1, 0], route_coords[0, 0]],
                         [route_coords[-1, 1], route_coords[0, 1]],
                         [route_coords[-1, 2], route_coords[0, 2]], 'r-', linewidth=1.5)
            self.ax.set_title(f"Melhor distância: {self.best_distance:.2f}")
        plt.draw()
        plt.pause(0.01)

    def tournament_selection(self):
        """Seleção por torneio."""
        selected = []
        for _ in range(self.pop_size):
            candidates = np.random.choice(self.pop_size, self.tournament_size, replace=False)
            winner = candidates[np.argmin(self.fitness[candidates])]
            selected.append(self.population[winner].copy())
        return np.array(selected)

    def order_crossover(self, parent1, parent2):
        """Order Crossover (OX) - mantém a ordem relativa."""
        size = len(parent1)
        start, end = sorted(np.random.choice(size, 2, replace=False))
        child = np.full(size, -1)
        child[start:end] = parent1[start:end]
        fill_pos = end
        for city in np.concatenate((parent2[end:], parent2[:end])):
            if city not in child:
                child[fill_pos % size] = city
                fill_pos += 1
        return child

    def crossover(self, selected):
        """Aplica crossover com probabilidade pc."""
        offspring = []
        for i in range(0, self.pop_size, 2):
            p1, p2 = selected[i], selected[i+1]
            if np.random.rand() < self.pc and i+1 < self.pop_size:
                c1 = self.order_crossover(p1, p2)
                c2 = self.order_crossover(p2, p1)
                offspring.extend([c1, c2])
            else:
                offspring.extend([p1, p2])
        return np.array(offspring[:self.pop_size])

    def mutate(self, individual):
        """Mutação por troca (swap) com probabilidade pm."""
        for i in range(len(individual)):
            if np.random.rand() < self.pm:
                j = np.random.randint(len(individual))
                individual[i], individual[j] = individual[j], individual[i]
        return individual

    def evolve(self):
        for gen in range(self.max_gen):
            # Elitismo: preserva os melhores
            elite_indices = np.argsort(self.fitness)[:self.elite_size]
            elite = self.population[elite_indices]

            # Seleção, crossover, mutação
            selected = self.tournament_selection()
            offspring = self.crossover(selected)
            offspring = np.array([self.mutate(ind) for ind in offspring])

            # Substituir população (mantém elite)
            offspring[:self.elite_size] = elite
            self.population = offspring
            self.fitness = np.array([total_distance(ind, self.cities) for ind in self.population])

            # Atualizar melhor global
            current_best_idx = np.argmin(self.fitness)
            current_best_dist = self.fitness[current_best_idx]
            if current_best_dist < self.best_distance:
                self.best_distance = current_best_dist
                self.best_route = self.population[current_best_idx].copy()

            # Histórico
            self.history['best'].append(self.best_distance)
            self.history['avg'].append(np.mean(self.fitness))

            # Atualizar gráfico a cada N gerações
            if gen % self.plot_interval == 0:
                self.update_plot()
                print(f"Geração {gen+1}/{self.max_gen} | Melhor: {self.best_distance:.2f}")

            # Critério de parada opcional: convergência
            if gen > 50 and abs(self.history['best'][-1] - self.history['best'][-50]) < 1e-3:
                print("Convergência detectada. Parando.")
                break

        plt.ioff()
        self.update_plot()
        self.plot_history()
        plt.show()

    def plot_history(self):
        plt.figure(figsize=(10, 5))
        plt.plot(self.history['best'], label='Melhor distância', color='red')
        plt.plot(self.history['avg'], label='Média da população', color='blue')
        plt.xlabel('Geração')
        plt.ylabel('Distância total')
        plt.title('Evolução do Algoritmo Genético para TSP')
        plt.legend()
        plt.grid(True)

# ========================================
# Execução principal
# ========================================

# Carregar cidades do CSV (ou gerar exemplo)
cities = load_cities("CaixeiroGruposGA.csv")

# Garantir que o número de cidades esteja entre 30 e 60
if cities.shape[0] < 30:
    print(f"Aviso: número de cidades ({cities.shape[0]}) < 30. Considerar gerar mais.")
elif cities.shape[0] > 60:
    print(f"Aviso: número de cidades ({cities.shape[0]}) > 60. Reduzindo para 60.")
    cities = cities[:60]

# Configurar e executar GA
ga = TSP_GeneticAlgorithm(
    cities=cities,
    pop_size=100,
    max_gen=500,
    tournament_size=5,
    pc=0.9,
    pm=0.01,  # 1% de mutação
    elite_size=5,
    plot_interval=20  # atualiza a cada 20 gerações
)
ga.evolve()

print(f"\n✅ Melhor distância encontrada: {ga.best_distance:.2f}")
print(f"Rota (índices das cidades): {ga.best_route}")