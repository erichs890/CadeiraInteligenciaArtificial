import numpy as np
import matplotlib.pyplot as plt

# =============================
# 1. CARREGAR OS DADOS (EMGDataset.csv)
# =============================
data = np.genfromtxt('EMGsDataset.csv', delimiter=',')
X_raw = data[:2, :]        # (2, 50000) — features: Corrugador e Zigomático
Y_raw = data[2, :].astype(int)  # (50000,) — rótulos: 1 a 5

# Para MQO: X ∈ R^(N×p), Y ∈ R^(N×C) → precisamos de one-hot
N, p = 50000, 2
C = 5
X_mqo = X_raw.T  # Shape: (50000, 2)
Y_onehot = np.zeros((N, C))
for i in range(N):
    if 1 <= Y_raw[i] <= 5:
        Y_onehot[i, Y_raw[i]-1] = 1  # one-hot: classe 1 → [1,0,0,0,0]

# Para modelos gaussianos: X ∈ R^(p×N), Y ∈ R^(C×N)
X_bayes = X_raw          # (2, 50000)
Y_bayes = np.zeros((C, N))  # (5, 50000)
for i in range(N):
    if 1 <= Y_raw[i] <= 5:
        Y_bayes[Y_raw[i]-1, i] = 1  # Y_bayes: cada linha é uma classe

print("Dados carregados:")
print(f"X_mqo shape: {X_mqo.shape}")
print(f"Y_onehot shape: {Y_onehot.shape}")
print(f"X_bayes shape: {X_bayes.shape}")
print(f"Y_bayes shape: {Y_bayes.shape}")

# =============================
# 2. VISUALIZAÇÃO INICIAL (Dispersão por classe)
# =============================
classes_names = ['Neutro', 'Sorriso', 'Sobrancelhas', 'Surpreso', 'Rabugento']
colors = ['blue', 'green', 'red', 'purple', 'orange']

plt.figure(figsize=(10, 8))
for c in range(1, 6):
    idx = Y_raw == c
    plt.scatter(X_raw[0, idx], X_raw[1, idx], c=colors[c-1], label=classes_names[c-1], alpha=0.6, s=10)

plt.xlabel('Corrugador (Sensor 1)')
plt.ylabel('Zigomático (Sensor 2)')
plt.title('Dispersão dos Sinais EMG por Expressão Facial')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.7)
plt.show()

# =============================
# 3. MODELOS IMPLEMENTADOS (APENAS NUMPY E MATPLOTLIB)
# =============================

class GaussianClassifier:
    def __init__(self, X_train, y_train):
        self.classes = np.unique(y_train[0, :])  # y_train: (C, N)
        self.C = len(self.classes)
        self.p, self.N = X_train.shape
        
        # Agrupar X por classe: lista de arrays (p, n_c)
        self.X = [X_train[:, y_train[i, :] == 1] for i in range(self.C)]
        self.n = [Xi.shape[1] for Xi in self.X]
        
        self.Sigma = [None]*self.C
        self.Sigma_det = [None]*self.C
        self.Sigma_inv = [None]*self.C
        self.mu = [None]*self.C
        self.P = [None]*self.C
        self.g = [None]*self.C
    
    def fit(self):
        for i in range(self.C):
            self.mu[i] = np.mean(self.X[i], axis=1).reshape(self.p, 1)
            self.Sigma[i] = np.cov(self.X[i])
            self.Sigma_det[i] = np.linalg.det(self.Sigma[i])
            # Evitar singularidade com pseudo-inversa
            self.Sigma_inv[i] = np.linalg.pinv(self.Sigma[i])
            self.P[i] = self.n[i] / self.N
    
    def predict(self, x_test):
        # x_test: (p, 1)
        for i in range(self.C):
            diff = x_test - self.mu[i]
            d_mahalanobis = (diff.T @ self.Sigma_inv[i] @ diff)[0, 0]
            self.g[i] = np.log(self.P[i]) - 0.5 * np.log(self.Sigma_det[i]) - 0.5 * d_mahalanobis
        return np.argmax(self.g) + 1  # retorna 1 a 5


class GaussianEqualCov:
    def __init__(self, X_train, y_train):
        self.classes = np.unique(y_train[0, :])
        self.C = len(self.classes)
        self.p, self.N = X_train.shape
        
        self.X = [X_train[:, y_train[i, :] == 1] for i in range(self.C)]
        self.n = [Xi.shape[1] for Xi in self.X]
        
        self.mu = [None]*self.C
        self.P = [None]*self.C
        self.Sigma = None
        self.Sigma_det = None
        self.Sigma_inv = None
    
    def fit(self):
        # Calcular médias e prioris
        for i in range(self.C):
            self.mu[i] = np.mean(self.X[i], axis=1).reshape(self.p, 1)
            self.P[i] = self.n[i] / self.N
        
        # Calcular covariância comum (ponderada)
        total_cov = np.zeros((self.p, self.p))
        for i in range(self.C):
            for j in range(self.X[i].shape[1]):
                diff = (self.X[i][:, j] - self.mu[i].flatten()).reshape(-1, 1)
                total_cov += diff @ diff.T
        self.Sigma = total_cov / (self.N - self.C)  # graus de liberdade
        self.Sigma_det = np.linalg.det(self.Sigma)
        self.Sigma_inv = np.linalg.pinv(self.Sigma)
    
    def predict(self, x_test):
        g = []
        for i in range(self.C):
            diff = (x_test - self.mu[i]).flatten()
            mahalanobis = diff.T @ self.Sigma_inv @ diff
            posterior = -0.5 * mahalanobis + np.log(self.P[i])
            g.append(posterior)
        return np.argmax(g) + 1


class GaussianAggregated:
    def __init__(self, X_train, y_train):
        self.classes = np.unique(y_train[0, :])
        self.C = len(self.classes)
        self.p, self.N = X_train.shape
        
        self.X = [X_train[:, y_train[i, :] == 1] for i in range(self.C)]
        self.n = [Xi.shape[1] for Xi in self.X]
        
        self.mu_global = None
        self.Sigma_global = None
        self.Sigma_det = None
        self.Sigma_inv = None
        self.P = [None]*self.C
    
    def fit(self):
        # Covariância global de TODAS as amostras
        self.mu_global = np.mean(X_bayes, axis=1).reshape(-1, 1)
        self.Sigma_global = np.cov(X_bayes)
        self.Sigma_det = np.linalg.det(self.Sigma_global)
        self.Sigma_inv = np.linalg.pinv(self.Sigma_global)
        
        # Prioris por classe
        for i in range(self.C):
            self.P[i] = self.n[i] / self.N
    
    def predict(self, x_test):
        g = []
        for i in range(self.C):
            diff = (x_test - self.mu_global).flatten()
            mahalanobis = diff.T @ self.Sigma_inv @ diff
            posterior = -0.5 * mahalanobis + np.log(self.P[i])
            g.append(posterior)
        return np.argmax(g) + 1


class NaiveBayes:
    def __init__(self, X_train, y_train):
        self.classes = np.unique(y_train[0, :])
        self.C = len(self.classes)
        self.p, self.N = X_train.shape
        
        self.X = [X_train[:, y_train[i, :] == 1] for i in range(self.C)]
        self.n = [Xi.shape[1] for Xi in self.X]
        
        self.mu = [None]*self.C
        self.var = [None]*self.C
        self.P = [None]*self.C
    
    def fit(self):
        for i in range(self.C):
            self.mu[i] = np.mean(self.X[i], axis=1)
            self.var[i] = np.var(self.X[i], axis=1, ddof=1)  # variância por feature
            self.P[i] = self.n[i] / self.N
    
    def predict(self, x_test):
        g = []
        for i in range(self.C):
            log_likelihood = 0
            for j in range(self.p):
                var_j = self.var[i][j]
                if var_j < 1e-8:
                    var_j = 1e-8
                term = -0.5 * np.log(2 * np.pi * var_j) - 0.5 * ((x_test[j] - self.mu[i][j])**2 / var_j)
                log_likelihood += term
            posterior = log_likelihood + np.log(self.P[i])
            g.append(posterior)
        return np.argmax(g) + 1


class MQO:
    def __init__(self):
        self.beta = None
    
    def fit(self, X, Y):
        # X: (N, p), Y: (N, C)
        ones = np.ones((X.shape[0], 1))
        X_aug = np.hstack([ones, X])  # adicionar intercepto: (N, p+1)
        self.beta = np.linalg.inv(X_aug.T @ X_aug) @ X_aug.T @ Y  # (p+1, C)
    
    def predict(self, X):
        ones = np.ones((X.shape[0], 1))
        X_aug = np.hstack([ones, X])
        Y_pred = X_aug @ self.beta  # (N, C)
        return np.argmax(Y_pred, axis=1) + 1  # retorna classe 1 a 5


class GaussianRegularized:
    def __init__(self, X_train, y_train, lambda_val):
        self.lambda_val = lambda_val
        self.classes = np.unique(y_train[0, :])
        self.C = len(self.classes)
        self.p, self.N = X_train.shape
        
        self.X = [X_train[:, y_train[i, :] == 1] for i in range(self.C)]
        self.n = [Xi.shape[1] for Xi in self.X]
        
        self.mu = [None]*self.C
        self.P = [None]*self.C
        self.Sigma = [None]*self.C
        self.Sigma_det = [None]*self.C
        self.Sigma_inv = [None]*self.C
        
        # Covariância global (usada na regularização)
        self.Sigma_global = np.cov(X_train)
    
    def fit(self):
        for i in range(self.C):
            self.mu[i] = np.mean(self.X[i], axis=1).reshape(self.p, 1)
            self.P[i] = self.n[i] / self.N
            
            Sigma_c = np.cov(self.X[i])
            # Regularização de Friedman: Σ̂_i = (1−λ)Σ_i + λΣ_total
            self.Sigma[i] = (1 - self.lambda_val) * Sigma_c + self.lambda_val * self.Sigma_global
            self.Sigma_det[i] = np.linalg.det(self.Sigma[i])
            self.Sigma_inv[i] = np.linalg.pinv(self.Sigma[i])
    
    def predict(self, x_test):
        g = []
        for i in range(self.C):
            diff = (x_test - self.mu[i]).flatten()
            mahalanobis = diff.T @ self.Sigma_inv[i] @ diff
            posterior = -0.5 * mahalanobis - 0.5 * np.log(self.Sigma_det[i]) + np.log(self.P[i])
            g.append(posterior)
        return np.argmax(g) + 1


# =============================
# 4. VALIDAÇÃO MONTE CARLO (R=500)
# =============================
np.random.seed(42)
R = 500

models = {
    "MQO tradicional": MQO(),
    "Classificador Gaussiano Tradicional": GaussianClassifier(X_bayes, Y_bayes),
    "Classificador Gaussiano(Cov. Igual)": GaussianEqualCov(X_bayes, Y_bayes),
    "Classificador Gaussiano(Cov. Agregada)": GaussianAggregated(X_bayes, Y_bayes),
    "Classificador de Bayes Ingênuo(Naive Bayes Classifier)": NaiveBayes(X_bayes, Y_bayes),
}

# Adicionar os 5 modelos regularizados
lambdas = [0.25, 0.5, 0.75, 1.0]
for l in lambdas:
    models[f"Classificador Gaussiano Regularizado(Friedman λ={l})"] = GaussianRegularized(X_bayes, Y_bayes, l)

# Armazenar resultados
results = {name: [] for name in models.keys()}

for r in range(R):
    print(f"\rRodada {r+1}/{R}", end="", flush=True)
    
    # Gerar índices aleatórios e dividir 80/20
    indices = np.arange(N)
    np.random.shuffle(indices)
    split_idx = int(0.8 * N)
    train_idx, test_idx = indices[:split_idx], indices[split_idx:]
    
    # Dados para MQO
    X_train_mqo = X_mqo[train_idx]      # (40000, 2)
    X_test_mqo = X_mqo[test_idx]        # (10000, 2)
    y_train_mqo = Y_raw[train_idx]      # (40000,)
    y_test_mqo = Y_raw[test_idx]        # (10000,)
    
    # Dados para modelos gaussianos (usam X_bayes e Y_bayes)
    X_train_bayes = X_bayes[:, train_idx]     # (2, 40000)
    X_test_bayes = X_bayes[:, test_idx]       # (2, 10000)
    Y_train_bayes = Y_bayes[:, train_idx]     # (5, 40000)
    Y_test_bayes = Y_bayes[:, test_idx]       # (5, 10000)
    
    for name, model in models.items():
        try:
            if name == "MQO tradicional":
                model.fit(X_train_mqo, Y_onehot[train_idx])
                y_pred = model.predict(X_test_mqo)
            elif "Regularizado" in name:
                model.fit()
                y_pred = np.array([model.predict(X_bayes[:, i].reshape(-1,1)) for i in test_idx])
            else:
                # Atualiza os dados de treino nos modelos gaussianos
                if hasattr(model, 'X'):  # Todos os gaussianos e naive bayes
                    model.X = [X_train_bayes[:, Y_train_bayes[i, :] == 1] for i in range(model.C)]
                    model.n = [Xi.shape[1] for Xi in model.X]
                    model.fit()
                y_pred = np.array([model.predict(X_bayes[:, i].reshape(-1,1)) for i in test_idx])
            
            acc = np.mean(y_pred == y_test_mqo)
            results[name].append(acc)
        except Exception as e:
            print(f"\nErro em {name}: {e}")
            results[name].append(0.0)

print("\n✅ Validação concluída!")

# =============================
# 5. RESULTADOS FINAIS — TABELA E GRÁFICO
# =============================
summary = {}
for model_name, scores in results.items():
    summary[model_name] = {
        'Média': np.mean(scores),
        'Desvio-Padrão': np.std(scores),
        'Maior Valor': np.max(scores),
        'Menor Valor': np.min(scores)
    }

df_results = np.array([[model, f"{summary[model]['Média']:.4f}", 
                        f"{summary[model]['Desvio-Padrão']:.4f}", 
                        f"{summary[model]['Maior Valor']:.4f}", 
                        f"{summary[model]['Menor Valor']:.4f}"] 
                       for model in models.keys()])

print("\n" + "="*80)
print("RESULTADOS DA TAREFA DE CLASSIFICAÇÃO (R=500 rodadas)")
print("="*80)
print(f"{'Modelo':<40} {'Média':<8} {'Desv.Pad.':<10} {'Maior':<8} {'Menor':<8}")
print("-"*80)
for row in df_results:
    print(f"{row[0]:<40} {row[1]:<8} {row[2]:<10} {row[3]:<8} {row[4]:<8}")

# =============================
# 6. GRÁFICO DE BOX PLOT
# =============================
plt.figure(figsize=(16, 10))
data_plot = [results[m] for m in models.keys()]
box = plt.boxplot(data_plot, labels=models.keys(), patch_artist=True, rot=90, vert=True)
plt.ylabel('Acurácia', fontsize=14)
plt.title('Distribuição das Acurácias por Modelo (Monte Carlo, R=500)', fontsize=16)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig('boxplot_acuracias.png', dpi=300, bbox_inches='tight')
plt.show()

# =============================
# 7. SALVAR RESULTADOS EM TXT (para relatório)
# =============================
with open('resultados_classificacao.txt', 'w') as f:
    f.write("RESULTADOS DA TAREFA DE CLASSIFICAÇÃO (R=500 rodadas)\n")
    f.write("="*80 + "\n")
    f.write(f"{'Modelo':<40} {'Média':<8} {'Desv.Pad.':<10} {'Maior':<8} {'Menor':<8}\n")
    f.write("-"*80 + "\n")
    for row in df_results:
        f.write(f"{row[0]:<40} {row[1]:<8} {row[2]:<10} {row[3]:<8} {row[4]:<8}\n")

print("\n📊 Resultados salvos em 'resultados_classificacao.txt' e 'boxplot_acuracias.png'")