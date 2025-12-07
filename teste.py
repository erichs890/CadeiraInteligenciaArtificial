import numpy as np
from numpy.linalg import inv
import matplotlib.pyplot as plt

# Carregar os dados
data = np.loadtxt("aerogerador.dat", delimiter="\t")  # ATENÇÃO: o arquivo é .dat, não .csv!
X = data[:, :-1]  # Primeira coluna: velocidade do vento
N, p = X.shape
y = data[:, -1:]  # Segunda coluna: potência gerada

# Plot inicial 
fig = plt.figure(1)
plt.scatter(X[:, 0], y[:, 0], color="pink", edgecolor="k", s=30)
plt.xlabel("Velocidade do Vento (m/s)")
plt.ylabel("Potência Gerada (kW)")
plt.title("Relação entre Velocidade do Vento e Potência Gerada")
plt.grid(True, alpha=0.3)

# Adicionar coluna de uns para o intercepto
X = np.hstack((np.ones((N, 1)), X))  # Agora X tem 2 colunas: [1, velocidade]

# Definição dos valores de lambda
lambdas = [0, 0.25, 0.5, 0.75, 1]
num_lambdas = len(lambdas)

# Listas para armazenar resultados por lambda
mediaSSE_ridge = [[] for _ in range(num_lambdas)]   # SSE por lambda
mediaMSE_ridge = [[] for _ in range(num_lambdas)]   # MSE por lambda
MQOSSE = []  # MQO/OLS (sem regularização)
MQOMSE = []

# Modelo Média (não muda)
mediaSSE_media = []
mediaMSE_media = []

# Definição da quantidade de rodadas
rodadas = 500
controle_plot = True

for r in range(rodadas):
    # Embaralhar o conjunto de dados
    idx = np.random.permutation(N)
    Xr = X[idx, :]
    yr = y[idx, :]

    # Particionamento (80% treino, 20% teste)
    X_treino = Xr[:int(0.8 * N), :]
    y_treino = yr[:int(0.8 * N), :]

    X_teste = Xr[int(0.8 * N):, :]
    y_teste = yr[int(0.8 * N):, :]

    # Modelo Média
    beta_media = np.array([[np.mean(y_treino)], [0]])
    y_hat_teste_media = X_teste @ beta_media
    y_hat_treino_media = X_treino @ beta_media

    SSE_media = np.sum((y_teste - y_hat_teste_media)**2)
    MSE_media = np.mean((y_teste - y_hat_teste_media)**2)
    mediaSSE_media.append(SSE_media)
    mediaMSE_media.append(MSE_media)

    # Modelo MQO/OLS (sem regularização)
    beta_ols = inv(X_treino.T @ X_treino) @ X_treino.T @ y_treino
    y_hat_teste_ols = X_teste @ beta_ols
    SSE_ols = np.sum((y_teste - y_hat_teste_ols)**2)
    MSE_ols = np.mean((y_teste - y_hat_teste_ols)**2)
    MQOSSE.append(SSE_ols)
    MQOMSE.append(MSE_ols)

    # Modelo Ridge para cada lambda
    for idx_lambda, lam in enumerate(lambdas):
        # Matriz de penalidade: não penaliza o intercepto (primeiro parâmetro)
        # I_p+1 com 0 na posição (0,0) e λ nas demais
        Lambda_matrix = np.eye(X_treino.shape[1])
        Lambda_matrix[0, 0] = 0  # Intercepto não é penalizado

        # Solução Ridge: (X^T X + λ Λ)^{-1} X^T y
        A = X_treino.T @ X_treino + lam * Lambda_matrix
        b = X_treino.T @ y_treino
        beta_ridge = inv(A) @ b

        # Previsões no teste
        y_hat_teste_ridge = X_teste @ beta_ridge
        SSE_ridge = np.sum((y_teste - y_hat_teste_ridge)**2)
        MSE_ridge = np.mean((y_teste - y_hat_teste_ridge)**2)

        mediaSSE_ridge[idx_lambda].append(SSE_ridge)
        mediaMSE_ridge[idx_lambda].append(MSE_ridge)

    # Plot apenas na primeira rodada
    if controle_plot:
        fig = plt.figure(2)
        plt.scatter(X_treino[:, 1], y_treino[:, 0], color="pink", edgecolor="k", s=30)
        plt.xlabel("Velocidade do Vento (m/s)")
        plt.ylabel("Potência Gerada (kW)")
        plt.title("Dados de Treino com Modelos (Média, MQO, Ridge)")

        x_line = np.linspace(X_treino[:, 1].min(), X_treino[:, 1].max(), 100)
        y_media_line = beta_media[0, 0] + beta_media[1, 0] * x_line
        plt.plot(x_line, y_media_line, color='blue', label='Modelo Média')

        y_ols_line = beta_ols[0, 0] + beta_ols[1, 0] * x_line
        plt.plot(x_line, y_ols_line, color='red', label='MQO/OLS')

        # Plotar Ridge para cada lambda
        colors = ['green', 'orange', 'purple', 'brown', 'cyan']
        for idx_lambda, lam in enumerate(lambdas):
            beta_ridge = inv(X_treino.T @ X_treino + lam * np.eye(X_treino.shape[1]) * np.array([0, 1])) @ X_treino.T @ y_treino
            y_ridge_line = beta_ridge[0, 0] + beta_ridge[1, 0] * x_line
            plt.plot(x_line, y_ridge_line, color=colors[idx_lambda], linestyle='--', label=f'Ridge λ={lam}')

        plt.legend()
        plt.grid(True, alpha=0.3)
        controle_plot = False

# Resultados finais: Médias por modelo
print("\n" + "="*70)
print("RESULTADOS FINAIS (MÉDIAS DAS 500 RODADAS)")
print("="*70)

# Modelo Média
mean_SSE_media = np.mean(mediaSSE_media)
mean_MSE_media = np.mean(mediaMSE_media)
print(f"Média dos SSE (Modelo Média): {mean_SSE_media:.5f}")
print(f"Média dos MSE (Modelo Média): {mean_MSE_media:.5f}")

# Modelo MQO/OLS
mean_SSE_ols = np.mean(MQOSSE)
mean_MSE_ols = np.mean(MQOMSE)
print(f"Média dos SSE (MQO/OLS): {mean_SSE_ols:.5f}")
print(f"Média dos MSE (MQO/OLS): {mean_MSE_ols:.5f}")

# Modelos Ridge
for idx, lam in enumerate(lambdas):
    mean_SSE_ridge = np.mean(mediaSSE_ridge[idx])
    mean_MSE_ridge = np.mean(mediaMSE_ridge[idx])
    print(f"Média dos SSE (Ridge λ={lam}): {mean_SSE_ridge:.5f}")
    print(f"Média dos MSE (Ridge λ={lam}): {mean_MSE_ridge:.5f}")

# Desvio padrão
print("\n" + "="*50)
print("DESVIO PADRÃO")
print("="*50)
print(f"Desvio Padrão SSE (Média): {np.std(mediaSSE_media):.5f}")
print(f"Desvio Padrão MSE (Média): {np.std(mediaMSE_media):.5f}")
print(f"Desvio Padrão SSE (MQO/OLS): {np.std(MQOSSE):.5f}")
print(f"Desvio Padrão MSE (MQO/OLS): {np.std(MQOMSE):.5f}")

for idx, lam in enumerate(lambdas):
    print(f"Desvio Padrão SSE (Ridge λ={lam}): {np.std(mediaSSE_ridge[idx]):.5f}")
    print(f"Desvio Padrão MSE (Ridge λ={lam}): {np.std(mediaMSE_ridge[idx]):.5f}")

# Valores mínimos e máximos
print("\n" + "="*50)
print("VALORES MÍNIMOS")
print("="*50)
print(f"Min SSE (Média): {np.min(mediaSSE_media):.5f}")
print(f"Min MSE (Média): {np.min(mediaMSE_media):.5f}")
print(f"Min SSE (MQO/OLS): {np.min(MQOSSE):.5f}")
print(f"Min MSE (MQO/OLS): {np.min(MQOMSE):.5f}")

for idx, lam in enumerate(lambdas):
    print(f"Min SSE (Ridge λ={lam}): {np.min(mediaSSE_ridge[idx]):.5f}")
    print(f"Min MSE (Ridge λ={lam}): {np.min(mediaMSE_ridge[idx]):.5f}")

print("\n" + "="*50)
print("VALORES MÁXIMOS")
print("="*50)
print(f"Max SSE (Média): {np.max(mediaSSE_media):.5f}")
print(f"Max MSE (Média): {np.max(mediaMSE_media):.5f}")
print(f"Max SSE (MQO/OLS): {np.max(MQOSSE):.5f}")
print(f"Max MSE (MQO/OLS): {np.max(MQOMSE):.5f}")

for idx, lam in enumerate(lambdas):
    print(f"Max SSE (Ridge λ={lam}): {np.max(mediaSSE_ridge[idx]):.5f}")
    print(f"Max MSE (Ridge λ={lam}): {np.max(mediaMSE_ridge[idx]):.5f}")

plt.show()