import numpy as np
import matplotlib.pyplot as plt

# Função para contar ataques
def count_attacks(state):
    attacks = 0
    n = len(state)
    for i in range(n):
        for j in range(i + 1, n):
            if state[i] == state[j] or abs(state[i] - state[j]) == abs(i - j):
                attacks += 1
    return attacks

# Perturbação: muda a linha de uma coluna aleatória
def perturb(state):
    new_state = state.copy()
    col = np.random.randint(0, 8)
    new_row = np.random.randint(0, 8)
    while new_row == new_state[col]:
        new_row = np.random.randint(0, 8)
    new_state[col] = new_row
    return new_state

# Função principal com visualização em tempo real
def simulated_annealing_8queens_visual(max_iter=10000, T0=100.0, alpha=0.995, pause_time=0.2):
    # Estado inicial
    current = np.random.randint(0, 8, size=8)
    current_attacks = count_attacks(current)
    best = current.copy()
    best_attacks = current_attacks

    T = T0
    history = [current_attacks]

    # Configuração do gráfico
    plt.ion()  # Ativa modo interativo
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_title(f"Ataques: {current_attacks} | T = {T:.2f}", fontsize=14)
    ax.set_xticks(range(8))
    ax.set_yticks(range(8))
    ax.grid(color='white', linestyle='-', linewidth=2)
    im = ax.imshow(np.zeros((8, 8)), cmap='gray', vmin=0, vmax=1)

    # Função para desenhar o tabuleiro
    def draw_board(state, attacks, T):
        ax.clear()
        ax.set_title(f"Ataques: {attacks} | T = {T:.2f}", fontsize=14)
        ax.set_xticks(range(8))
        ax.set_yticks(range(8))
        ax.grid(color='white', linestyle='-', linewidth=2)
        board = np.zeros((8, 8))
        for col, row in enumerate(state):
            board[row, col] = 1
        ax.imshow(board, cmap='gray', vmin=0, vmax=1)
        for col, row in enumerate(state):
            ax.text(col, row, '♛', fontsize=30, ha='center', va='center', color='gold')
        plt.draw()
        plt.pause(pause_time)

    # Desenha estado inicial
    draw_board(current, current_attacks, T)

    for i in range(max_iter):
        candidate = perturb(current)
        cand_attacks = count_attacks(candidate)

        delta = cand_attacks - current_attacks

        if delta < 0 or np.random.rand() < np.exp(-delta / T):
            current = candidate
            current_attacks = cand_attacks

            if current_attacks < best_attacks:
                best = current.copy()
                best_attacks = current_attacks

            # Atualiza visualização
            draw_board(current, current_attacks, T)

            if current_attacks == 0:
                print(f"\n✅ Solução ótima encontrada na iteração {i+1}!")
                draw_board(current, current_attacks, T)
                plt.ioff()
                plt.show()
                return current, current_attacks, history

        T = max(T * alpha, 1e-3)
        history.append(current_attacks)

    plt.ioff()
    plt.show()
    print(f"\n⚠️  Máximo de iterações atingido. Melhor solução: {best_attacks} ataques.")
    return best, best_attacks, history

# Executa o algoritmo com visualização
np.random.seed()  # diferente a cada execução
solution, attacks, hist = simulated_annealing_8queens_visual(
    max_iter=5000,
    T0=100.0,
    alpha=0.995,
    pause_time=0.2
)

# Mostra histórico
plt.figure(figsize=(10, 4))
plt.plot(hist)
plt.title("Evolução do número de ataques")
plt.xlabel("Iteração")
plt.ylabel("Ataques")
plt.grid(True)
plt.show()