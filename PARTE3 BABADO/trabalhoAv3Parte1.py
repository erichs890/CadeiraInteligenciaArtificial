import numpy as np
import matplotlib.pyplot as plt

# === Funções objetivo ===
def f1(x): return x[0]**2 + x[1]**2
def f2(x): return np.exp(-(x[0]**2 + x[1]**2)) + 2 * np.exp(-((x[0] - 1.7)**2 + (x[1] - 1.7)**2))
def f3(x): 
    term1 = -20 * np.exp(-0.2 * np.sqrt(0.5 * (x[0]**2 + x[1]**2)))
    term2 = -np.exp(0.5 * (np.cos(2*np.pi*x[0]) + np.cos(2*np.pi*x[1])))
    return term1 + term2 + 20 + np.exp(1)
def f4(x): 
    return (x[0]**2 - 10 * np.cos(2*np.pi*x[0]) + 10) + (x[1]**2 - 10 * np.cos(2*np.pi*x[1]) + 10)
def f5(x): 
    return (x[0] * np.cos(x[0]) / 20.0) + 2 * np.exp(-x[0]**2 - (x[1] - 1)**2) + 0.01 * x[0] * x[1]
def f6(x): 
    return x[0] * np.sin(4*np.pi*x[0]) - x[1] * np.sin(4*np.pi*x[1] + np.pi) + 1

problems = [
    ("f1", f1, [[-100, 100], [-100, 100]], "min"),
    ("f2", f2, [[-2, 4], [-2, 5]], "max"),
    ("f3", f3, [[-8, 8], [-8, 8]], "min"),
    ("f4", f4, [[-5.12, 5.12], [-5.12, 5.12]], "min"),
    ("f5", f5, [[-10, 10], [-10, 10]], "max"),
    ("f6", f6, [[-1, 3], [-1, 3]], "max")
]

# === Algoritmos com histórico ===
class HillClimbing:
    def __init__(self, func, bounds, optimization, max_iter=1000, no_improve_limit=50, epsilon=0.1, track=False):
        self.func = func
        self.bounds = np.array(bounds)
        self.optimization = optimization
        self.max_iter = max_iter
        self.no_improve_limit = no_improve_limit
        self.epsilon = epsilon
        self.track = track

    def run(self):
        x = self.bounds[:, 0].astype(float)
        f_best = self.func(x)
        path = [x.copy()] if self.track else []
        no_improve = 0
        n = len(x)

        for _ in range(self.max_iter):
            x_new = x + np.random.uniform(-self.epsilon, self.epsilon, size=n)
            x_new = np.clip(x_new, self.bounds[:, 0], self.bounds[:, 1])
            f_new = self.func(x_new)

            better = (self.optimization == "min" and f_new < f_best) or \
                     (self.optimization == "max" and f_new > f_best)

            if better:
                x, f_best = x_new, f_new
                no_improve = 0
                if self.track:
                    path.append(x.copy())
            else:
                no_improve += 1
                if no_improve >= self.no_improve_limit:
                    break
        if self.track:
            return f_best, np.array(path)
        return f_best

class LocalRandomSearch:
    def __init__(self, func, bounds, optimization, max_iter=1000, no_improve_limit=50, sigma=0.5, track=False):
        self.func = func
        self.bounds = np.array(bounds)
        self.optimization = optimization
        self.max_iter = max_iter
        self.no_improve_limit = no_improve_limit
        self.sigma = sigma
        self.track = track

    def run(self):
        x = np.random.uniform(self.bounds[:, 0], self.bounds[:, 1])
        f_best = self.func(x)
        path = [x.copy()] if self.track else []
        no_improve = 0
        n = len(x)

        for _ in range(self.max_iter):
            x_new = x + np.random.normal(0, self.sigma, size=n)
            x_new = np.clip(x_new, self.bounds[:, 0], self.bounds[:, 1])
            f_new = self.func(x_new)

            better = (self.optimization == "min" and f_new < f_best) or \
                     (self.optimization == "max" and f_new > f_best)

            if better:
                x, f_best = x_new, f_new
                no_improve = 0
                if self.track:
                    path.append(x.copy())
            else:
                no_improve += 1
                if no_improve >= self.no_improve_limit:
                    break
        if self.track:
            return f_best, np.array(path)
        return f_best

class GlobalRandomSearch:
    def __init__(self, func, bounds, optimization, max_iter=1000, no_improve_limit=50, track=False):
        self.func = func
        self.bounds = np.array(bounds)
        self.optimization = optimization
        self.max_iter = max_iter
        self.no_improve_limit = no_improve_limit
        self.track = track

    def run(self):
        x_best = np.random.uniform(self.bounds[:, 0], self.bounds[:, 1])
        f_best = self.func(x_best)
        path = [x_best.copy()] if self.track else []
        no_improve = 0

        for _ in range(self.max_iter):
            x_cand = np.random.uniform(self.bounds[:, 0], self.bounds[:, 1])
            f_cand = self.func(x_cand)

            better = (self.optimization == "min" and f_cand < f_best) or \
                     (self.optimization == "max" and f_cand > f_best)

            if better:
                x_best, f_best = x_cand, f_cand
                no_improve = 0
                if self.track:
                    path.append(x_best.copy())
            else:
                no_improve += 1
                if no_improve >= self.no_improve_limit:
                    break
        if self.track:
            return f_best, np.array(path)
        return f_best

def manual_mode(arr):
    rounded = np.round(arr, decimals=6)
    values, counts = np.unique(rounded, return_counts=True)
    return values[np.argmax(counts)]

# === Rodar experimentos (sem plot) ===
R = 100
results = {}
for name, func, bounds, opt in problems:
    print(f"\nResolvendo {name} ({opt})...")
    hc_vals = []
    lrs_vals = []
    grs_vals = []

    for r in range(R):
        hc_vals.append(HillClimbing(func, bounds, opt).run())
        lrs_vals.append(LocalRandomSearch(func, bounds, opt).run())
        grs_vals.append(GlobalRandomSearch(func, bounds, opt).run())

    results[name] = {
        "Hill Climbing": manual_mode(np.array(hc_vals)),
        "LRS": manual_mode(np.array(lrs_vals)),
        "GRS": manual_mode(np.array(grs_vals))
    }

# === Exibir tabela ===
print("\n" + "="*60)
print("TABELA FINAL (Moda dos valores da função objetivo)")
print("="*60)
print(f"{'Função':<8} {'HC':>12} {'LRS':>12} {'GRS':>12}")
print("-"*60)
for name in results:
    r = results[name]
    print(f"{name:<8} {r['Hill Climbing']:>12.6f} {r['LRS']:>12.6f} {r['GRS']:>12.6f}")

# === Plotar gráficos com trajetórias (uma rodada por algoritmo) ===
for name, func, bounds, opt in problems:
    print(f"\nGerando gráfico para {name}...")

    # Gerar grade para contorno
    x_range = np.linspace(bounds[0][0], bounds[0][1], 200)
    y_range = np.linspace(bounds[1][0], bounds[1][1], 200)
    X, Y = np.meshgrid(x_range, y_range)
    Z = np.array([[func([x, y]) for x in x_range] for y in y_range])

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    titles = ["Hill Climbing", "Local Random Search", "Global Random Search"]
    algs = [
        HillClimbing(func, bounds, opt, track=True),
        LocalRandomSearch(func, bounds, opt, track=True),
        GlobalRandomSearch(func, bounds, opt, track=True)
    ]

    for ax, title, alg in zip(axes, titles, algs):
        f_val, path = alg.run()
        # Plot contorno
        contour = ax.contourf(X, Y, Z, levels=30, cmap='viridis', alpha=0.7)
        fig.colorbar(contour, ax=ax, shrink=0.8)
        # Plot trajetória
        if len(path) > 1:
            ax.plot(path[:, 0], path[:, 1], 'r.-', markersize=4, linewidth=1, label='Trajetória')
            ax.plot(path[0, 0], path[0, 1], 'go', label='Início')
            ax.plot(path[-1, 0], path[-1, 1], 'ro', label='Fim')
        ax.set_title(f"{name} - {title}\nÓtimo: {f_val:.5f}")
        ax.set_xlabel("x1")
        ax.set_ylabel("x2")
        ax.legend()
        ax.grid(True, linestyle='--', alpha=0.5)

    plt.tight_layout()
    plt.show()