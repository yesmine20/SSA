import numpy as np
import matplotlib.pyplot as plt

def objective_function(x):
    return np.sum(x**2)

def ssa_optimization(obj_func, dim, n_salps, max_iter, lb, ub):
    salp_positions = np.random.uniform(lb, ub, (n_salps, dim))
    food_pos = np.zeros(dim)
    food_score = float("inf")
    convergence_curve = []

    for iteration in range(max_iter):
        # Évaluation
        for i in range(n_salps):
            fitness = obj_func(salp_positions[i, :])
            if fitness < food_score:
                food_score = fitness
                food_pos = salp_positions[i, :].copy()

        # Coefficient c1 (décroissance exponentielle pour affiner la recherche)
        c1 = 2 * np.exp(-(4 * iteration / max_iter) ** 2)

        for i in range(n_salps):
            if i == 0:
                for j in range(dim):
                    c2, c3 = np.random.random(), np.random.random()
                    if c3 >= 0.5:
                        salp_positions[i, j] = food_pos[j] + c1 * ((ub - lb) * c2 + lb)
                    else:
                        salp_positions[i, j] = food_pos[j] - c1 * ((ub - lb) * c2 + lb)
            else:
                salp_positions[i, :] = 0.5 * (salp_positions[i, :] + salp_positions[i-1, :])

            salp_positions[i, :] = np.clip(salp_positions[i, :], lb, ub)

        convergence_curve.append(food_score)
    
    return food_pos, food_score, convergence_curve

# --- Paramètres augmentés pour atteindre 0 ---
dim = 5            # On réduit un peu la dimension pour converger plus vite
n_salps = 50       
max_iter = 500     
lb, ub = -10, 10   # Espace de recherche plus serré

best_pos, best_score, curve = ssa_optimization(objective_function, dim, n_salps, max_iter, lb, ub)

# --- Affichage du graphique ---
plt.figure(figsize=(10, 5))
plt.plot(curve, color='blue', linewidth=2)
plt.yscale('log') # Échelle logarithmique pour voir la précision proche de 0
plt.title("Convergence de l'algorithme SSA (Fonction Sphere)")
plt.xlabel("Itérations")
plt.ylabel("Meilleur Score (Log)")
plt.grid(True)
plt.show()

print(f"Meilleur score trouvé : {best_score:.15f}")
print(f"Position optimale : {best_pos}")