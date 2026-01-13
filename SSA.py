import numpy as np

def objective_function(x):
    # Fonction Sphere : f(x) = sum(x^2). Minimum à (0,0...0)
    return np.sum(x**2)

def ssa_optimization(obj_func, dim, n_salps, max_iter, lb, ub):
    # Initialisation des positions des salpes
    salp_positions = np.random.uniform(lb, ub, (n_salps, dim))
    food_pos = np.zeros(dim)
    food_score = float("inf")
    
    # Historique pour le suivi de la convergence
    convergence_curve = []

    for iteration in range(max_iter):
        # 1. Évaluation du score (Trouver la nourriture = meilleure position)
        for i in range(n_salps):
            fitness = obj_func(salp_positions[i, :])
            if fitness < food_score:
                food_score = fitness
                food_pos = salp_positions[i, :].copy()

        # Coefficient c1 : équilibre entre exploration et exploitation
        # Il diminue de façon exponentielle au fil des itérations
        c1 = 2 * np.exp(-(4 * iteration / max_iter) ** 2)

        for i in range(n_salps):
            if i == 0:
                # Mise à jour du Leader (la première salpe)
                for j in range(dim):
                    c2 = np.random.random()
                    c3 = np.random.random()
                    
                    if c3 >= 0.5:
                        salp_positions[i, j] = food_pos[j] + c1 * ((ub - lb) * c2 + lb)
                    else:
                        salp_positions[i, j] = food_pos[j] - c1 * ((ub - lb) * c2 + lb)
            else:
                # Mise à jour des Suiveurs (le reste de la chaîne)
                # Chaque salpe suit la position de celle qui la précède
                salp_positions[i, :] = 0.5 * (salp_positions[i, :] + salp_positions[i-1, :])

            # Respect des bornes (clamping)
            salp_positions[i, :] = np.clip(salp_positions[i, :], lb, ub)

        convergence_curve.append(food_score)
        if iteration % 10 == 0:
            print(f"Itération {iteration}: Meilleur Score = {food_score:.6f}")

    return food_pos, food_score, convergence_curve

# --- Paramètres ---
dim = 10           # Nombre de variables de décision
n_salps = 30       # Taille de la population
max_iter = 100     # Nombre d'itérations
lb = -100          # Borne inférieure
ub = 100           # Borne supérieure

# Exécution
best_pos, best_score, curve = ssa_optimization(objective_function, dim, n_salps, max_iter, lb, ub)

print(f"\nRésultat final : {best_score}")