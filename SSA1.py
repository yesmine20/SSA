import numpy as np
import matplotlib.pyplot as plt

# 1. La fonction la plus simple possible : f(x) = x²
def simple_function(x):
    return x**2

def ssa_simple(n_salps, max_iter):
    # Initialisation : 10 salpes placées au hasard entre -50 et 50
    lb, ub = -50, 50
    salps = np.random.uniform(lb, ub, n_salps)
    
    best_score = float('inf')
    best_pos = 0
    history = []

    for t in range(max_iter):
        # Trouver la meilleure salpe (la nourriture)
        for s in salps:
            score = simple_function(s)
            if score < best_score:
                best_score = score
                best_pos = s

        # Mettre à jour le coefficient c1 (le guide)
        c1 = 2 * np.exp(-(4 * t / max_iter)**2)

        # Déplacer les salpes
        for i in range(n_salps):
            if i == 0: # Le Leader
                c2 = np.random.random()
                c3 = np.random.random()
                if c3 >= 0.5:
                    salps[i] = best_pos + c1 * ((ub - lb) * c2 + lb)
                else:
                    salps[i] = best_pos - c1 * ((ub - lb) * c2 + lb)
            else: # Les Suiveurs
                salps[i] = 0.5 * (salps[i] + salps[i-1])
            
            # Garder les salpes dans les bornes
            salps[i] = np.clip(salps[i], lb, ub)
        
        history.append(best_score)

    return best_pos, best_score, history

# --- Exécution ---
pos, score, curve = ssa_simple(n_salps=10, max_iter=50)

print(f"Position finale (x) : {pos:.6f}")
print(f"Score final (f(x)) : {score:.6e}") # .6e affiche l'exposant (ex: 10^-12)

# Graphique de la fonction et de la position finale
x_range = np.linspace(-50, 50, 400)
y_range = simple_function(x_range)

plt.figure(figsize=(8, 4))
plt.plot(x_range, y_range, label="Fonction f(x)=x²")
plt.plot(pos, score, 'ro', label="Position finale des salpes")
plt.title("Le SSA a trouvé le fond de la cuvette !")
plt.legend()
plt.show()