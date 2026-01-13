import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# 1. Nouvelle Fonction Objectif : Rastrigin (plus complexe que x²)
# Le minimum global est en x=0, où f(x)=0
def objective_function(x):
    return 10 + x**2 - 10 * np.cos(2 * np.pi * x)

# 2. Algorithme SSA avec historique
def ssa_full_history(n_salps, max_iter, lb, ub):
    salps = np.random.uniform(lb, ub, n_salps)
    best_score = float('inf')
    best_pos = 0
    
    h_pos = []     
    h_best_pos = [] 
    h_curve = []    
    h_c1 = []       
    
    for t in range(max_iter):
        for s in salps:
            score = objective_function(s)
            if score < best_score:
                best_score = score
                best_pos = s
        
        # Paramètre crucial de l'algorithme pour équilibrer Exploration/Exploitation
        c1 = 2 * np.exp(-(4 * t / max_iter)**2)
        
        new_salps = np.zeros_like(salps)
        for i in range(n_salps):
            if i == 0: # Leader
                c2, c3 = np.random.random(), np.random.random()
                if c3 >= 0.5:
                    new_salps[i] = best_pos + c1 * ((ub - lb) * c2 + lb)
                else:
                    new_salps[i] = best_pos - c1 * ((ub - lb) * c2 + lb)
            else: # Followers
                new_salps[i] = 0.5 * (salps[i] + salps[i-1])
            
            new_salps[i] = np.clip(new_salps[i], lb, ub)
            
        salps = new_salps
        h_pos.append(salps.copy())
        h_best_pos.append(best_pos)
        h_curve.append(best_score)
        h_c1.append(c1)
        
    return h_pos, h_best_pos, h_curve, h_c1

# 3. Paramètres (Ajustés pour cette fonction)
n_salps, max_iter = 20, 100
lb, ub = -5.12, 5.12 # Limites standards pour Rastrigin
h_pos, h_best, h_curve, h_c1 = ssa_full_history(n_salps, max_iter, lb, ub)

# 4. Interface Graphique
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

# --- Graphique 1 : Mouvement sur Rastrigin ---
x_range = np.linspace(lb, ub, 1000)
ax1.plot(x_range, objective_function(x_range), color='lightgray', label="f(x) Rastrigin", zorder=1)
salp_dots, = ax1.plot([], [], 'ro', label="Salpes (Swarm)", zorder=3)
food_dot, = ax1.plot([], [], 'gD', markersize=10, label="Cible (Nourriture)", zorder=4)
phase_text = ax1.text(0.5, 0.9, '', transform=ax1.transAxes, ha='center', fontsize=12, fontweight='bold')
ax1.set_title("Espace de Recherche (Fonction Multi-modale)")
ax1.set_xlabel("Position x")
ax1.set_ylabel("f(x)")
ax1.legend()

# --- Graphique 2 : Courbe de Convergence ---
conv_line, = ax2.plot([], [], color='blue', lw=2)
ax2.set_xlim(0, max_iter)
ax2.set_title("Courbe de Convergence")
ax2.set_xlabel("Itérations")
ax2.set_ylabel("Meilleur Score trouvé")
ax2.grid(True, linestyle=':')

# 5. Fonction de mise à jour
def update(frame):
    # Mise à jour Swarm
    curr_s = h_pos[frame]
    salp_dots.set_data(curr_s, objective_function(curr_s))
    
    # Cible
    food_dot.set_data([h_best[frame]], [objective_function(h_best[frame])])
    
    # Texte des phases
    c1 = h_c1[frame]
    if c1 > 0.5: # Seuil d'ajustement pour l'affichage
        phase_text.set_text("PHASE : EXPLORATION"); phase_text.set_color("red")
    else:
        phase_text.set_text("PHASE : EXPLOITATION"); phase_text.set_color("green")
    
    # Mise à jour Courbe de Convergence
    iterations = np.arange(frame)
    scores = h_curve[:frame]
    conv_line.set_data(iterations, scores)
    
    # Auto-ajustement dynamique de l'échelle
    if frame > 1:
        ax2.set_ylim(-1, max(h_curve[:frame]) * 1.1)

    return salp_dots, food_dot, phase_text, conv_line

# 6. Lancement
ani = animation.FuncAnimation(fig, update, frames=max_iter, interval=100, blit=True)
plt.tight_layout()
plt.show()

print(f"Optimisation terminée. Meilleur score final : {h_curve[-1]:.6f}")