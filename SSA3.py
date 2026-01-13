import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# 1. Fonction Objectif
def objective_function(x):
    return x**2

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
        
        c1 = 2 * np.exp(-(4 * t / max_iter)**2)
        
        new_salps = np.zeros_like(salps)
        for i in range(n_salps):
            if i == 0:
                c2, c3 = np.random.random(), np.random.random()
                if c3 >= 0.5:
                    new_salps[i] = best_pos + c1 * ((ub - lb) * c2 + lb)
                else:
                    new_salps[i] = best_pos - c1 * ((ub - lb) * c2 + lb)
            else:
                new_salps[i] = 0.5 * (salps[i] + salps[i-1])
            new_salps[i] = np.clip(new_salps[i], lb, ub)
            
        salps = new_salps
        h_pos.append(salps.copy())
        h_best_pos.append(best_pos)
        h_curve.append(best_score)
        h_c1.append(c1)
        
    return h_pos, h_best_pos, h_curve, h_c1

# 3. Paramètres
n_salps, max_iter = 15, 100
lb, ub = -50, 50
h_pos, h_best, h_curve, h_c1 = ssa_full_history(n_salps, max_iter, lb, ub)

# 4. Interface Graphique (2 Graphiques)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

# --- Graphique 1 : Mouvement ---
x_range = np.linspace(lb, ub, 400)
ax1.plot(x_range, objective_function(x_range), color='lightgray', label="f(x)=x²")
salp_dots, = ax1.plot([], [], 'ro', label="Salpes (Swarm)")
food_dot, = ax1.plot([], [], 'gD', markersize=10, label="Cible (Nourriture)")
phase_text = ax1.text(0.5, 0.9, '', transform=ax1.transAxes, ha='center', fontsize=12, fontweight='bold')
ax1.set_title("Espace de Recherche (Mouvement)")
ax1.set_xlabel("Position x")
ax1.set_ylabel("f(x)")
ax1.legend()

# --- Graphique 2 : Courbe de Convergence ---
conv_line, = ax2.plot([], [], color='blue', lw=2)
ax2.set_xlim(0, max_iter)
ax2.set_ylim(0, max(h_curve))
ax2.set_title("Courbe de Convergence")
ax2.set_xlabel("Itérations")
ax2.set_ylabel("Meilleur Score trouvé")
ax2.grid(True, linestyle=':')

# 5. Fonction de mise à jour (Correction de l'erreur ici)
def update(frame):
    # Mise à jour Swarm
    curr_s = h_pos[frame]
    salp_dots.set_data(curr_s, objective_function(curr_s))
    
    # CORRECTION : On met les valeurs dans des listes [ ]
    food_dot.set_data([h_best[frame]], [objective_function(h_best[frame])])
    
    # Texte des phases
    c1 = h_c1[frame]
    if c1 > 1.0:
        phase_text.set_text("PHASE : EXPLORATION"); phase_text.set_color("red")
    else:
        phase_text.set_text("PHASE : EXPLOITATION"); phase_text.set_color("green")
    
    # Mise à jour Courbe de Convergence
    iterations = np.arange(frame)
    scores = h_curve[:frame]
    conv_line.set_data(iterations, scores)
    
    # Auto-ajustement de l'échelle de la courbe
    if frame > 1:
        ax2.set_ylim(-1, max(h_curve[:frame]) * 1.1)

    return salp_dots, food_dot, phase_text, conv_line

# 6. Lancement de l'animation
ani = animation.FuncAnimation(fig, update, frames=max_iter, interval=80, blit=True)
plt.tight_layout()
plt.show()

print(f"Optimisation terminée. Score final : {h_curve[-1]:.2e}")