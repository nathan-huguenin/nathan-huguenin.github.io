import numpy as np
import numpy.random as rd
import matplotlib.pyplot as plt
import scipy.stats as st
from math import *

## Simulation de la chaîne de Markov

# Fonction qui simule la trace d'une MC (P : matrice de transition, X0 : état initial, n : nombre de pas souhaité)
def sim_MC(P, X0, n):
    state = X0
    trace = [state]

    for _ in range(n):
        # On choisit l'état d'après selon la matrice de transition P
        next_state = np.random.choice(len(P), p=P[state])
        trace.append(next_state)
        state = next_state

    return trace

# La matrice P de l'exemple
P = np.array([
    [1, 1, 1, 1, 1, 1, 0],
    [1, 0, 1, 1, 1, 1, 1],
    [1, 0, 0, 1, 1, 2, 1],
    [2, 0, 0, 0, 1, 2, 1],
    [2, 0, 0, 0, 1, 2, 1],
    [1, 0, 1, 1, 1, 1, 1],
    [0, 0, 0, 0, 0, 0, 6]
])/6

## Loi du temps d'absorption

# Simulation de la MC jusqu'au temps d'absorption
def sim_absorption_time(P, X0):
    state = X0
    T = 0

    while state != 6:  # L'état absorbant est 6
        state = np.random.choice(len(P), p=P[state])
        T += 1

    return T

# Nombre de simulations
N_sim = 1000

# Stockage des temps d'absorption
absorption_times = []

# Simulations des temps d'absorption
for _ in range(N_sim):
    X0 = 0 # ou random.randint(0, 5) par exemple
    T = sim_absorption_time(P, X0)
    absorption_times.append(T)

# Tracé de l'histogramme des temps d'absorption
plt.hist(absorption_times, bins=30, edgecolor='black', density=True)
plt.title(f'Distribution empirique du temps d absorption')
plt.xlabel('Temps d absorption')
plt.ylabel('Fréquence')
plt.grid(True)
plt.show()

# On observe bien la majoration par une géométrique. On peut également simuler le temps d'absorption en partant de la loi presque-invariante tilde_mu.

# La loi presque-invariante
tilde_mu = np.array([0.251, 0.048, 0.109, 0.130, 0.190, 0.272])

# Simulation de la MC jusqu'au temps d'absorption avec loi initiale tilde_mu
def sim_absorption_time(P):
    state = np.random.choice(len(P)-1, p=tilde_mu)
    T = 0

    while state != 6:  # L'état absorbant est 6
        state = np.random.choice(len(P), p=P[state])
        T += 1

    return T

# Nombre de simulations
N_sim = 1000

# Stockage des temps d'absorption
absorption_times = []

# Simulations des temps d'absorption
for _ in range(N_sim):
    T = sim_absorption_time(P)
    absorption_times.append(T)

# Valeur supérieure au-delà de laquelle on ne prend plus en compte le temps d'absorption
N_max = 50

# Distribution empirique de T
absorption_time_counts = np.zeros(N_max)
for i in absorption_times:
    if i<50:
        absorption_time_counts[i] += 1

# Tracé de la f.d.r. empirique des temps d'absorption
plt.plot(range(N_max),np.cumsum(absorption_time_counts)/N_sim, label='Empirique')
# comparaison avec la géométrique donnée
plt.plot(range(N_max),st.geom.cdf(range(N_max),1-0.875), label='Théorique', color='red')
plt.title(f'Fonction de répartition empirique du temps d absorption \n (loi initiale presque-invariante)')
plt.xlabel('Temps d absorption')
plt.ylabel('Fréquence')
plt.legend()
plt.grid(True)
plt.show()


## Comparaison avec la mesure presque-invariante

# Modification de sim_MC pour ne garder en mémoire que le dernier état
def sim_last_step(P, X0, n):
    state = X0

    for _ in range(n):
        # On choisit l'état d'après selon la matrice de transition P
        state = np.random.choice(len(P), p=P[state])

    return state

# Nombre de simulations et d'étapes
N_sim = 1000
n = 10

# Stockage des états finaux de la chaîne
final_states = []

# Simulation des états finaux de la chaîne
for _ in range(N_sim):
    X0 = 0 # ou random.randint(0, 5) par exemple
    final_state = sim_last_step(P, X0, n)
    final_states.append(final_state)

# Distribution empirique de X_n après n étapes
state_counts = np.zeros(7)
for i in final_states:
    state_counts[i] += 1

# Normalisation pour avoir une proba
mu_n = state_counts / N_sim

# Tracé de la distribution empirique après n étapes
plt.bar(range(7), mu_n, tick_label=[1, 2, 3, 4, 5, 6, 7])
plt.title(f'Distribution de X_n après {n} pas')
plt.xlabel('Etat')
plt.ylabel('Probabilité')
plt.grid(True)
plt.show()

# On observe qu'on reconnait la figure 2 dans la restriction de la simulation aux 6 premiers états.

## Convergence exponentielle vers la mesure presque-invariante

# Premièrement, on peut simuler la loi de la chaîne conditionnée à ne pas atteindre l'état 7

def sim_last_step_conditioned(P, X0, n):
    state = X0
    for step in range(n):
        state = np.random.choice(len(P), p=P[state])
        if state == 6:
            return None  # On indique que la chaîne a été absorbée
    return state

# Nombre de simulations et de pas
N_sim = 1000
n = 10

# Stockage des états finaux de la chaîne
final_states = []

# Simulation
for _ in range(N_sim):
    X0 = 0 # ou random.randint(0, 5)
    final_state = sim_last_step_conditioned(P, X0, n)
    if final_state is not None:  # On ne retient que les chaînes non absorbées
        final_states.append(final_state)

# Distribution empirique
state_counts = np.zeros(7)
for i in final_states:
    state_counts[i] += 1

# Normalisation
mu_n_conditioned = state_counts / len(final_states)

# Histogramme
plt.bar(range(7), mu_n_conditioned, tick_label=[1, 2, 3, 4, 5, 6, 7])
plt.title(f'Distribution de X_n après {n} pas (Conditionné à ne pas atteindre 7)')
plt.xlabel('Etat')
plt.ylabel('Probabilité')
plt.grid(True)
plt.show()

# On illustre maintenant la convergence de cette mesure vers la mesure presque-invariante

# Distance en variation totale entre deux probas
def dist_TV(p, q):
    return 0.5 * np.sum(np.abs(p - q))

# Nombre de simulations
N_sim = 1000

# Stockage des distances
distances = []

# Simulations pour différents pas (de 1 à 10)
for n in range(1, 11):
    final_states = []

    # Simulation un grand nombre de fois
    for _ in range(N_sim):
        X0 = 0
        final_state = sim_last_step_conditioned(P, X0, n)
        if final_state is not None:
            final_states.append(final_state)

    # Distribution empirique
    state_counts = np.zeros(6)
    for i in final_states:
        state_counts[i] += 1

    # Normalisation
    mu_n_conditioned = state_counts / len(final_states)

    # Distance en v.t. entre la proba obtenue et tilde_mu
    distance = dist_TV(mu_n_conditioned, tilde_mu)
    distances.append(distance)

# Tracé
plt.plot(range(1, 11), distances, marker='o', linestyle='-', color='b')
plt.title('Convergence vers la distribution presque-stationnaire')
plt.xlabel('Nombre de pas')
plt.ylabel('Distance en variation totale')
plt.grid(True)
plt.show()

#On va meintenant illustrer la convergence de mu_n vers tilde_mu en calculant explicitement la suite mu_n = mu_0*P^n

# Nombre de pas
n = 10

# Valeur de mu_n_conditioned pour chaque valeur de n
mu_0 = np.array([1,0,0,0,0,0])

# Stockage des mu_n et des distances en TV avec tilde_mu
mu = []
distances = []

mu.append(mu_0)

for i in range(n):
    mu.append(mu[i]@P[:6,:6]/sum(mu[i]))
    distances.append(dist_TV(mu[i], tilde_mu))

distances.append(dist_TV(mu[n], tilde_mu)) # dernière valeur

# Tracé
plt.plot(range(11), distances, marker='o', linestyle='-', color='b')
plt.title('Convergence vers la distribution presque-stationnaire')
plt.xlabel('Nombre de pas')
plt.ylabel('Distance en variation totale')
plt.grid(True)
plt.show()