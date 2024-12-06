import numpy as np
from tools import bruit_gauss, calc_erreur
from gaussian_mixture import *
from markov_chain import *

# Paramètres pour le modèle indépendant
w = np.array([1, 2], dtype=int)
p_indep = [0.25, 0.75]


# Paramètres pour la chaîne de Marko
p_markov = [0.25, 0.75]
A_markov = np.array([[0.8, 0.2], [0.07, 0.93]])
n = 1000  # Taille des signaux à générer

# Liste des bruits à utiliser
bruits = [(0, 3, 1, 2), 
          (1, 1, 1, 5), 
          (0, 1, 1, 1)]

for i, (m1, sig1, m2, sig2) in enumerate(bruits):
    print(f"Bruit {i + 1} (m1={m1}, m2={m2}, sig1={sig1}, sig2={sig2})")

    # Modèle indépendant
    signal_indep = simu_gm(n, w, p_indep)
    signal_noisy = bruit_gauss(signal_indep, w, m1, sig1, m2, sig2)
    p_markov_est, A_markov_est = calc_probaprio_mc(signal_indep, w)
    

    signal_restored_indep = mpm_gm(signal_noisy, w, p_indep, m1, sig1, m2, sig2)
    error_rate_indep = calc_erreur(signal_indep, signal_restored_indep)

    signal_restored_indep_MPM = mpm_mc(signal_noisy, w, p_markov_est, A_markov_est, m1, sig1, m2, sig2)
    error_rate_MPM = calc_erreur(signal_indep, signal_restored_indep_MPM)
    print(f"Modèle indépendant + modèle indé : Taux d'erreur = {error_rate_indep:.4f}")
    print(f"Modèle indépendant + chaine de markov : Taux d'erreur = {error_rate_MPM:.4f}")

    # Chaîne de Markov
    signal_markov = simu_mc(n, w, p_markov, A_markov)
    signal_noisy = bruit_gauss(signal_markov, w, m1, sig1, m2, sig2)
    p_indep_est = calc_probaprio_gm(signal_markov, w)

    signal_restored_markov = mpm_mc(signal_noisy, w, p_markov, A_markov, m1, sig1, m2, sig2)
    error_rate = calc_erreur(signal_markov, signal_restored_markov)

    signal_restored_indep = mpm_gm(signal_noisy, w, p_indep_est, m1, sig1, m2, sig2)
    error_rate_indep = calc_erreur(signal_markov, signal_restored_indep)

    print(f"Chaine de Markov + modèle indé : Taux d'erreur = {error_rate_indep:.4f}")
    print(f"Chaine de Markov + chaine de markov : Taux d'erreur = {error_rate_MPM:.4f}")

    print()

print("Fin de l'étude")