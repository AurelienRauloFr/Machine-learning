import numpy as np
from tools import bruit_gauss, calc_erreur
from markov_chain import *
import matplotlib.pyplot as plt

# Définissez les paramètres de la chaîne de Markov et les paramètres du bruit
n = 1000  # Taille du signal
w1, w2 = 0, 1  # Valeurs des classes
p = np.array([0.5, 0.5])  # Probabilité d'apparition a priori pour chaque classe

matrices_transition = [
    np.array([[0.8, 0.2], [0.2, 0.8]]),  # Première matrice de transition
    np.array([[0.6, 0.4], [0.4, 0.6]]),  # Deuxième matrice de transition
    np.array([[0.9, 0.1], [0.1, 0.9]])  # Troisième matrice de transition
]
parametres_bruit = [
    (0, 3, 1, 2),
    (1, 1, 1, 5),
    (0, 1, 1, 1)
]

resultats = []  # Liste pour stocker les résultats
figures = []


for A in matrices_transition:
    erreurs = []  # Liste pour stocker les taux d'erreur
    for m1, m2, sig1, sig2 in parametres_bruit:
        # Générez la chaîne de Markov
        CM = simu_mc(n, [w1, w2], p, A)

        # Bruitez la chaîne de Markov
        CM_bruit = bruit_gauss(CM, [w1, w2], m1, sig1, m2, sig2)

        # Segmentez le signal bruité en utilisant le MPM dans les chaînes de Markov cachées
        CM_bruit_MPM = mpm_mc(CM_bruit, [w1, w2], p, A, m1, sig1, m2, sig2)

        # Calculez le taux d'erreur
        erreur = calc_erreur(CM, CM_bruit_MPM)
        erreurs.append(erreur)

        plt.figure(figsize=(12, 4))
        plt.subplot(131)
        plt.plot(CM)
        plt.title("Signal Markov")
        plt.subplot(132)
        plt.plot(CM_bruit)
        plt.title("Signal Bruité")
        plt.subplot(133)
        plt.plot(CM_bruit_MPM)
        plt.title("Signal Segmenté")
        figures.append(plt)

    resultats.append(erreurs)


# Affichez les résultats dans un tableau récapitulatif
for i, erreurs in enumerate(resultats):
    print(f"Matrice de transition {i + 1}:")
    for j, erreur in enumerate(erreurs):
        print(f"Bruit {j + 1}: Taux d'erreur = {erreur:.4f}")

# Afficher les figures
for i, figure in enumerate(figures):
    figure.suptitle(f"Matrice de transition {i//3 + 1} - Bruit {i%3 + 1}")
    plt.show()