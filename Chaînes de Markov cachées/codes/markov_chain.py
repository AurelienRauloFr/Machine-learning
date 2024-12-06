import numpy as np
from tools import gauss


def forward(A, p, gauss):
    """
    Cette fonction calcule récursivement (mais ce n'est pas une fonction récursive!) les valeurs forward de la chaîne
    :param A: Matrice (2*2) de transition de la chaîne
    :param p: vecteur de taille 2 avec la probailité d'apparition a priori pour chaque classe
    :param gauss: numpy array (longeur de signal_noisy)*2 qui correspond aux valeurs des densité gaussiennes pour chaque élément du signal bruité
    :return: un vecteur de taille la longueur de la chaîne, contenant tous les forward (de 1 à n)
    """

    n = len(gauss)
    forward_values = np.zeros((n, 2))  # Créer un tableau pour stocker les forward values
    forward_values[0] = p * gauss[0]  # Initialisation du premier pas de temps

    scaling_factors = np.zeros(n)
    scaling_factors[0] = 1 / np.sum(forward_values[0])
    forward_values[0] *= scaling_factors[0]


    for t in range(1, n):
        forward_values[t] = gauss[t] * np.dot(A.T, forward_values[t - 1])  # Calcul des forward values
        scaling_factors[t] = np.sum(forward_values[t])
        forward_values[t] /= scaling_factors[t]


    return forward_values


def backward(A, gauss):
    """
    Cette fonction calcule récursivement (mais ce n'est pas une fonction récursive!) les valeurs backward de la chaîne
    :param A: Matrice (2*2) de transition de la chaîne
    :param p: vecteur de taille 2 avec la probailité d'apparition a priori pour chaque classe
    :param gauss: numpy array (longeur de signal_noisy)*2 qui correspond aux valeurs des densité gaussiennes pour chaque élément du signal bruité
    :return: un vecteur de taille la longueur de la chaîne, contenant tous les backward (de 1 à n).
    Attention, si on calcule les backward en partant de la fin de la chaine, je conseille quand même d'ordonner le vecteur backward du début à la fin
    """

    n = len(gauss)
    backward_values = np.zeros((n, 2))  # Créer un tableau pour stocker les backward values
    backward_values[-1] = 1.0  # Initialisation du dernier pas de temps

    scaling_factors = np.zeros(n)  # Définition des facteurs d'échelle

    scaling_factors[-1] = np.sum(backward_values[-1])
    backward_values[-1] /= scaling_factors[-1]  # Normalisation du dernier pas de temps


    for t in range(n - 2, -1, -1):
        backward_values[t] = np.dot(A, gauss[t + 1] * backward_values[t + 1])  # Calcul des backward values
        scaling_factors[t] = np.sum(backward_values[t])
        backward_values[t] /= scaling_factors[t]  # Normalisation à l'aide des facteurs d'échelle

    
    return backward_values


def mpm_mc(signal_noisy, cl, p, A, m1, sig1, m2, sig2):
    """
     Cette fonction permet d'appliquer la méthode mpm pour retrouver notre signal d'origine à partir de sa version bruité et des paramètres du model.
    :param signal_noisy: Signal bruité (numpy array 1D de float)
    :param cl: vecteur dont la première composante est la valeur de la classe w1 et la deuxième est la valeur de la classe w2
    :param p: vecteur de taille 2 avec la probailité d'apparition a priori pour chaque classe
    :param A: Matrice (2*2) de transition de la chaîne
    :param m1: La moyenne de la première gaussienne
    :param sig1: La variance de la première gaussienne
    :param m2: La moyenne de la deuxième gaussienne
    :param sig2: La variance de la deuxième gaussienne
    :return: Un signal discret à 2 classe (numpy array 1D d'int)
    """
    gausses = gauss(signal_noisy, m1, sig1, m2, sig2)
    alpha = forward(A,p,gausses)
    beta = backward(A,gausses)

    posterior_probabilities = alpha * beta
    posterior_probabilities /= np.sum(posterior_probabilities, axis=0)
    classified_signal = np.where(posterior_probabilities.T[0] > posterior_probabilities.T[1], cl[0], cl[1])
    

    return classified_signal


def calc_probaprio_mc(signal, cl):
    """
    Cete fonction permet de calculer les probabilité a priori des classes w1 et w2 et les transitions a priori d'une classe à l'autre,
    en observant notre signal non bruité
    :param signal: Signal discret non bruité à deux classes (numpy array 1D d'int)
    :param cl: vecteur dont la première composante est la valeur de la classe w1 et la deuxième est la valeur de la classe w2
    :return: un vecteur de taille 2 avec la probailité d'apparition a priori pour chaque classe
    """

    n = len(signal)
    A = np.zeros((2, 2))

    # Calculer les probabilités a priori des classes
    p1 = (np.sum((signal == cl[0]))/signal.shape[0])
    p2 = (np.sum((signal == cl[1]))/signal.shape[0])
    p = np.array([p1,p2])

    # Compter les transitions de chaque classe à l'autre
    for i in range(n - 1):
        if signal[i] == cl[0] and signal[i + 1] == cl[0]:
            A[0, 0] += 1
        elif signal[i] == cl[0] and signal[i + 1] == cl[1]:
            A[0, 1] += 1
        elif signal[i] == cl[1] and signal[i + 1] == cl[0]:
            A[1, 0] += 1
        elif signal[i] == cl[1] and signal[i + 1] == cl[1]:
            A[1, 1] += 1

    # Normalisation des probabilités de transition
    A /= np.sum(A, axis=1)[:, np.newaxis]

    return p, A


def simu_mc(n, w, p, A):
    """
    Cette fonction permet de simuler un signal discret à 2 classe de taille n à partir des probabilité d'apparition des deux classes et de la Matrice de transition
    :param n: taille du signal
    :param w: vecteur dont la première composante est la valeur de la classe w1 et la deuxième est la valeur de la classe w2
    :param p: vecteur de taille 2 avec la probailité d'apparition pour chaque classe
    :param A: Matrice (2*2) de transition de la chaîne
    :return: Un signal discret à 2 classe (numpy array 1D d'int)
    """
    simu = np.zeros((n,), dtype=int)
    aux = np.random.multinomial(1, p)
    simu[0] = w[np.argmax(aux)]
    for i in range(1, n):
        aux = np.random.multinomial(1, A[np.where(w == simu[i - 1])[0][0], :])
        simu[i] = w[np.argmax(aux)]
    return simu


def calc_param_EM_mc(signal_noisy, p, A, m1, sig1, m2, sig2):
    """
    Cette fonction permet de calculer les nouveaux paramètres estimé pour une itération de EM
    :param signal_noisy: Signal bruité (numpy array 1D de float)
    :param p: vecteur de taille 2 avec la probailité d'apparition a priori pour chaque classe
    :param A: Matrice (2*2) de transition de la chaîne
    :param m1: La moyenne de la première gaussienne
    :param sig1: La variance de la première gaussienne
    :param m2: La moyenne de la deuxième gaussienne
    :param sig2: La variance de la deuxième gaussienne
    :return: tous les paramètres réestimés donc p, A, m1, sig1, m2, sig2
    """
    gausses = gauss(signal_noisy, m1, sig1, m2, sig2)
    alpha = forward(A, p, gausses)
    beta = backward(A, gausses)
    n = len(signal_noisy)

    # Calcul de psi:
    psi = np.zeros((n-1, 2, 2))
    for k in range(n-1):
        for i in range(2):
            for j in range(2):
                psi[k,i,j] = alpha[k,i] * A[i,j] *  gausses[k+1,j] * beta[k+1,j]
    psi /= np.sum(psi, axis=(1, 2), keepdims=True)
    
    # Calcul de chsi:
    chsi = np.zeros((n, 2))
    for k in range(n):
        chsi[k, 0] = alpha[k, 0] * beta[k, 0] / (alpha[k, 0] * beta[k, 0] + alpha[k, 1] * beta[k, 1])
        chsi[k, 1] = alpha[k, 1] * beta[k, 1] / (alpha[k, 0] * beta[k, 0] + alpha[k, 1] * beta[k, 1])

    #Calcul de p:
    p = np.mean(chsi, axis = 0)

    # Calcul de A:
    A = np.zeros((2,2))
    for i in range(2):
        for j in range(2):
            A[i,j] = np.sum(psi[:,i,j])/np.sum(chsi[:,i])

    # Calcul de m1 et m2::
    m1 = np.sum(signal_noisy * chsi[:,0])/np.sum(chsi[:,0])
    m2 = np.sum(signal_noisy * chsi[:,1])/np.sum(chsi[:,1])

    #Calcul de sig1 et sig2:
    sig1 = np.sqrt(np.sum(((signal_noisy-m1)**2)*chsi[:,0])/sum(chsi[:,0]))
    sig2 = np.sqrt(np.sum(((signal_noisy-m2)**2)*chsi[:,1])/sum(chsi[:,1]))
    
    return p, A, m1, sig1, m2, sig2

def estim_param_EM_mc(iter, signal_noisy, p, A, m1, sig1, m2, sig2):
    """
    Cette fonction est l'implémentation de l'algorithme EM pour le modèle en question
    :param iter: Nombre d'itération choisie
    :param signal_noisy: Signal bruité (numpy array 1D de float)
    :param p: la valeur d'initialisation du vecteur de proba
    :param A: la valeur d'initialisation de la matrice de transition de la chaîne
    :param m1: la valeur d'initialisation de la moyenne de la première gaussienne
    :param sig1: la valeur d'initialisation de la variance de la première gaussienne
    :param m2: la valeur d'initialisation de la moyenne de la deuxième gaussienne
    :param sig2: la valeur d'initialisation de la variance de la deuxième gaussienne
    :return: Tous les paramètres réestimés à la fin de l'algorithme EM donc p, A, m1, sig1, m2, sig2
    """
    #print(f"p, A, m1, sig1, m2, sig2 {p, A, m1, sig1, m2, sig2}")
    p_est = p
    A_est = A
    m1_est = m1
    sig1_est = sig1
    m2_est = m2
    sig2_est = sig2

    for i in range(iter):
        p_est, A_est, m1_est, sig1_est, m2_est, sig2_est = calc_param_EM_mc(signal_noisy, p_est, A_est, m1_est, sig1_est, m2_est, sig2_est)
        #print({'p':p_est, 'A':A_est, 'm1':m1_est, 'sig1':sig1_est, 'm2':m2_est, 'sig2':sig2_est})
    return p_est, A_est, m1_est, sig1_est, m2_est, sig2_est

