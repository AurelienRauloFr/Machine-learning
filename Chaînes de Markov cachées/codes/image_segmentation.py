
import numpy as np
import cv2 as cv
from sklearn.cluster import KMeans
from scipy.stats import norm
import matplotlib.pyplot as plt
from math import log2, sqrt
from tools import *
from gaussian_mixture import *
from markov_chain import *


bruits = [[0, 3, 1, 2], 
          [1, 1, 1, 5], 
          [0, 1, 1, 1]]


img_1 = 'images/alpha2.bmp', 'images/beee2.bmp', 'images/cible2.bmp'
img_2 = 'images/country2.bmp', 'images/promenade2.bmp', 'images/veau2.bmp'
img_3 = 'images/zebre2.bmp', 'images/cit2.bmp'

images = ['images/cible2.bmp', 'images/country2.bmp', 'images/zebre2.bmp' ]

for img_file in images:
    img = cv.imread(img_file, cv.IMREAD_GRAYSCALE)
    img_flat = peano_transform_img(img)
    cl = np.array(sorted(list(set(img_flat))))


    for i, (m1, sig1, m2, sig2) in enumerate(bruits):
        img_flat_noisy = bruit_gauss(img_flat, cl, m1, sig1, m2, sig2)

        kmeans = KMeans(n_clusters=2, random_state=0, n_init=100).fit(img_flat_noisy.reshape(-1, 1))
        labels = kmeans.labels_
        kmeans_classes = np.array(sorted(list(set(labels))))
        p_init, A_init = calc_probaprio_mc(labels, kmeans_classes)

        m1_init = img_flat_noisy[np.where(labels == kmeans_classes[0])].mean()
        sig1_init = img_flat_noisy[np.where(labels == kmeans_classes[0])].std()
        m2_init = img_flat_noisy[np.where(labels == kmeans_classes[1])].mean()
        sig2_init = img_flat_noisy[np.where(labels == kmeans_classes[1])].std()

        iter_em = 20


        ### Classification par la méthode du MPM chaîne de Markov : ###
        p_est_mc, A_est_mc, m1_est_mc, sig1_est_mc, m2_est_mc, sig2_est_mc = estim_param_EM_mc(iter_em, img_flat_noisy, p_init, A_init, m1_init, sig1_init, m2_init, sig2_init)
        signal_restored_markov = mpm_mc(img_flat_noisy, cl, p_est_mc, A_est_mc, m1_est_mc, sig1_est_mc, m2_est_mc, sig2_est_mc)
        erreur_markov = calc_erreur(labels, signal_restored_markov)
        if erreur_markov > 0.5: #Si l'erreur est trop élevé on inverse les couleurs
                    signal_restored_markov = np.where(signal_restored_markov == cl[0], cl[1], np.where(signal_restored_markov == cl[1], cl[0], signal_restored_markov))
                    erreur_markov = 1-erreur_markov

        seg_markov_img = transform_peano_in_img(signal_restored_markov, img.shape[0])



        ### Classification par la méthode du MPM gaussian mixture : ###
        p_est_ind, m1_est_ind, sig1_est_ind, m2_est_ind, sig2_est_ind = estim_param_EM_gm(iter_em, img_flat_noisy, p_init, m1_init, sig1_init, m2_init, sig2_init)
        signal_restored_ind = mpm_gm(img_flat_noisy, cl, p_est_ind, m1_est_ind, sig1_est_ind, m2_est_ind, sig2_est_ind)
        erreur_ind = calc_erreur(img_flat, signal_restored_ind)
        if erreur_ind > 0.5: #Si l'erreur est trop élevé on inverse les couleurs
                    signal_restored_ind = np.where(signal_restored_ind == cl[0], cl[1], np.where(signal_restored_ind == cl[1], cl[0], signal_restored_ind))
                    erreur_ind = 1-erreur_ind

        seg_ind_img = transform_peano_in_img(signal_restored_ind, img.shape[0])



        fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize=(10, 5))
        ax1.imshow(img, cmap='gray')
        ax1.set_title('Image d\'origine')
        ax1.axis('off')
        ax2.imshow(transform_peano_in_img(img_flat_noisy, int(sqrt(len(img_flat)))), cmap='gray')
        ax2.set_title(f'Noise {i+1}')
        ax2.axis('off')
        ax3.imshow(seg_markov_img, cmap='gray')
        ax3.set_title(f'Segmentée par mc\nErreur = {erreur_markov:.2%}')
        ax3.axis('off')
        ax4.imshow(seg_ind_img, cmap='gray')
        ax4.set_title(f'Segmentée par gm\nErreur = {erreur_ind:.2%}')
        ax4.axis('off')
        plt.show()



"""
# Afficher l'image originale, le signal bruité, et les résultats de segmentation
        cv.imshow(f"Image Originale - Bruit {i+1}", img)
        cv.imshow(f"Signal Bruité - Bruit {i+1}", transform_peano_in_img(img_flat_noisy,int(sqrt(len(img_flat)))))
        cv.imshow(f"Segmentation Markov - Bruit {i+1}", seg_markov_img)
        cv.imshow(f"Segmentation Indépendant - Bruit {i+1}", seg_ind_img)
        cv.waitKey(0)
        cv.destroyAllWindows()



Noise_1 = [0, 3, 1, 2]
Noise_2 = [1, 1, 1, 5]
Noise_3 = [0, 1, 1, 1]
Noise_list = [Noise_1, Noise_2, Noise_3]

for image_name in ["alfa2", "country2", "zebre2"]:
    # Chargement de l'image, aplatissement avec parcours de Peano et bruitage
    image = cv.imread(f'images/{image_name}.bmp', cv.IMREAD_GRAYSCALE)
    image_peano = peano_transform_img(image)
    image_classes = np.array(sorted(list(set(image_peano))))

    for i in range(3):
        image_peano_noisy = bruit_gauss(image_peano, image_classes, Noise_list[i][0], Noise_list[i][2], Noise_list[i][1], Noise_list[i][3])

        ### Initialisation des paramètres avec classification initiale par méthode du KMeans
        kmeans = KMeans(n_clusters=2, random_state=0, n_init=100).fit(image_peano_noisy.reshape(-1,1)) # Classification
        kmeans_classes = np.array(sorted(list(set(kmeans.labels_))))
        p_i, A_i = calc_probaprio_mc(kmeans.labels_, kmeans_classes) # Probabilités et matrice de passage de la classification KMeans
        m1_i = image_peano_noisy[np.where(kmeans.labels_ == kmeans_classes[0])].mean()
        m2_i = image_peano_noisy[np.where(kmeans.labels_ == kmeans_classes[1])].mean() 
        sig1_i = image_peano_noisy[np.where(kmeans.labels_ == kmeans_classes[0])].std()               
        sig2_i = image_peano_noisy[np.where(kmeans.labels_ == kmeans_classes[1])].std()

        ### Classification par la méthode du MPM chaîne de Markov : ###

        # Calcul des paramètres d'une chaîne de Markov par l'algorithme EM
        N=20
        p_mc_est, A_mc_est, m1_mc_est, sig1_mc_est, m2_mc_est, sig2_mc_est = estim_param_EM_mc(N, image_peano_noisy, p_i, A_i, m1_i, sig1_i, m2_i, sig2_i)
        
        # Restauration par MPM suivant la chaîne de Markov et taux d'erreur
        signal_classif_mc = mpm_mc(image_peano_noisy, image_classes, p_mc_est, A_mc_est, m1_mc_est, sig1_mc_est, m2_mc_est, sig2_mc_est)
        erreur_mc = calc_erreur(image_peano, signal_classif_mc)
        
        # L'algorithme ne peut pas savoir si cl1 signifie "blanc" ou s'il signifie "noir". 
        # Il a donc 1 chance sur deux de trouver la bonne couleur.
        # On l'aide donc en "trichant" un peu (car nous connaissons l'image d'origine) en regardant l'erreur obtenue
        if erreur_mc > 0.5:
            signal_classif_mc = np.where(signal_classif_mc == image_classes[0], image_classes[1], np.where(signal_classif_mc == image_classes[1], image_classes[0], signal_classif_mc))
            erreur_mc = 1-erreur_mc
        
        # Récupérer image
        image_classif_mc = transform_peano_in_img(signal_classif_mc, int(sqrt(len(image_peano))))

        ### Classification par la méthode du MPM gaussian mixture : ###
        
        # Calcul des paramètres du modèle indépendant par l'algorithme EM
        N=20
        p_gm_est, m1_gm_est, sig1_gm_est, m2_gm_est, sig2_gm_est = estim_param_EM_gm(N, image_peano_noisy, p_i, m1_i, sig1_i, m2_i, sig2_i)
        
        # Restauration par MPM suivant la chaîne de Markov et taux d'erreur
        signal_classif_gm = mpm_gm(image_peano_noisy, image_classes, p_gm_est, m1_gm_est, sig1_gm_est, m2_gm_est, sig2_gm_est)
        erreur_gm = calc_erreur(image_peano, signal_classif_gm)
        
        # L'algorithme ne peut pas savoir si cl1 signifie "blanc" ou s'il signifie "noir". 
        # Il a donc 1 chance sur deux de trouver la bonne couleur.
        # On l'aide donc en "trichant" un peu (car nous connaissons l'image d'origine) en regardant l'erreur obtenue
        if erreur_gm > 0.5:
            signal_classif_gm = np.where(signal_classif_gm == image_classes[0], image_classes[1], np.where(signal_classif_gm == image_classes[1], image_classes[0], signal_classif_gm))
            erreur_gm = 1-erreur_gm

        # Récupérer image
        image_classif_gm = transform_peano_in_img(signal_classif_gm, int(sqrt(len(image_peano))))

        ### Montrer les 4 images : ###
        fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize=(10, 5))
        ax1.imshow(image, cmap='gray')
        ax1.set_title('Image d\'origine')
        ax1.axis('off')
        ax2.imshow(transform_peano_in_img(image_peano_noisy, int(sqrt(len(image_peano)))), cmap='gray')
        ax2.set_title(f'Noise {i+1}')
        ax2.axis('off')
        ax3.imshow(image_classif_mc, cmap='gray')
        ax3.set_title(f'Segmentée par mc\nErreur = {erreur_mc:.2%}')
        ax3.axis('off')
        ax4.imshow(image_classif_gm, cmap='gray')
        ax4.set_title(f'Segmentée par gm\nErreur = {erreur_gm:.2%}')
        ax4.axis('off')
        plt.show()

"""