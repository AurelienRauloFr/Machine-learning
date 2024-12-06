from tp_perceptron_source import *



#### Question 1 - 2 - 3

X1, y1 = rand_bi_gauss(n1=100, n2=100, mu1=[1, 1], mu2=[-1, -1], sigmas1=[0.1, 0.1],sigmas2=[0.1, 0.1])
# Renvoie les X et y (loi de gauss) (deux gaussiennnes)=> y correspond au labels
"""""plot_2d(X1, y1)
plt.show()

X2, y2 = rand_clown(n1=500, n2=500, sigma1=1, sigma2=2)
# Renvoie les X et y (loi rand clown)(renvoie un sourire de clown) => y correspond au labels
plot_2d(X2, y2)
plt.show()

X3, y3 = rand_checkers(n1=500, n2=500, sigma=0.1)
# Renvoie les X et y (loi noisy checkers) (renvoie un damier) => y correspond au labels
plot_2d(X3, y3)
plt.show()
"""""

#### Perceptron Question 1
# pour p=2 ==> droite
# 
# Bonne séparatrices :
# Pour les (X1,y1)
# f = -x
# Pour les (X2,y2)
# f = 0
# Pour les (X3,y3)
# f = 0
#
# f grand ==> 
#
# f négatif ==>
#
# f positif ==>
#
# signification de f ==> <^w, x>
#
# w0 correspond à la position de l'hyperplan
#
#

d = len(X1)
w =np.random.normal(0, 1, d) #= np.random.multivariate_normal(np.zeros(d),np.identity(1))
w = w / np.linalg.norm(w)

#### Question 2
print(predict(X1, w))
print(predict_class(X1, w))


#### Question 3
# Pourcentage d'erreur : 
# erreur quadratique : 
# erreur hinge :
