
## MEMO PYTHON POUR L'OPTION A

## Librairies

import numpy as np
import numpy.random as rd
import matplotlib.pyplot as plt
import scipy.stats as st
from math import *

## Général

# Booléens
1 == True, 0 == False, True and False, True or False, not True, 1 < 0, 1 != 0, 1 in [0,1]

# Tableaux
np.zeros(5)                 # Rempli de 0
np.ones((2, 3))             # Rempli de 1
np.eye(4)                   # Matrice identité
a = np.diag([5, 7, 2])      # Matrice diagonale
print("Matrice diagonale\n", a, "\nShape :", a.shape)
b = np.tile(a, (3, 2))      # Répétition d'une matrice
print("\nRépétition d'un motif\n", b, "\nShape :", b.shape)
c = np.arange(1, 7, 2)      # Suite arithmétique
print("\nSuite arithmétique\n", c, "\nShape :", c.shape)
np.tile(b, 2)
np.repeat(b, 2)             # Répétition de chaque élément

A = np.concatenate((a, a))  # Concatenation (selon les lignes)
np.concatenate((a, a), axis=1)      # Selon les colonnes
print("\nConcaténation\n", A, "\nShape :", A.shape)

a[a > 0]                    # Extraction d'éléments. La condition est un tableau booléen de même taille.
a[c > 3]                    # La condition peut être un vecteur booléen de même taille que le nombre de lignes.

## Syntaxe

# Définition de fonctions
#def nom_de_la_fonction(arg1,arg2,etc...):
    #corps de la fonction
    #return resultat

# Conditions
#if condition1:
    #action1
#elif condition2:
    #action2
#else:
    #action3

# Boucle while
#while condition:
    #action

#Possibilité d'utiliser un booléen :
#loop = True
#while loop:
    #action
    #if condition:
        #loop = False

# Boucle for
#for i in range(n):
    #action

#for x in A:
    #action


## Génération de nombres aléatoires

# rd.nom_de_la_loi(parametres, size=shape) Renvoie un tableau de taille donnée par shape
exemple = rd.normal(loc=1, scale=2, size=(2,3,2)) # Tableau 2x3x2 de variables normales de moyenne = 1 et écart-type = 2.
print("Loi normale\n", exemple, "\nShape :", exemple.shape)

exemple = rd.random(10) # 10 variables uniformes sur [0,1]
print("\nLoi uniforme\n", exemple, "\nShape :", exemple.shape)

exemple = rd.choice(exemple, size=(2,3), replace=False) # échantillonnage
print("\nSous-échantillonnage\n", exemple, "\nShape :", exemple.shape)

# les noms des lois sont listés dans la doc


## Statistiques descriptives

N = 100
X = rd.normal(loc=0, scale=1, size=N)

m = np.mean(X)
print("Moyenne empirique\n", m)
S = np.var(X) # division par n dans le calcul
print("Variance empirique \n", S)
Sprime = np.var(X, ddof=1) # division par n-1
print("Estimateur sans biais de la variance \n", Sprime)
sd = np.std(X) # même comportement avec l'option ddof
print("Ecart-type empirique\n", sd)


## Fonction de répartition et quantiles

# Fonction de répartition empirique d'un échantillon X
N = 10
X = rd.normal(loc=0, scale=1, size=N)
table = np.unique(X, return_counts=True)
frequence_cum = np.cumsum(table[1])/N
plt.figure()
plt.title("Fonction de réparition empirique")
plt.step(table[0], frequence_cum, where="post")
plt.xlabel("X")
plt.ylabel("Probabilité")
plt.legend()
plt.show()
# Si la distribution est discrète ou que N est petit, on voit que la fonction de répartition empirique ne part pas de y=0.
# Pour remédier à cela, on peut utiliser le code suivant :
# plt.step(np.insert(table[0], 0, min(table[0])), np.insert(frequence_cum, 0, 0), label="Empirique", where="post")

# Fonction densité : st.nom_de_la_loi(parametres).pdf(p)
x = np.linspace(-3, 3, num=100)

plt.figure()
plt.subplot(2,3,1)
plt.title("Densité")
plt.plot(x, st.norm(loc=0, scale=1).pdf(x))
plt.xlabel("x")
plt.ylabel("Probabilité")

# Fonction de répartition : st.nom_de_la_loi(parametres).cdf(x)
x = np.linspace(-3, 3, num=100)

plt.subplot(2,3,2)
plt.title("Fonction de répartition")
plt.plot(x, st.norm.cdf(x))
plt.xlabel("x")
plt.ylabel("Probabilité")

# Fonction quantile : st.nom_de_la_loi(parametres).ppf(p)
prob = np.linspace(0, 1, num=100)

plt.subplot(2,3,3)
plt.title("Fonction quantile")
plt.plot(prob, st.norm.ppf(prob))
plt.xlabel("Probabilité")
plt.ylabel("x")

# Pour les lois discrètes, seule la méthode pdf change : elle est remplacée par pmf
n = 10; p = 1/2;
x = range(n+1)

plt.subplot(2,3,4)
plt.title("Fonction de masse")
plt.vlines(x, 0, st.binom(n, p).pmf(x))
plt.xlabel("x")
plt.ylabel("Probabilité")

plt.subplot(2,3,5)
plt.title("Fonction de répartition")
plt.step(x, st.binom(n, p).cdf(x), where="post")
plt.xlabel("x")
plt.ylabel("Probabilité")

prob = np.linspace(0, 1, num=100000)[1:] # On prend un très grand nombre d'éléments pour capturer la très petite probabilité P(X=0). On enlève le premier élément (=0) pour éviter d'avoir une valeur -1 renvoyée par ppf

plt.subplot(2,3,6)
plt.title("Fonction quantile")
plt.step(prob, st.binom(n, p).ppf(prob))
plt.xlabel("Probabilité")
plt.ylabel("x")
plt.show()


## Nuage de points

XY = rd.normal(size=(2,100))

plt.figure()
plt.title("Nuage de points des données")
plt.scatter(XY[0,:], XY[1,:])
plt.xlabel("x")
plt.ylabel("y")
plt.show()


## Convergence p.s. (LGN)

Nsimu = 5
n = 100
k = range(1,n+1)
X = rd.random((Nsimu, n))
X_bar = np.cumsum(X, axis=1)/k

plt.figure()
plt.title("Loi des grands nombres")
plt.plot(k, X_bar.T)
plt.hlines(0.5, 1, n, label="Limite", color='k', linestyle='--')
plt.xlabel("k")
plt.ylabel("X")
plt.grid() # affiche une grille en pointillés
plt.axis([0, n,   0, 1]) # définit la taille des axes [xmin, xmax,  ymin, ymax]
plt.legend() # ajoute une légende qui reprend les labels des tracés
plt.show()


## Convergence en loi (TCL)

# Par fonction de répartition
Nsimu = 1000
n = 100
X = rd.random((Nsimu, n))
std = 1/np.sqrt(12)
X_bar = np.sum(X, axis=1)/n
Z = np.sqrt(n)*(X_bar-1/2)/std # Variables centrées réduites
table = np.unique(Z, return_counts=True)
frequence_cum = np.cumsum(table[1])/Nsimu

plt.figure()
plt.title("Théorème central limite")
plt.step(table[0], frequence_cum, label="Empirique")
plt.plot(table[0], st.norm.cdf(table[0]), label="Théorique", color='k', linestyle='--')
plt.xlabel("Z")
plt.ylabel("Probabilité")
plt.legend()
plt.show()

# Par histogramme
Nsimu = 1000
n = 100
X = rd.random((Nsimu, n))
std = 1/np.sqrt(12)
X_bar = np.sum(X, axis=1)/n
Z = np.sqrt(n)*(X_bar-1/2)/std # Variables centrées réduites
x = np.linspace(min(Z), max(Z), num=1000)

plt.figure()
plt.title("Théorème central limite")
plt.hist(Z, density=True, label="Empirique")
plt.plot(x, st.norm.pdf(x), label="Théorique", color='k', linestyle='--')
plt.xlabel("Z")
plt.ylabel("Probabilité")
plt.legend()
plt.show()

# Par diagramme en batons
Nsimu = 10000
n = 10
X = rd.binomial(n, 1/2, size=Nsimu)
table = np.unique(X, return_counts=True)
valeur = table[0]
frequence = table[1]/Nsimu

plt.figure()
plt.title("Comparaison fréquences empiriques/théoriques")
plt.bar(valeur,frequence, label="Fréquence empirique") # Diagramme en batons
plt.vlines(valeur, 0*(n+1), st.binom(n, 1/2).pmf(valeur), color='black', linestyle='--', label="Probabilité théorique")
plt.legend(loc="lower right")
plt.xlabel("Position")
plt.ylabel('Fréquence')
plt.show()


## Lire/Sauvegarder des données
X = rd.normal(0.97,0.1,size = 25) # exemple de données
# Sauver et charger en .npy
np.save("data.npy",X)
data1 = np.load("data.npy")
# Pareil en .txt
np.savetxt("data.txt",X)
data2 = np.loadtxt("data.txt")