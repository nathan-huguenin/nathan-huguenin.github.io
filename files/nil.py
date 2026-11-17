import matplotlib.pyplot as plt
import numpy as np
import numpy.random as rd
import scipy.stats as st
from scipy.linalg import toeplitz

## Données et modélisation naïve (suite de v.a. i.i.d.)

# Chargement des données (attention au répertoire)

donnees = np.loadtxt('donnees.txt')
X = donnees[:,1]

# Représentation des données (usage de plot pour faire
# apparaître le profil du bruit blanc)

plt.figure()
plt.plot(donnees[:,0],X)
plt.title("Hauteur max du Nil selon l'année'")

Xbar = np.mean(X) # moyenne empirique
n = len(X)-1

# Corrélogramme

K=30

cov = np.zeros(K+1) # covariance empirique
for k in range(K+1):
    cov[k] = np.mean((X[k:]-Xbar)*(X[:n-k+1]-Xbar))

correl = cov/cov[0] # correlation empirique

D = 1/2 # exposant arbitraire pour illustrer la loi puissance

plt.figure()
plt.plot(range(K+1),correl)
plt.plot(range(K+1)[1:],np.power(range(K+1)[1:],-D))
plt.title("Corrélogramme")

# Test "d'indépendance" bilatéral

Sn = np.sqrt(n)*np.abs(correl[1:]) # la statistique de test
pvaleur = 2*st.norm.sf(Sn) # car test bilatéral
plt.figure()
plt.scatter(range(K),pvaleur)
plt.title("p-valeurs (2P(X>Sn) où X~N(0,1))")


## Modélisation par un bruit Gaussien fractionnaire : estimation du paramètre H

# Processus agrégé

Y = np.zeros(len(X)+1)
Y[1:] = np.cumsum(X-Xbar)
plt.figure()
plt.plot(Y)
plt.title("Processus agrégé Y sur les données réelles")

# Moyenne empirique des variations quadratiques de Y

def var(Y,a):
    m = int((len(Y)-1)/a)-1
    V = np.zeros(m)
    for t in range(m):
        V[t] = (Y[a*(t+2)]-2*Y[a*(t+1)]+Y[a*t])**2
    return np.mean(V)

# Estimation du paramètre H par régression linéaire

Vemp = [var(Y,a) for a in range(1,K+1)]
plt.figure()
plt.loglog(range(1,K+1),Vemp)
plt.title("Tracé en échelle loglog de la moyenne empirique \n des variations quadratiques de Y")

# estimation de H : pente de la droite / 2
slope = np.polyfit(np.log(range(1,K+1)),np.log(Vemp),1)[0] / 2

plt.text(1.5, max(Vemp)/2, f"Estimation de H: {slope:.4f}", fontsize=12, color='red')

## Simulation du bruit Gaussien fractionnaire

# covariance (sigma^2:=1)
def r(k,H):
    return (1/2)*(np.abs(k+1)**(2*H) + np.abs(k-1)**(2*H) - 2*np.abs(k)**(2*H))

# simulation d'un vecteur Gaussien quelconque (covariance définie positive)
def vectGauss(m,gamma,dim):
    x = rd.normal(0,1,dim)
    A = np.linalg.cholesky(gamma)
    xt = x.reshape(-1,1)
    return A@xt + m

# simulation d'une trajectoire du bruit Gaussien fractionnaire
def bruitGaussFrac(n,H):
    # création de la matrice de covariance (c'est une matrice de Toeplitz)
    gamma = toeplitz(r(np.arange(n+1),H))

    #gamma=np.zeros(shape=(n+1,n+1))
    #for i in range(n+1):
    #    for j in range(n+1):
    #        gamma[i,j]=r(i-j,H)
    return vectGauss(0,gamma,n+1)

# tracé pour plusieurs valeurs de H du bruit Gaussien et du processus agrégé
n=1000

plt.figure()
for (i,H) in zip(range(1,5),[0.5,0.65,0.8,0.95]):
    X = bruitGaussFrac(n,H)
    Y = np.cumsum(X)

    plt.subplot(2,4,i)
    plt.plot(X)

    plt.subplot(2,4,4+i)
    plt.plot(Y)

    plt.title("H="+str(H))


# Estimation de H à partir de la simulation

n=1000
H=0.88
X = bruitGaussFrac(n,H)
Y = np.cumsum(X)

# régression linéaire

Vemp = [var(Y,a) for a in range(1,K+1)]
plt.figure()
plt.loglog(range(1,K+1),Vemp)
plt.title("Tracé en échelle loglog de $S_n(a)$")

# estimation de H : pente de la droite / 2
slope = np.polyfit(np.log(range(1,K+1)),np.log(Vemp),1)[0] / 2

plt.text(1.5, max(Vemp)/2, f"Estimation de H: {slope:.4f}", fontsize=12, color='red')

## Illustration du théorème central limite sur S_n(a)

# paramètres
a = 3
n = 300 # taille des trajectoires
N = 1000 # nombre de trajectoires
H = 0.88

# moyennes empiriques des variations quadratiques des trajectoires
Vemp = np.zeros(N)
for i in range(N):
    X = bruitGaussFrac(n,H)
    Y = np.cumsum(X)
    Vemp[i] = var(Y,a)

Vemp_normalise = np.sqrt(n)*(Vemp - (4-2**(2*H))* a**(2*H))

# variance gamma(a)
l = np.arange(-100,101)
gamma = a**(4*H+1)/2*np.sum(
    (np.abs(l+2)**(2*H)
     +np.abs(l-2)**(2*H)
     -4*np.abs(l+1)**(2*H)
     -4*np.abs(l-1)**(2*H)
     +6*np.abs(l)**(2*H))**2)

# normalisation par l'écart-type pour avoir une N(0,1) à la limite
Vemp_normalise=Vemp_normalise/np.sqrt(gamma)

# tracé :
# histogramme
plt.figure()
plt.hist(Vemp_normalise,bins=30,density=True)
x = np.linspace(-3,3,100)
plt.plot(x,st.norm.pdf(x)) # densité de N(0,1)
plt.title("Illustration du TCL pour $S_n(a)$")
# fonction de répartition
plt.figure()
plt.plot(np.sort(Vemp_normalise),np.arange(0,1,1/len(Vemp)))
plt.plot(x,st.norm.cdf(x))
plt.title("Illustration du TCL pour $S_n(a)$")
plt.show()
