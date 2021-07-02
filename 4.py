import numpy as np
from numpy import linalg as LA
from collections import Counter
import matplotlib.pyplot as plt

class Metodo_KNN:
    def __init__(self, n,rango,N_datos,nro_clases,nro_vec):
        self.n=n
        self.N_train=N_datos
        self.n_clases = nro_clases
        self.n_vec = nro_vec
        self.dist_train=np.empty((self.n_clases,self.N_train,self.n))
        self.clases_train = np.empty((self.n_clases, self.N_train))

        for i in range(self.n_clases):
            np.random.seed(i*404)
            real_mean = (2*rango)*np.random.uniform(-1.5,1.5)*np.random.rand(1,n)
            real_std = (rango/2.5)*np.abs(np.random.rand(1,n))
            self.dist_train[i]=np.random.normal(real_mean, real_std, (self.N_train,n))
            self.clases_train[i] = i * np.ones(self.N_train)
            np.random.shuffle(self.dist_train[i])

        self.dist_train=np.vstack(self.dist_train)
        self.clases_train=np.hstack(self.clases_train)

    def KNN_1dato(self,x):
        distancias=[LA.norm((x-self.dist_train[i])) for i in range(len(self.dist_train))]
        ind = np.argpartition(distancias, self.n_vec)[:self.n_vec]
        k_clases=[self.clases_train[ind[i]] for i in range(len(ind))]
        return Counter(k_clases).most_common()[0][0]

    def KNN(self,datos_test):
        return([self.KNN_1dato(datos_test[i]) for i in range(len(datos_test))])

    def Accuracy_KNN(self,clases_reales,datos):
        ac=0
        for i in range(len(datos)):
            if (self.KNN(datos)[i]-clases_reales[i])==0:
                ac+=1
        return ac/len(datos)

    def Plot_Distribuciones(self,rango,k):
        colores = ['blue', 'magenta', 'grey', 'cyan', 'pink', 'orange', 'green']

        x = np.linspace(-2*rango, rango*2, 100)
        y = np.linspace(-2*rango, rango*2, 100)

        xx, yy=np.meshgrid(x,y)
        z=self.KNN(np.c_[xx.ravel(),yy.ravel()])
        zz=np.reshape(z,xx.shape)
        plt.contourf(xx,yy,zz, self.n_clases ,alpha=0.7)
        plt.scatter(self.dist_train[:, 0], self.dist_train[:, 1], c=self.clases_train, edgecolor='k', alpha=1)
        plt.xticks([])
        plt.yticks([])
        plt.title('K={0} vecinos'.format(k))
        plt.tight_layout()

n=2
nro_clases=5
kvecinos=[1,3,5,7]
rango=20
dim_graficada=1
N_datos=10

plt.figure()

d=Metodo_KNN(n,rango,N_datos,nro_clases,1)
ax1 = plt.subplot(221)
d.Plot_Distribuciones(rango,1)

d=Metodo_KNN(n,rango,N_datos,nro_clases,3)
ax2 = plt.subplot(222)
d.Plot_Distribuciones(rango,3)

d=Metodo_KNN(n,rango,N_datos,nro_clases,5)
ax3 = plt.subplot(223)
d.Plot_Distribuciones(rango,5)

d=Metodo_KNN(n,rango,N_datos,nro_clases,7)
ax4 = plt.subplot(224)
d.Plot_Distribuciones(rango,7)
plt.show()
#print(d.Accuracy_KNN(d.clases_train,d.dist_train))

