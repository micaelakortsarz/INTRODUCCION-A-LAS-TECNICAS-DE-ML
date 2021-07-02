import numpy as np
from numpy import linalg as LA
import matplotlib.pyplot as plt


class Distribucion:
    def __init__(self, n,rango,N_datos):
        self.n=n
        self.N_datos=N_datos
        self.real_mean = (rango)*np.random.rand(1,n)
        self.real_std = (rango/7)*np.abs(np.random.rand(1,n))
        self.dist=np.random.normal(self.real_mean, self.real_std, (N_datos,n))
        np.random.shuffle(self.dist)
        self.clases=np.empty(N_datos)
    def Agrupa(self,means):
        for i in range(self.N_datos):
            self.clases[i]=np.argmin([LA.norm(self.dist[i]-means[j][0]) for j in range(len(means))])


class Data:
    def __init__(self,p,n,rango,N_datos,means_iniciales):
        self.p=p
        self.distribuciones=[Distribucion(n,rango,N_datos) for i in range(p)]
        self.init_means=means_iniciales
        self.means=np.copy(self.init_means)
        for i in range(p):
            self.distribuciones[i].Agrupa(self.means)

    def Actualizar_Medias(self):
        index=np.empty((len(self.init_means), self.p),dtype=list)
        for i in range(len(self.init_means)):
            for j in range(self.p):
                aux=(np.where(self.distribuciones[j].clases==i))[0]
                index[i][j]=aux
        for l in range(len(self.init_means)):
            x=[np.stack([self.distribuciones[i].dist[index[l][i][j]] for j in range(len(index[l][i]))],axis=0)
               for i in range(self.p) if len(index[l][i]) != 0]
            if len(x) != 0:
                if len(x)>1:
                    x=np.vstack(x)
                if type(x)==np.ndarray:
                    aux2=np.average(x,axis=0)
                elif type(x)==list and x!=[]:
                    aux2=np.average(x[0],axis=0)
                self.means[l] = aux2
            else: continue

    def k_means_1iter(self):
        self.Actualizar_Medias()
        for i in range(self.p):
            self.distribuciones[i].Agrupa(self.means)

    def k_means(self,tolerancia,dim,i):
        iter = 0
        aux=10000
        plt.figure(figsize=(9,9))

        ax1 = plt.subplot(221)
        self.Plot_Distribuciones_Reales(dim)
        ax2 = plt.subplot(222)
        self.Plot_K_Means(dim,iter)
        while aux>tolerancia:
            aux_m=np.copy(self.means)
            self.k_means_1iter()
            iter+=1
            aux=LA.norm(aux_m-self.means)
            if iter==i:
                ax3 = plt.subplot(223)
                self.Plot_K_Means(dim,iter)

        ax4 = plt.subplot(224)
        self.Plot_K_Means(dim, iter)
        plt.tight_layout()
        plt.show()
        return iter

    def Plot_K_Means(self,dim_a_graficar,iter):
        colores = ['blue', 'magenta', 'grey', 'cyan','orange','green']
        label = ['Clase {}'.format(i) for i in range(len(self.means))]
        with plt.style.context('seaborn-darkgrid'):
            plt.xticks([])
            plt.yticks([])
            plt.title('Iteración número {}'.format(iter))
            plt.xlabel('Característica 0')
            plt.ylabel('Característica {}'.format(dim_a_graficar))
            plt.grid(True)
            for i in range(self.p):
                colores_asociados=[colores[int(self.distribuciones[i].clases[j])] for j in range(self.distribuciones[i].N_datos)]
                plt.scatter(self.distribuciones[i].dist[:, 0], self.distribuciones[i].dist[:, dim_a_graficar], alpha=0.4,
                            color=colores_asociados)
            for i in range(len(self.means)):
                plt.scatter(self.means[i][0][0],self.means[i][0][dim_a_graficar],label=label[i], color=colores[i], edgecolor='k')
                plt.legend()





    def Plot_Distribuciones_Reales(self, dim_a_graficar):
        colores=['blue','magenta','grey','cyan']
        with plt.style.context('seaborn-darkgrid'):
            plt.xticks([])
            plt.yticks([])
            plt.title('Distribuciones reales')
            plt.xlabel('Característica 0')
            plt.ylabel('Característica {}'.format(dim_a_graficar))
            label=['Clase {}'.format(i) for i in range(len(self.means))]
            for i in range(self.p):
                plt.scatter(self.distribuciones[i].dist[:,0], self.distribuciones[i].dist[:,dim_a_graficar],alpha=0.4, color=colores[i])
                plt.scatter(self.distribuciones[i].real_mean[0][0], self.distribuciones[i].real_mean[0][dim_a_graficar],
                label=label[i],color=colores[i],edgecolor='k')
                plt.legend()
                plt.grid()


n=3
p=4
rango=10
dim_graficada=1
k=4
N_datos=100
means_k = [(rango) * np.random.rand(1, n) for i in range(k)]
d=Data(p,n,rango,N_datos,means_k)
print("Cantidad de iteraciones: {}".format(d.k_means(1e-5,dim_graficada,3)))