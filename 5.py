import numpy as np
from numpy import linalg as LA
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.datasets import mnist
import matplotlib.pyplot as plt

np.random.seed(40467379)

class LinearClassifier(object):
    def __init__(self, n_clases,delta,lambd,batch_size,alpha,X_train,Y_train,X_test,Y_test):
        self.n_clases = n_clases
        self.delta=delta
        self.lambd=lambd
        self.batch=batch_size
        self.alpha=alpha
        self.im_shape = X_train.shape[1:]
        b=np.ones((len(X_train),1))
        bt = np.ones((len(X_test), 1))
        self.W=np.random.randn(self.n_clases, np.prod(self.im_shape)+1)*1e-3 #Agrego el bias
        self.X_train = (2*np.hstack((np.int16(np.reshape(X_train, (X_train.shape[0], np.prod(self.im_shape)))),b))-np.max(X_train[0]))/np.max(X_train[0])
        self.Y_train =np.reshape(Y_train,(len(Y_train)))
        self.X_test=(2*np.hstack((np.int16(np.reshape(X_test, (X_test.shape[0], np.prod(self.im_shape)))),bt))-np.max(X_train[0]))/np.max(X_train[0])
        self.Y_test =np.reshape(Y_test,(len(Y_test)))
        self.X_batch=np.empty((self.batch, np.prod(self.im_shape)+1))
        self.Y_batch = np.empty((self.batch, self.n_clases))




    def regularizacion(self):
        return 0.5*np.sum(self.W*self.W)*self.lambd

    def fit(self,tolerancia):
            aux = np.inf
            epocas = 0
            epocas_totales=50
            loss=np.empty(epocas_totales)
            ac=np.empty(epocas_totales)
            ac_train = np.empty(epocas_totales)
            while (epocas<epocas_totales): #and aux>tolerancia):
                old_W=self.W
                ac[epocas]=self.accuracy()
                ac_train[epocas] = self.accuracy_train()
                loss[epocas]=self.loss_gradient()
                aux=LA.norm((old_W-self.W))
                epocas += 1
            print('Transcurrieron {} epocas'.format(epocas))
            return loss,ac,ac_train,epocas

    def loss_gradient(self):
        id_batch=np.arange(len(self.X_train))
        np.random.shuffle(id_batch)
        s=0
        for i in range(0,len(self.X_train),self.batch):
            self.X_batch= self.X_train[id_batch[i : (i+self.batch)]]
            self.Y_batch = self.Y_train[id_batch[i : (i+self.batch)]]
            dW,loss=self.gradient()
            s+=loss
            self.W-=dW*self.alpha
        return s/int(len(self.X_train)/self.batch)

    def predict(self):
       aux=[self.W.dot(self.X_test[i]) for i in range(len(self.X_test))]
       Y_p=[np.argmax(aux[i]) for i in range(len(aux))]
       return Y_p

    def predict_train(self):
       aux=[self.W.dot(self.X_train[i]) for i in range(len(self.X_train))]
       Y_p=[np.argmax(aux[i]) for i in range(len(aux))]
       return Y_p

    def accuracy(self):
            aux = 0
            Yp=self.predict()
            for i in range(len(Yp)):
                if Yp[i] == self.Y_test[i]: aux += 1
            return aux / len(Yp)

    def accuracy_train(self):
            aux = 0
            Yp=self.predict_train()
            for i in range(len(Yp)):
                if Yp[i] == self.Y_train[i]: aux += 1
            return aux / len(Yp)

class SVM(LinearClassifier):
    def __init__(self, n_clases,delta,lambd,batch_size,dw,X_train,Y_train,X_test,Y_test):
        super().__init__(n_clases,delta,lambd,batch_size,dw,X_train,Y_train,X_test,Y_test)

    def gradient(self):
        dW=np.zeros_like(self.W)
        W=np.copy(self.W).T
        scores=self.X_batch.dot(W)

        scores_y = scores[np.arange(self.X_batch.shape[0]), self.Y_batch]
        margins = scores - scores_y[:, np.newaxis] + self.delta
        margins = np.maximum(0, margins)

        margins[np.arange(self.X_batch.shape[0]), self.Y_batch] = 0

        loss = np.mean(np.sum(margins, axis=1)) + self.regularizacion()

        binary = margins.copy()

        binary[binary>0]=1
        row_s = np.sum(binary, axis=1)
        binary[np.arange(self.X_batch.shape[0]), self.Y_batch] = -row_s
        dW = np.dot(np.transpose(binary), self.X_batch)
        dW /= self.X_batch.shape[0]
        dW += self.lambd * self.W
        return dW, loss

class SM(LinearClassifier):
    def __init__(self, n_clases,delta,lambd,batch_size,dw,X_train,Y_train,X_test,Y_test):
        super().__init__(n_clases,delta,lambd,batch_size,dw,X_train,Y_train,X_test,Y_test)


    def gradient(self):
        W=self.W.T
        score=self.X_batch.dot(W)
        score-=np.max(score,axis=1)[:,np.newaxis]
        score_real=score[np.arange(self.X_batch.shape[0]),self.Y_batch]
        e_score=np.exp(score)
        e_score_sum=np.sum(e_score,axis=1)
        s=np.log(e_score_sum)-score_real
        loss=np.mean(s)+self.regularizacion()

        g=np.zeros(score.shape)
        g[np.arange(self.X_batch.shape[0]),self.Y_batch]=-1
        grad=g+e_score/e_score_sum[:,np.newaxis]
        dW=grad[:,np.newaxis,:]*self.X_batch[:,:,np.newaxis]
        dW=np.mean(dW,axis=0).T

        dW+=self.lambd*self.W

        return dW,loss

#La función de ploteo tira un warning por graficar varias veces sobre la misma figura. En otras versiones
#de matplotlib puede que no funcione correctamente
def Plot_Resultados(n_clases,delta,lambd,batch_size,alpha,X_train,Y_train,X_test,Y_test, n_test, metodo,label):
    with plt.style.context('seaborn-darkgrid'):
        plt.grid(True)
        if metodo=='SVM':
            model = SVM(n_clases ,delta,lambd,batch_size,alpha,X_train,Y_train, X_test[: n_test],Y_test[: n_test])
        if metodo=='SM':
            model = SM(n_clases, delta, lambd, batch_size, alpha, X_train, Y_train, X_test[: n_test], Y_test[ : n_test])

        s, ac, ac_train, epocas_transcurridas = model.fit(1e-200)
        ax1 = plt.subplot(311)
        plt.plot(np.linspace(0, epocas_transcurridas, epocas_transcurridas), s[: epocas_transcurridas], 'o-',
                 label=label)
        plt.ylabel(r'Loss')
        plt.xlabel(r'Epocas transcurridas')
        plt.legend()
        ax2 = plt.subplot(312)
        plt.plot(np.linspace(0, epocas_transcurridas, epocas_transcurridas), ac_train[: epocas_transcurridas], 'o-',
                 label=label)
        plt.ylabel(r'Accuracy train')
        plt.xlabel(r'Epocas transcurridas')
        plt.legend()
        plt.tight_layout()

        ax3 = plt.subplot(313)
        plt.plot(np.linspace(0, epocas_transcurridas, epocas_transcurridas), ac[: epocas_transcurridas], 'o-',
                 label=label)
        plt.ylabel(r'Accuracy test')
        plt.xlabel(r'Epocas transcurridas')
        plt.legend()
        plt.tight_layout()

#Copio datos de cifar10/mnist
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
#(x_train, y_train), (x_test, y_test) =mnist.load_data()
n_clases=10
n_test=100
#comparacion
plt.figure()
Plot_Resultados(n_clases, 1, 1e-5, 32, 1e-3, x_train, y_train, x_test, y_test, len(x_test), 'SVM','SVM')
Plot_Resultados(n_clases, 0, 1e-3, 32, 1e-2, x_train, y_train, x_test, y_test, len(x_test), 'SM','SM')
plt.show()







''''
# Exploración de hiperparámetros para SM
# Learning rate
plt.figure()
for alpha in [1e-1, 1e-2, 1e-3, 1e-4]:
    Plot_Resultados(n_clases, 1, 1e-4, 32, alpha, x_train, y_train, x_test, y_test, n_test, 'SM',
                        r'$\alpha={}$'.format(alpha))
plt.show()


# Tamaño de batch
plt.figure()
for bs in [4, 32, 128, 256]:
    Plot_Resultados(n_clases, 1, 1e-4, bs, 1e-2, x_train, y_train, x_test, y_test, n_test, 'SM',
                        r'Batch size={}'.format(bs))
plt.show()

# Regularización
plt.figure()
for lambd in [1e-1, 1e-2, 1e-3,1e-5]:
    Plot_Resultados(n_clases, 1, lambd, 32, 1e-2, x_train, y_train, x_test, y_test, n_test, 'SM',
                        r'$\lambda={}$'.format(lambd))
plt.show()


# Exploración de hiperparámetros para SVM
# Learning rate
plt.figure()
for alpha in [1e-1, 1e-2, 1e-3, 1e-4]:
    Plot_Resultados(n_clases, 1, 1e-4, 32, alpha, x_train, y_train, x_test, y_test, n_test, 'SVM',
                        r'$\alpha={}$'.format(alpha))
plt.show()


# Tamaño de batch
plt.figure()
for bs in [4, 32, 128, 256]:
    Plot_Resultados(n_clases, 1, 1e-4, bs, 1e-3, x_train, y_train, x_test, y_test, n_test, 'SVM',
                        r'Batch size={}'.format(bs))
plt.show()


# Regularización
plt.figure()
for lambd in [1e-1, 1e-2, 1e-3,1e-5]:
    Plot_Resultados(n_clases, 1, lambd, 32, 1e-3, x_train, y_train, x_test, y_test, n_test, 'SVM',
                        r'$\lambda={}$'.format(lambd))
plt.show()


plt.figure()
for d in [1e1, 1e-0, 1e-1,0]:
    Plot_Resultados(n_clases, d, 1e-2, 32, 1e-3, x_train, y_train, x_test, y_test, n_test, 'SVM',
                        r'$\Delta={}$'.format(d))
plt.show()




'''