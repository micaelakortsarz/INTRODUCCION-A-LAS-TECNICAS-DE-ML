import numpy as np
import matplotlib.pyplot as plt

def Generador_de_datos(N_datos,N_dim,rango,a):
    x=np.ones((N_datos,N_dim+1))
    y=np.zeros((N_datos,N_dim+1))
    for i in range(N_datos):
        x[i][0:N_dim]=[np.random.uniform(-rango,rango) for j in range(N_dim)]
    y=x.dot(a)+(rango/5)*np.random.normal(0,rango/10,(N_datos,))
    return [x,y]

def Regresion_Lineal(x,y):
    xt=x.transpose()
    aux=np.linalg.inv(xt.dot(x))
    aux=aux.dot(xt)
    return aux.dot(y)

def Error(a_real,A):
    e=np.linalg.norm([a_real[i]-A[i] for i in range(len(A))],ord=2)
    return e

def print_resultados(A,N_dim):
    aux=np.transpose(A)
    print("b={0:.3f}".format(A[N_dim]))
    for i in range(N_dim):
        print("a{0:}={1:.3f}".format(i,A[i]))

def plot_resultados(x,y,A,n,dim,N_datos,rango):
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    labels=['Datos generados','Ajuste']
    ax.spines['left'].set_position('center')
    ax.spines['right'].set_color('none')
    ax.spines['top'].set_color('none')
    with plt.style.context('seaborn-darkgrid'):
        plt.scatter(x, y,label=labels[0])
        x_aux =[ np.linspace(-rango,rango,N_datos) for i in range(n)]
        x_aux=np.reshape(x_aux,(N_datos,n))
        x_aux=[np.append(x_aux[i],1) for i in range(N_datos)]
        y_aux=np.dot(x_aux,A)
        plt.plot(np.transpose(x_aux)[dim],y_aux,c='r',label=labels[1])
        plt.legend()
        plt.show()

N_datos_max=200
rango=10
dim=0

for N_datos in [50,100,150,200]:
    paso = 1
    i=0
    e = np.zeros((int(N_datos / paso)))
    for N_dim in range(1,N_datos,paso):
        a=np.transpose(np.linspace(1,6,N_dim+1))
        aux=Generador_de_datos(N_datos,N_dim,rango,a)
        A=Regresion_Lineal(aux[0],aux[1])
        e[i] = Error(a, A)
        i += 1
    with plt.style.context('seaborn-darkgrid'):
        plt.scatter(np.linspace(1, N_datos, (int(N_datos / paso))-1), e[:-1], label='Número de datos={0}'.format(N_datos))
        plt.grid()


plt.xlabel(r'Número de dimensiones n')
plt.ylabel(r'Error ||$A_{Calculado}-A_{Real}$||')

plt.legend()
plt.show()

#Distribucion en 1D para ploteo
a = np.transpose(np.linspace(1, 2, 2))
aux = Generador_de_datos(100, 1, rango, a)
A = Regresion_Lineal(aux[0], aux[1])
x=(np.transpose(aux[0]))[0]
plot_resultados(x,aux[1],A,1,0,100,rango)









