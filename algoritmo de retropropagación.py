import numpy as np
import matplotlib.pyplot as plt
import csv
from MLP import *

def MLP_binary_classification_2d(X,Y,net):
    plt.figure()
    for i in range(X.shape[1]):
        if Y[0,i]==0:
            plt.plot(X[0,i], X[1,i], 'ro', markersize=9)
        else:
            plt.plot(X[0,i], X[1,i], 'bo',markersize=9)
    xmin, ymin=np.min(X[0,:])-0.5, np.min(X[1,:])-0.5
    xmax, ymax=np.max(X[0,:])+0.5, np.max(X[1,:])+0.5
    xx, yy = np.meshgrid(np.linspace(xmin,xmax, 100), np.linspace(ymin,ymax, 100))
    data = [xx.ravel(), yy.ravel()]
    zz = net.predict(data)
    zz = zz.reshape(xx.shape)
    plt.contour(xx,yy,zz,[0.5], colors='k',  linestyles='--', linewidths=2)
    plt.contourf(xx,yy,zz, alpha=0.8, cmap=plt.cm.RdBu)
    plt.xlim([xmin,xmax])
    plt.ylim([ymin,ymax])
    plt.grid()
    plt.show()

def Data_File(archivo):
    print(archivo)

    f= open(archivo)
    reader = csv.reader(f)
    result = list(reader)

    del result[0] #eliminar cabecera
    return result[:]

if  __name__ == '__main__':

    '''X = np.array([[0, 0, 1, 1],[0, 1, 0, 1]])
    Y = np.array([[0, 1, 1, 0]]) 

    net = MLP((2,20,1))
    print(net.predict(X))
    MLP_binary_classification_2d(X,Y,net)

    net.train(X, Y)
    print(net.predict(X))
    MLP_binary_classification_2d(X,Y,net)
'''
    archivo = ""
    
    opc=0
    while(opc != 5):
        
        print("Neurona Lineal")
        print("1. Concentlite")
        print("2. CIRCLES")
        print("3. MOONS")
        print("4. XOR")
        print("5. Salir")
        print(">",end="")

        opc=int(input())

        if(opc > 0):
            if(opc < 6):
                
                if(opc == 1):
                    archivo = "A6/concentlite.csv"
                elif(opc == 2):
                    archivo = "A6/circles.csv"
                elif(opc == 3):
                    archivo = "A6/Moons.csv"
                elif(opc == 4):
                    archivo = "A6/XOR.csv"
                else:
                    print('has salido del programa')
                    break
                
                if(archivo != ""):
                    

                    X=[[],[]]
                    Y=[[]]

                    data = Data_File(archivo)
                    for i in range(len(data)):
                        for j in range(len(data[i])):
                            if( j < 2):
                                X[j].append(data[i][j])
                            else:
                                Y[j-2].append(data[i][j])
                    
                    samples = len(data) #filas
                    classes = len(Y) #columnas

                    Xnp = np.array(X,dtype=float)
                    Ynp = np.array(Y,dtype=float)

                    net = MLP((2,100,1))
                    print("Predicción:")
                    net.predict(Xnp)
                    #print(net.predict(Xnp))
                    MLP_binary_classification_2d(Xnp,Ynp,net)

                    net.train(Xnp, Ynp)
                    print("\nDespués de entrenar:")
                    net.predict(Xnp)
                    #print(net.predict(Xnp))
                    MLP_binary_classification_2d(Xnp,Ynp,net)

            else:
                print('Opcion no valida')
        else:
                print('Opcion no valida')



