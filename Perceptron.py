import numpy as np
import matplotlib.pyplot as plt

class Perceptron:
    def __init__(self,n_input, learning_rate):
        self.w = -1 + 2 * np.random.rand(n_input)
        self.b = -1 + 2 * np.random.rand()
        self.eta = learning_rate

    def predict(self,X):
        
        p = X.shape[1]
        
        y_est = np.zeros(p)
        for i in range(p):
            y_est[i] = np.dot(self.w, X[:,i]) + self.b
            if(y_est[i]) >=0:
                y_est[i] = 1
            else:
                y_est[i] = 0
        return y_est

    def fit(self, X, Y, epochs=50):
        p = X.shape[1]
        for _ in range(epochs):
            for i in range(p):
                y_est = self.predict(X[:,i].reshape(-1,1))
                self.w += self.eta * (Y[i] - y_est) * X[:,i]
                self.b += self.eta * (Y[i]-y_est) * 1

def draw_2d(model):
    w1, w2, b = model.w[0], model.w[1], model.b
    li, ls = -2, 2
    plt.plot([li,ls],[(1/w2)*(-w1*(li)-b),(1/w2)*(-w1*(ls)-b)], '--k')

def graficar(X,Y,neuron,titulo,limites,labelsXY):
    _, p = X.shape
    for i in range(p):
        if(Y[i] == 0):
            plt.plot(X[0,i],X[1,i], 'or')
        else:
            plt.plot(X[0,i],X[1,i],'ob')

    plt.title(titulo)
    plt.grid('on')
    plt.xlim(limites[0])
    plt.ylim(limites[1])
    plt.xlabel(labelsXY[0])
    plt.ylabel(labelsXY[1])

    draw_2d(neuron)

    plt.show()

def generar_datos(top):
    X=[[],[]]
    Y=[]
    for i in range(top):
        altura = 1.2 + (2.20 - 1.2) * np.random.rand()
        peso = 40 + (130 - 40) * np.random.rand()

        imc = peso / pow(altura,2)

        if imc > 25:
            ytemp = 1 #Sobrepeso
        else:
            ytemp = 0 #Peso normal
        
        X[0].append(altura)
        X[1].append(peso)
        Y.append(ytemp)
    
    return X,Y



def AproxIMC(neuron):

    #Creación de datos aleatorios
    Xt,Y = generar_datos(100)

    #Normalizacion
    for i in range(2):
        maxX = max(Xt[i])
        minX = min(Xt[i])
        for j, z in enumerate(Xt[i]):
            Xt[i][j] = (z-minX)/(maxX-minX)
    X=np.array(Xt)
    neuron.fit(X,Y)
    limites =[[-0.20,1.20],[-0.20,1.20]]
    labelsXY =["Altura(m)","Peso(kg)"]

    graficar(X,Y,neuron,'Aproximación de la definición de sobrepeso con Perceptrón',limites,labelsXY)


if __name__ == '__main__':
    opc = 0

    neuron = Perceptron(2,0.1)

    X = np.array([[0,0,1,1],
                [0,1,0,1]])
    
    while(opc != 5):
        
        print("Perceptron")
        print("1. and")
        print("2. or")
        print("3. xor")
        print("4. Pesos")
        print("5. Salir")
        print(">",end="")

        opc=int(input())

        if(opc > 0):
            if(opc < 4):

                title ="Perceptron "
                limites =[[-2,2],[-2,2]]
                labelsXY =[r'$x_1$',r'$x_2$']

                if(plt):
                    plt.clf()
                    plt.cla()
                    
                if(opc == 1):
                    title += "(AND)"
                    Y = np.array([0,0,0,1])
                    print("Seleccionaste la compuerta AND")

                elif(opc == 2):
                    title += "(OR)"
                    Y = np.array([0,1,1,1])
                    print("Seleccionaste la compuerta OR")

                elif(opc == 3):
                    title += "(XOR)"
                    Y = np.array([0,1,1,0])
                    print("Seleccionaste la compuerta XOR")

                neuron.fit(X,Y)
                
                graficar(X,Y,neuron,title,limites,labelsXY)
            elif(opc==4):
                print("Seleccionaste la aproximación de la definición de sobrepeso")
                AproxIMC(neuron)
            elif(opc==5):
                print("Has salido del programa")
            else:
                print("Opcion no valida")
        else:
            print("Opcion no valida")
        print("\n--------------------------")
