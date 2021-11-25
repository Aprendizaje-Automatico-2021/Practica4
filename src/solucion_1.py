import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
from displayData import *

def sigmoide_fun(Z):
    return 1 / (1 + (np.exp(-Z)))

def matrix_forward_propagate(X, thetas1, thetas2):
    m = X.shape[0]

    A1 = X
    A1 = np.hstack([np.ones([m, 1]), A1])
    # (5K, 401) * (401, 25) = (5K, 25)
    Z2 = np.matmul(A1, thetas1.T)
    A2 = sigmoide_fun(Z2)   
    A2 = np.hstack([np.ones([m, 1]), A2])

    # (5k, 26) * (26, 10) = (5K, 10)
    Z3 = np.matmul(A2, thetas2.T)
    A3 = sigmoide_fun(Z3)
    return A1, A2, A3
    
def J(theta1, theta2, X, y, k = 10):
    """
        Calculates the J function of the cost
        of the Logistic Regresion    
    """
    m = X.shape[0]

    total = 0
    
    # Para cada imagen
    a1, a2, h = matrix_forward_propagate(X, theta1, theta2) # (10, )
    for i in range(m):
        sum1 = -y[i] * np.log(h[i] + 0.000001)
        sum2 = (1 - y[i]) * (np.log(1 - h[i] + 0.00001))
        total += np.sum(sum1 - sum2)
    
    return (1 / m) * total

def regularization(thetas1, thetas2, m, lamb):
    """
    Calcula el termino regularizado de la función de coste
    en función de lambda
    """
    total = 0
    # t1(25, 400), t2(10, 25)
    sum1 = np.sum(np.power(thetas1, 2))
    sum2 = np.sum(np.power(thetas2, 2))
    total = sum1 + sum2
    return (lamb / (2 * m)) * total

def backprop (params_rn, num_entradas, num_ocultas, num_etiquetas, X, y , reg, num_layers = 3):
    """
    ah
    """
    Theta1 = np.reshape(params_rn[:num_ocultas * ( num_entradas + 1)], (num_ocultas, (num_entradas + 1)))
    Theta2 = np.reshape(params_rn[num_ocultas * ( num_entradas + 1):], (num_etiquetas, (num_ocultas + 1)))

    delta1 = np.zeros((Theta1.shape[0], Theta1.shape[1]))
    delta2 = np.zeros((Theta2.shape[0], Theta2.shape[1]))

    A1, A2, H = matrix_forward_propagate(X, Theta1, Theta2)
    m = X.shape[0]
    for t in range(m):
        a1t = A1[t, :]
        a2t = A2[t, :]
        ht = H[t, :]
        yt = y[t]

        d3t = ht - yt   # Este es el error comparando el coste obtenido con el que deberia obtenerse
        d2t = np.dot(Theta2.T, d3t) * np.dot(a2t, (1-a2t))
        delta1 = delta1 + np.dot(d2t[1:, np.newaxis], a1t[np.newaxis, :])
        delta2 = delta2 + np.dot(d3t[:, np.newaxis], a2t[np.newaxis, :])

    delta1 /= m
    delta2 /= m

    gradient = np.concatenate((delta1.ravel(), delta2.ravel()))

    return gradient


def main():
    data = loadmat ('./src/ex4data1.mat')
    thetas = loadmat("./src/ex4weights.mat")
    # Ejemplos de entrenamiento
    y = data ['y']  # (5K, 1)
    y = y.ravel() # (5K, 1) --> (5K,)
    X = data ['X']  # (5K, 400)
    #  Valores de Thetas
    thetas1,thetas2 = thetas["Theta1"], thetas["Theta2"] # (25, 401) - (10, 26)

    # Parte 1 - Coste sin regularizar y regularizado
    m = len(y) #5K
    num_labels = 10
    y = y - 1
    y_onehot = np.zeros((m,num_labels)) #5K * 10 - 5k arrays of len 10 filled with 0

    for i in range(m): #each array in y_onehot is getting changed
        y_onehot[i][y[i]] = 1 

    cost = J(thetas1, thetas2, X, y_onehot)
    print("Coste sin regularizar: ", cost)

    lamb = 1
    m = X.shape[0]
    t1 = np.delete(thetas1, 0, 1)
    t2 = np.delete(thetas2, 0, 1)
    cost += regularization(t1, t2, m, lamb)
    print("Coste regularizado: ", cost)

    # Parte 2 - Calculo del gradiente
    #  Reconstruir params_rn en funcion de las thetas
    params_rn= np.concatenate((thetas1.ravel(),  thetas2.ravel()))
    cost, params_rn = backprop(params_rn, 400, 25, 10, X, y, lamb)

    print("Coste segunda parte: ", cost)

    # Display Part
    sample = np.random.choice(X.shape[0], 100)
    newMat = np.zeros((100,400))

    for i in range(100):
        newMat[i] = X[sample[i]]

    aux = displayData(newMat)
    plt.show()

main()