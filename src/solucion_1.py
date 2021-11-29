import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
from checkNNGradients import checkNNGradients
from displayData import *
import scipy.optimize as opt

def loadWeights():
	weights = loadmat('./src/ex4weights.mat')
	theta1, theta2 = weights['Theta1'], weights['Theta2']
	return theta1, theta2

def loadData():
	data = loadmat('./src/ex4data1.mat')
	y = data['y']
	X = data['X'] 
	return X, y

def y_onehot(y,numLabels):
	m = len(y)
	
	y = (y-1)
	y_onehot = np.zeros((m,numLabels))

	for i in range(m):
		y_onehot[i][y[i]] = 1

	return y_onehot

def sigmoide_fun(Z):
    return 1 / (1 + np.exp(-Z))

def gradient_regularitation(delta, m, reg, theta):
	index0 = delta[0]
	delta = delta + (reg / m) * theta
	delta[0] = index0
	return delta

def matrix_forward_propagate(X, thetas1, thetas2):
    m = X.shape[0]
    # Input
    A1 = np.hstack([np.ones([m, 1]), X])

    # Hidden
    # (5K, 401) * (401, 25) = (5K, 25)
    Z2 = np.dot(A1, thetas1.T)
    A2 = np.hstack([np.ones([m, 1]), sigmoide_fun(Z2)])

    # Output
    # (5k, 26) * (26, 10) = (5K, 10)
    Z3 = np.dot(A2, thetas2.T)
    A3 = sigmoide_fun(Z3)

    return A1, A2, A3
    
def J(theta1, theta2, X, y, k = 10):
    """
        Calculates the J function of the cost
        of the Logistic Regresion    
    """
    m = X.shape[0]
    a1, a2, h = matrix_forward_propagate(X, theta1, theta2) # (10, )
    sum1 = y * np.log(h + 1e-9)
    sum2 = (1 - y) * np.log(1 - h + 1e-9)
    total = np.sum(sum1 + sum2)

    return (-1 / m) * total

def regularization(thetas1, thetas2, m, lamb):
    """
    Calcula el termino regularizado de la función de coste
    en función de lambda
    """
    total = 0
    # t1(25, 400), t2(10, 25)
    sum1 = np.sum(np.power(thetas1[1:], 2))
    sum2 = np.sum(np.power(thetas2[1:], 2))
    total = sum1 + sum2
    return (lamb / (2 * m)) * total

def backprop (params_rn, num_entradas, num_ocultas, num_etiquetas, X, y , reg):
    """
    Back-Propagation
    """
    Theta1 = np.reshape(params_rn[:num_ocultas * ( num_entradas + 1)], (num_ocultas, (num_entradas + 1)))
    Theta2 = np.reshape(params_rn[num_ocultas * ( num_entradas + 1):], (num_etiquetas, (num_ocultas + 1)))

    delta1 = np.zeros_like(Theta1)
    delta2 = np.zeros_like(Theta2)

    A1, A2, H = matrix_forward_propagate(X, Theta1, Theta2)

    m = X.shape[0]
    cost = J(Theta1, Theta2, X, y) + regularization(Theta1, Theta2, m, reg)

    for t in range(m):
        a1t = A1[t, :]
        a2t = A2[t, :]
        ht = H[t, :]
        yt = y[t]

        d3t = ht - yt   # Este es el error comparando el valor obtenido con el que deberia obtenerse
        d2t = np.dot(Theta2.T, d3t) * (a2t * (1 - a2t)) 
        # ¡OJO!: En d2t había un fallo tremendo porque se estaba haciendo np.dot(a2t * (1 - a2t)) en lugar
        # de la multiplicación actual, lo cual provocaba cambios en el valor delta1 bastante catastróficos

        delta1 = delta1 + np.dot(d2t[1:, np.newaxis], a1t[np.newaxis, :])
        delta2 = delta2 + np.dot(d3t[:, np.newaxis], a2t[np.newaxis, :])

    delta1 /= m
    delta2 /= m

    # Delta's gradients
    delta1 = gradient_regularitation(delta1, m, reg, Theta1)
    delta2 = gradient_regularitation(delta2, m, reg, Theta2) 

    gradient = np.concatenate((delta1.ravel(), delta2.ravel()))

    return cost, gradient


def main():
    X, y = loadData()
    thetas1, thetas2 = loadWeights()
    y = y.ravel() # (5K, 1) --> (5K,)


#----------------------------------------------------------------------------------------------------------#
    # Parte 1 - Coste sin regularizar y regularizado
    print("-----PARTE-1-----")
    num_labels = 10

    yOneHot = y_onehot(y.ravel(), num_labels)

    cost = J(thetas1, thetas2, X, yOneHot)
    print("Coste sin regularizar: ", cost)

    lamb = 1
    m = X.shape[0]
    t1 = np.delete(thetas1, 0, 1)
    t2 = np.delete(thetas2, 0, 1)
    cost += regularization(t1, t2, m, lamb)
    print("Coste regularizado: ", cost)
#----------------------------------------------------------------------------------------------------------#

    # Parte 2 - Calculo del gradiente
    #  Reconstruir params_rn en funcion de las thetas
    print("-----PARTE-2-----")
    input_layer = X.shape[1]    #-> coincide con el número de atributos de cada ejemplo de entrenamiento
    hidden_layer = 25
    params_rn= np.concatenate([thetas1.ravel(),  thetas2.ravel()])

    cost, params_rn = backprop(params_rn, input_layer, hidden_layer, num_labels, X, yOneHot, lamb)
    print("Coste en backprop: ", cost)

    checking = checkNNGradients(backprop, lamb)
    print("Comprobación de checkeo...", checking.sum() < 10e-9)
#----------------------------------------------------------------------------------------------------------#

    # Parte 3 - Aprendizaje de los parámetros óptimos
    # Una vez se haya hecho que la comprobación es la correcta, pasamos a la parte del aprendizaje automático
    print("-----PARTE-3-----")

    epsilon = 0.12
    iterations = 70
    pesos = np.random.uniform(-epsilon, epsilon, params_rn.shape[0])
    result = opt.minimize(fun = backprop, x0 = pesos,
                     args = (input_layer, hidden_layer, num_labels, X, yOneHot, lamb), 
                     method='TNC', jac=True, options={'maxiter': iterations})
    
    optT1 = np.reshape(result.x[:hidden_layer * ( input_layer + 1)], (hidden_layer, (input_layer + 1)))
    optT2 = np.reshape(result.x[hidden_layer * ( input_layer + 1):], (num_labels, (hidden_layer + 1)))

    correct = 0
    h = matrix_forward_propagate(X, optT1, optT2)[2]

    # Indices maximos
    max = np.argmax(h, axis = 1)
    max = max + 1

    correct = np.sum(max == y.ravel())
    print(f"Porcentaje de acierto: {correct * 100 /np.shape(h)[0]}%")

    # Display Part
#    sample = np.random.choice(X.shape[0], 100)
#    newMat = np.zeros((100,400))
#
#    for i in range(100):
#        newMat[i] = X[sample[i]]
#
#    aux = displayData(newMat)
#    plt.show()

main()