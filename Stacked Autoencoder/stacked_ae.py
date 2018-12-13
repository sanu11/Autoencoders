import numpy as np
import matplotlib.pyplot as plt
import os
from load_mnist import mnist
from autoencoder import stacked_autoencoder
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier


def tanh(Z):
    A = np.tanh(Z)
    cache = {}
    cache["Z"] = Z
    return A, cache

def tanh_der(dA, cache):
    A, temp=tanh(cache["Z"])
    dZ = np.multiply(dA , (1 - np.square(A)))
    return dZ

def relu(Z):
    A = np.maximum(0, Z)
    cache = {}
    cache["Z"] = Z
    return A, cache

def relu_der(dA, cache):
    dZ = np.array(dA, copy=True)
    Z = cache["Z"]
    dZ[Z<0]=0
    return dZ

def linear(Z):
    A = Z
    cache = {}
    return A, cache

def linear_der(dA, cache):
    dZ = np.array(dA, copy=True)
    return dZ

def softmax_cross_entropy_loss(Z, Y):
    Y = Y.astype(int)
    m = np.float64(Y.shape[1])
    int_m = Y.shape[1]
    onehot_y = np.zeros(shape=(10, int(m)))
    onehot_y[Y, range(int_m)] = 1
    mz = np.float64(np.max(Z))
    exp1 = np.exp(Z - mz)
    exp2 = np.sum(np.exp(Z - mz), keepdims=True, axis=0)
    A = np.divide(exp1, exp2)
    indices = np.argmax(onehot_y, axis=1).astype(int)
    prob = A[np.arange(len(A)), indices]
    log_preds = np.log(prob)
    loss = -1.0 * np.sum(log_preds) / len(log_preds)
    return A, loss

def softmax_cross_entropy_loss_der(A, Y):
    Y = Y.astype(int)
    m = Y.shape[1]
    onehot_y = np.zeros(shape=(10, int(m)))
    onehot_y[Y, range(m)] = 1
    dZ = A - onehot_y
    return dZ

def linear_forward(A, W, b):
    Z = np.dot(W, A) + b
    cache = {}
    cache["A"] = A
    return Z, cache

def layer_forward(A_prev, W, b, activation):
    Z, lin_cache = linear_forward(A_prev, W, b)
    if activation == "relu":
        A, act_cache = relu(Z)
    elif activation == "linear":
        A, act_cache = linear(Z)
    elif activation == "tanh":
        A, act_cache = tanh(Z)

    cache = {}
    cache["lin_cache"] = lin_cache
    cache["act_cache"] = act_cache
    return A, cache

def multi_layer_forward(X, parameters):
    L = len(parameters) // 2
    A = X
    caches = []
    for l in range(1, L):  # since there is no W0 and b0
        A, cache = layer_forward(A, parameters["W" + str(l)], parameters["b" + str(l)], "tanh")
        caches.append(cache)

    AL, cache = layer_forward(A, parameters["W" + str(L)], parameters["b" + str(L)], "linear")
    caches.append(cache)
    return AL, caches

def linear_backward(dZ, cache, W, b):
    A = cache["A"]
    m = np.float64(A.shape[1])
    dW = np.multiply((1 / m), np.dot(dZ, A.T))
    db = np.multiply((1 / m), np.sum(dZ, axis=1, keepdims=True))
    dA_prev = np.dot(W.T, dZ)
    ## CODE HERE
    return dA_prev, dW, db

def layer_backward(dA, cache, W, b, activation):
    lin_cache = cache["lin_cache"]
    act_cache = cache["act_cache"]

    if activation == "relu":
        dZ = relu_der(dA, act_cache)
    elif activation == "linear":
        dZ = linear_der(dA, act_cache)
    elif activation=="tanh":
        dZ = tanh_der(dA, act_cache)
    dA_prev, dW, db = linear_backward(dZ, lin_cache, W, b)
    return dA_prev, dW, db

def multi_layer_backward(dAL, caches, parameters):
    L = len(caches)
    gradients = {}
    dA = dAL
    activation = "linear"
    for l in reversed(range(1, L + 1)):
        dA, gradients["dW" + str(l)], gradients["db" + str(l)] = \
            layer_backward(dA, caches[l - 1], \
                           parameters["W" + str(l)], parameters["b" + str(l)], \
                           activation)
        activation = "tanh"
    return gradients

def initialize_multilayer_weights(net_dims):
    np.random.seed(0)
    numLayers = len(net_dims)
    parameters = {}
    for l in range(numLayers - 1):
        parameters["W" + str(l + 1)] = np.multiply(np.float64(0.01), np.random.randn(net_dims[l + 1], net_dims[l]))
        parameters["b" + str(l + 1)] = np.multiply(np.float64(0.01), np.random.randn(net_dims[l + 1], 1))
    return parameters

def update_parameters_mlp(parameters, gradients, epoch, learning_rate, decay_rate=0.0):
    alpha = 0.1 * (1 / (1 + decay_rate * epoch))
    L = len(parameters) // 2
    for l in range(L):
        parameters["W" + str(l + 1)] -= alpha * gradients["dW" + str(l + 1)]
        parameters["b" + str(l + 1)] -= alpha * gradients["db" + str(l + 1)]
    return parameters, alpha

def update_parameters(parameters, gradients, epoch, learning_rate, decay_rate=0.01, n_layers=4):
    alpha_trained = 0.02 * (1 / (1 + decay_rate * epoch))
    if n_layers==4:
        parameters["W4"] -= alpha_trained * gradients["dW4"]
        parameters["b4"] -= alpha_trained * gradients["db4"]
    else:
        parameters["W3"] -= alpha_trained * gradients["dW3"]
        parameters["b3"] -= alpha_trained * gradients["db3"]
    return parameters, alpha_trained

def classify(X, parameters):
    Z, cache = multi_layer_forward(X, parameters)
    mz = np.float64(np.max(Z))
    AL = np.divide(np.exp(Z - mz), np.sum(np.exp(Z - mz), keepdims=True, axis=0))
    AL = np.argmax(AL, axis=0)
    return AL

def accuracy(predictions, labels):
    correct_predictions = np.sum(predictions == labels)
    accuracy = 100.0 * correct_predictions / predictions.shape[0]
    return accuracy

def multi_layer_network(n_layer, parameters, X , Y, net_dims, num_epochs, learning_rate, decay_rate=0.01, mlp=False):
    A0 = X
    costs=[]
    for ii in range(num_epochs):
        Z, cache = multi_layer_forward(A0, parameters)
        AL, cost = softmax_cross_entropy_loss(Z, Y)

        dZl = softmax_cross_entropy_loss_der(AL, Y)
        gradients = multi_layer_backward(dZl, cache, parameters)
        if mlp==True:
            parameters, alpha = update_parameters_mlp(parameters, gradients, ii, learning_rate, decay_rate)
        else:
            parameters, alpha = update_parameters(parameters, gradients, ii, learning_rate, decay_rate, n_layer)

        if ii %10 == 0:
            costs.append(cost)
        if ii%100 ==0:
            print("Cost at %i is: %.05f, learning rate: %.05f" % (ii, cost, alpha))

    return parameters, costs


def main():
    digit_range=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    train_data, train_label, test_data, test_label = \
            mnist(noTrSamples=6000,noTsSamples=1000,\
            digit_range=digit_range,\
            noTrPerClass=600, noTsPerClass=100)
    num_epochs = 1000
    learning_rate = 0.5

    #train layer1
    stacked_net_dims = [784, 500, 300, 100, 10]
    n_layer=4
    parameters = stacked_autoencoder(train_data, stacked_net_dims, num_epochs)
    if n_layer==4:
        parameters["W4"] = np.multiply(np.float64(0.01), np.random.randn(stacked_net_dims[4],stacked_net_dims[3]))
        parameters["b4"] = np.multiply(np.float64(0.01), np.random.randn(stacked_net_dims[4],1))
    else:
        parameters["W3"] = np.multiply(np.float64(0.01), np.random.randn(stacked_net_dims[3], stacked_net_dims[2]))
        parameters["b3"] = np.multiply(np.float64(0.01), np.random.randn(stacked_net_dims[3], 1))
    X, Y, t1, t2 = mnist(noTrSamples=10, noTsSamples=10, digit_range=digit_range, noTrPerClass=1,noTsPerClass=1)



    svm_classifier(X.T, Y[0], test_data, test_label)
    knn_classifier(X.T, Y[0], test_data, test_label)
    parameters_mlp=initialize_multilayer_weights(stacked_net_dims)
    parameters_mlp, temp = multi_layer_network(n_layer, parameters_mlp,  X, Y, stacked_net_dims, num_epochs, learning_rate, mlp=True)

    test_Pred_mlp = classify(test_data, parameters_mlp)
    print ("Accuracy for Multi Layer Perceptron model {0:0.3f} %".format(accuracy(test_Pred_mlp, test_label)))

    train_Pred = classify(train_data, parameters)
    test_Pred = classify(test_data, parameters)
    trAcc = accuracy(train_Pred, train_label)
    teAcc = accuracy(test_Pred, test_label)
    print("Accuracy for testing set without finetuning is {0:0.3f} %".format(teAcc))

    parameters, costs = multi_layer_network(n_layer, parameters, X, Y, stacked_net_dims, num_epochs, learning_rate)

    test_Pred = classify(test_data, parameters)

    teAcc = accuracy(test_Pred, test_label)
    print("Accuracy for testing set is {0:0.3f} %".format(teAcc))
    plt.xlabel("iterations")
    plt.ylabel("cost")
    plt.title("Stacked Autoencoder - Iterations vs Cost ")
    plt.plot(costs, "-g", label='train')
    plt.legend()
    plt.show()

def svm_classifier(train_data, train_label, test_data, test_label):
    clf=svm.SVC(gamma=0.1)
    clf.fit(train_data, train_label)
    pred=clf.predict(test_data.T)
    print("SVM accuracy for test {0:0.3f} %".format(accuracy(pred, test_label)))

def knn_classifier(train_data, train_label, test_data, test_label):
    neigh = KNeighborsClassifier(n_neighbors=10)
    neigh.fit(train_data, train_label)
    pred=neigh.predict(test_data.T)
    print("KNN accuracy for test {0:0.3f} %".format(accuracy(pred, test_label)))


if __name__ == "__main__":
    main()