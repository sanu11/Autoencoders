import numpy as np
import matplotlib.pyplot as plt
from mnist_reader import load_mnist
from load_mnist import mnist

def relu(Z):
    A = np.maximum(0,Z)
    cache = {}
    cache["A"] = A
    cache["Z"] = Z
    return A, cache

def relu_der(dA, cache):
    dZ = np.array(dA, copy=True)
    Z = cache["Z"]
    dZ[Z < 0] = 0
    return dZ

def sigmoid(Z):
    A = 1/(1+np.exp(-Z))
    cache = {}
    cache["A"] = A
    cache["Z"] = Z
    return A, cache

def sigmoid_der(dA, cache):
    Z = cache["Z"]
    A, cache = sigmoid(Z)
    dZ = dA * A * (1 - A)
    return dZ

def tanh(Z):
    A = np.tanh(Z)
    cache = {}
    cache["Z"] = Z
    return A, cache

def tanh_der(dA, cache):
    Z = cache["Z"]
    A, cache = tanh(Z)
    dZ = dA * (1 - A * A)
    return dZ

def linear_forward(A, W, b):
    Z = np.dot(W, A) + b
    cache = {}
    cache["Z"] = Z
    cache["A"] = A
    return Z, cache

def layer_forward(A_prev, W, b, activation):
    Z, lin_cache = linear_forward(A_prev, W, b)
    if activation == "sigmoid":
        A, act_cache = sigmoid(Z)
    elif activation == "relu":
        A, act_cache = relu(Z)
    cache = {}
    cache["lin_cache"] = lin_cache
    cache["act_cache"] = act_cache
    return A, cache

def linear_backward(dZ, W, cache):
    dA_prev = np.dot(W.T, dZ)
    dW = np.dot(dZ, cache["A"].T)
    db = np.sum(dZ, axis=1, keepdims=True)
    return dA_prev, dW, db

def layer_backward(dA, W, cache, activation):
    lin_cache = cache["lin_cache"]
    act_cache = cache["act_cache"]

    if activation == "sigmoid":
        dZ = sigmoid_der(dA, act_cache)
    elif activation == "relu":
        dZ = relu_der(dA, act_cache)
    elif activation == "tanh":
        dZ = tanh_der(dA, act_cache)

    dA_prev, dW, db = linear_backward(dZ, W, lin_cache)
    return dA_prev, dW, db

def cost_estimate(A2, Y):
    cost = np.sum((A2 - Y)**2) / float(Y.shape[1])
    return cost

def initialize_2layer_weights(n_in, n_h, n_out):
    W1 = np.random.normal(0, (2 / float(n_in)) ** 0.5, (n_h, n_in))
    b1 = np.random.normal(0, (2 / float(n_in)) ** 0.5, (n_h, 1))
    b2 = np.random.normal(0, (2 / float(n_h)) ** 0.5, (n_out, 1))

    parameters = {}
    parameters["W1"] = W1
    parameters["b1"] = b1
    parameters["W2"] = W1.T
    parameters["b2"] = b2
    return parameters

def encoder_decoder_twolayer(X, denoised_X, net_dims, num_epochs, batch_size, learning_rate):
    n_in, n_h, n_out = net_dims
    parameters = initialize_2layer_weights(n_in, n_h, n_out)

    costs = []

    for ii in range(num_epochs):
        for j in range(X.shape[1]//batch_size):
            batch = X[:, batch_size*j : (batch_size*j)+batch_size]
            denoised_batch = denoised_X[:, batch_size*j : (batch_size*j)+batch_size]

            A1, cache1 = layer_forward(batch, parameters["W1"], parameters["b1"], "sigmoid")
            A2, cache2 = layer_forward(A1, parameters["W2"], parameters["b2"], "sigmoid")


            dA2 = (A2 - denoised_batch)/denoised_batch.shape[1]
            dA1, dW2, db2 = layer_backward(dA2, parameters["W2"], cache2, "sigmoid")
            dA0, dW1, db1 = layer_backward(dA1, parameters["W1"], cache1, "sigmoid")

            parameters["W1"] = parameters["W1"] - learning_rate * dW1
            parameters["b1"] = parameters["b1"] - learning_rate * db1
            parameters["W2"] = parameters["W2"] - learning_rate * dW2
            parameters["b2"] = parameters["b2"] - learning_rate * db2

        if ii % 10 == 0:
          A1, cache1 = layer_forward(X, parameters["W1"], parameters["b1"], "sigmoid")
          A2, cache2 = layer_forward(A1, parameters["W2"], parameters["b2"], "sigmoid")

          cost = cost_estimate(A2, denoised_X)
          costs.append(cost)
          print("Cost at epoch %i is: %f" %(ii, cost))
    return parameters, costs

def noisyMethod1(X_train):  #sigmoid? #more hidden neurons #percentage noise #cross entropy loss/squared diff?
    samples = X_train.shape[0]
    # noise1 = np.random.normal(1, 0.1, (samples//2, X_train.shape[1]))
    # noise2 = np.random.normal(1, 0.2, (samples//2, X_train.shape[1]))
    #X_noisytrain = np.zeros(X_train.shape)   
    # X_noisytrain[0:samples//2, :] = X_train[0:samples//2, :] * noise1
    # X_noisytrain[samples//2:samples, :] = X_train[samples//2:samples, :] * noise2
    noise = np.random.normal(1, 0.2, (samples, X_train.shape[1]))
    X_noisytrain = X_train * noise
    return X_noisytrain
  
def noisyMethod2(X_train, noisyPercentage):
    X_train.shape
    pixelIndices = np.arange(X_train.shape[1])
    numNoisyPixels = (noisyPercentage * X_train.shape[1])//100
    X_noisytrain = np.copy(X_train)
    for i in range(X_train.shape[0]):
      indices = np.random.choice(pixelIndices,numNoisyPixels)
      X_noisytrain[i,indices] = 0.0
    return X_noisytrain

def main():
    digit_range = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    X_train, _, X_test , _ = \
        mnist(noTrSamples=6000, noTsSamples=1000, \
              digit_range=digit_range, \
              noTrPerClass=600, noTsPerClass=100)

    X_noisytrain = noisyMethod1(X_train)
    # X_noisytrain = noisyMethod2(X_train, 10)
    
    # X_noisytrain = X_noisytrain.T
    # X_train = X_train.T

    num_epochs = 1000
    learning_rate = 0.01
    batch_size = 256

    net_dims = [784, 1000, 784]
    parameters, costs = encoder_decoder_twolayer(X_noisytrain, X_train, net_dims, num_epochs, batch_size, learning_rate)

    l1,_= sigmoid(np.dot(parameters["W1"],X_train ) + parameters["b1"])
    Image,_ = sigmoid(np.dot(parameters["W2"],l1) + parameters["b2"])

    fig,axes = plt.subplots(3,5)
    for i in range(5):
        axes[0,i].imshow(X_train[:, i].reshape((28, 28)), cmap="gray")
        axes[1,i].imshow(X_noisytrain[:,i].reshape((28,28)), cmap='gray')
        axes[2,i].imshow(Image[:, i].reshape((28,28)), cmap="gray")
    plt.savefig("Original_Noisy_Reconstructed_Images.png")
    plt.show()

if __name__ == "__main__":
    main()

