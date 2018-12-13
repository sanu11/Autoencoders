import numpy as np

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
    elif activation == "tanh":
        A, act_cache = tanh(Z)
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

def encoder_decoder_twolayer(X,  net_dims, num_epochs, batch_size, learning_rate):
    n_in, n_h, n_out = net_dims
    parameters = initialize_2layer_weights(n_in, n_h, n_out)
    costs = []
    act="tanh"

    for ii in range(num_epochs):
        for j in range(X.shape[1]//batch_size):
            batch = X[:, batch_size*j : (batch_size*j)+batch_size]

            A1, cache1 = layer_forward(batch, parameters["W1"], parameters["b1"], act)
            A2, cache2 = layer_forward(A1, parameters["W2"], parameters["b2"], act)


            dA2 = (A2 - batch)/batch.shape[1]
            dA1, dW2, db2 = layer_backward(dA2, parameters["W2"], cache2, act)
            dA0, dW1, db1 = layer_backward(dA1, parameters["W1"], cache1, act)

            alpha = learning_rate * ( 1 / (1 + 0.01 * ii))
            parameters["W1"] = parameters["W1"] - alpha * dW1
            parameters["b1"] = parameters["b1"] - alpha * db1
            parameters["W2"] = parameters["W2"] - alpha * dW2
            parameters["b2"] = parameters["b2"] - alpha * db2

        if ii % 10 == 0:
            A1, cache1 = layer_forward(X, parameters["W1"], parameters["b1"], act)
            A2, cache2 = layer_forward(A1, parameters["W2"], parameters["b2"], act)

            cost = cost_estimate(A2, X)
            costs.append(cost)
        if ii%100==0:
            print("Cost at epoch %i is: %f" %(ii, cost))

    H, temp= layer_forward(X, parameters["W1"], parameters["b1"], "tanh")
    return H, parameters["W1"], parameters["b1"]#,parameters["W2"],parameters["b2"]


def stacked_autoencoder(train_data, stack_net_dims,num_epochs):
    parameters={}
    net_dims=[784, stack_net_dims[1], 784]
    learning_rate1=0.1
    print("Learning rate of layer 1="+str(learning_rate1))
    learning_rate2 = 0.1
    print("Learning rate of layer 2=" + str(learning_rate2))
    learning_rate3 = 0.15
    print("Learning rate of layer 3=" + str(learning_rate3))
    H_1, parameters["W1"], parameters["b1"]=encoder_decoder_twolayer(train_data, net_dims, num_epochs, 256, learning_rate1)
    net_dims=[stack_net_dims[1], stack_net_dims[2], stack_net_dims[1]]
    H_2, parameters["W2"], parameters["b2"]=encoder_decoder_twolayer(H_1, net_dims, num_epochs, 256, learning_rate2)
    net_dims=[stack_net_dims[2], stack_net_dims[3], stack_net_dims[2]]
    H_3, parameters["W3"], parameters["b3"]=encoder_decoder_twolayer(H_2, net_dims, num_epochs, 256, learning_rate3)
    return parameters



