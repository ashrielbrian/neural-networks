import numpy as np
import h5py
from load_dataset import load_dataset

""" 
    A model to predict whether an image is a cat or not. Uses LINEAR -> ReLU -> LINEAR -> ReLU -> LINEAR -> SIGMOID
    Dataset and the load_data boilerplate are taken from deeplearning.ai. Everything else is my own.

    Each input image is size (64,64,3). We flatten the image to a column vector (12288, 1), and stacking by examples,
    yielding X  with dims (12288, m).
"""
np.random.seed(1)

def relu(x):
    return np.maximum(x,0)

def reluDeriv(dA, Z):
    """ 
        Backward pass on the ReLU activation. This combines the chain-rule into a single function, 
        i.e. dLayer_2 = da_layer_2 * relu'(layer_2), is now simply dLayer_2 = reluDeriv(da_layer_2, layer_2) 
    """
    dZ = np.array(dA, copy=True)                                        # Makes a copy of the post-activation array
    dZ[Z <= 0] = 0                                                      # Sets to zero, all neurons that were switched off in the earlier forward pass.
    return dZ

def sigmoid(x, deriv=False):
    sig = 1 / (1 + np.exp(-x))
    if (deriv == True):
        return sig * (1 - sig)
    return sig
    
def initialise(n_x):
    """ 
        Argument: 
            n_x - number of input features from the dataset
        Returns:
            W - tuple of weights; (n_h, n_x) - n_h: number of hidden units, n_x: number of input features
    """
    layer_1_units = 20
    layer_2_units = 7
    W0 = np.random.randn(layer_1_units, n_x) * 0.01                         # first layer FC
    W1 = np.random.randn(layer_2_units, layer_1_units) * 0.01               # second layer of weights; 4 parameters, 2 units
    W2 = np.random.randn(1, layer_2_units) * 0.01
    return (W0, W1, W2)

def forward(X, W0, W1, W2):
    # Forward pass
    layer_0 = X
    layer_1 = np.dot(W0, layer_0)                   # (layer_1_units, m)
    a_layer_1 = relu(layer_1)                       # has dims (layer_1_units,m)
    layer_2 = np.dot(W1, a_layer_1)                 # layer_2 has dims (2,m)
    a_layer_2 = relu(layer_2)                       # (layer_2_units,m)
    layer_3 = np.dot(W2, a_layer_2)                 # (1,m)
    Y_hat = sigmoid(layer_3)                        # (1,m) predicted values

    # Caches the activations for the backward pass computation
    cache = {"W0": W0, "W1": W1, "W2": W2, "layer_0": layer_0, "layer_1": layer_1, "a_layer_1": a_layer_1, "layer_2": layer_2, "a_layer_2": a_layer_2, "layer_3": layer_3 }
    return Y_hat, cache

def computeLoss(Y, Y_hat):
    # Computing the error 
    m = Y.shape[1]
    L = - (Y * np.log(Y_hat) + (1 - Y) * np.log(1 - Y_hat))         # Cross-entropy
    cost = (1/m) * np.sum(L)                                        # Average cost
    return cost

def backward(Y, Y_hat, cache):
    # Backward pass
    m = Y.shape[1]
    W1, W2, layer_0, layer_1, a_layer_1, layer_2, a_layer_2, layer_3 = cache['W1'], cache["W2"], cache['layer_0'], cache['layer_1'], cache['a_layer_1'], cache['layer_2'], cache["a_layer_2"], cache["layer_3"]
    dY_hat = - np.divide(Y, Y_hat) + np.divide(1 - Y,1 - Y_hat)     # (1,m)
    dLayer_3 = dY_hat * sigmoid(layer_3, deriv=True)                # (1,m)
    da_layer_2 = np.dot(W2.T, dLayer_3)                             # (layer_2_units,m)
    dW2 = (1/m) * np.dot(dLayer_3, a_layer_2.T)                     # (1,layer_2_units)
    dLayer_2 = reluDeriv(da_layer_2, layer_2)                       # (2,m)
    da_layer_1 = np.dot(W1.T, dLayer_2)                             # (layer_1_units,m)
    dW1 = (1/m) * np.dot(dLayer_2, a_layer_1.T)                     # (layer_2_units, layer_1_units)
    dLayer_1 = reluDeriv(da_layer_1, layer_1)                       # (2,m)
    dW0 = (1/m) * np.dot(dLayer_1, layer_0.T)                       # (layer_1_units, n_x)
    return dW0, dW1, dW2

def update(grads, cache, alpha):
    # Update parameters
    dW0, dW1, dW2 = grads
    W0, W1, W2 = cache["W0"], cache["W1"], cache["W2"]
    W2 = W2 - alpha * dW2
    W1 = W1 - alpha * dW1
    W0 = W0 - alpha * dW0
    return W0, W1, W2

def computeAccuracy(Y, Y_hat):
    m = Y.shape[1]
    return 1 - (np.sum(abs(Y - Y_hat)) / m)

# Load dataset and initialise weights
(train_set_x_orig, train_set_y_orig, test_set_x_orig, test_set_y_orig, classes) = load_dataset()
X_train = train_set_x_orig.reshape(train_set_x_orig.shape[0], -1).T                                     # (12288, m_train)
X_test = test_set_x_orig.reshape(test_set_x_orig.shape[0], -1).T                                        # (12288, m_test)

# Normalising
X_train = X_train/255
X_test = X_test/255

m_train = X_train.shape[1]
m_test = X_test.shape[1]
alpha = 0.01

W0, W1, W2 = initialise(X_train.shape[0])

for i in range(8000):
    Y_hat, cache = forward(X_train, W0, W1, W2)
    cost = computeLoss(train_set_y_orig, Y_hat)
    grads = backward(train_set_y_orig, Y_hat, cache)
    W0, W1, W2 = update(grads, cache, alpha)

    if i % 2000 == 0:
        print ('Current Loss: ', cost)

train_accuracy = computeAccuracy(train_set_y_orig, Y_hat)
Y_hat_test, _ = forward(X_test, W0, W1, W2)
test_accuracy = computeAccuracy(test_set_y_orig, Y_hat_test)

print ('Training accuracy: -----> ', train_accuracy)
print ('Test accuracy: -----> ', test_accuracy)








    
