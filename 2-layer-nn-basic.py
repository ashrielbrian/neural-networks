import numpy as np

""" 
    First layer with a single hidden unit, followed by a sigmoid activation function.
    Uses a RMS loss function.
"""
np.random.seed(1)                       # Seeding for deterministic outputs

def sigmoid(x, deriv=False):
    # sigmoid function
    sig = 1 / (1 + np.exp(-x))
    if (deriv == True):
        return sig * (1 - sig)
    return sig

                                        # x has dims (4,5); (m, 5)
x = np.array([  [0,0,1,1,0],
                [1,1,1,1,0],
                [1,0,1,0,1],
                [0,1,1,0,1] ])
m = x.shape[0]                          # m is the number of training examples

                                        # y has dims (4, 1); (m, 1)
y = np.array([  [0],
                [1],
                [1],
                [0] ])

W = np.zeros((5,1))                     # (5,1) weights, corresponding to 5 input features for each e.g.
alpha = 0.1                             # learning rate

for i in range(10000):
    # Forward Pass
    layer_0 = x                         # layer_0 - input layer, x (m, 5)
    layer_1 = np.dot(layer_0, W)        # layer_1 - contains a single hidden unit of weights, col vector W (5,1); (m,1)
    y_hat = sigmoid(layer_1)            # layer_2 - activation layer; (m, 1)
    
    # Loss 
    L = np.sum(((y_hat - y) ** 2)) / (2 * m)                    # average Loss

    # Back Pass
    dy_hat = (y_hat - y)                                        # Loss error, wrt layer_2; (m,1)
    dLayer_1 = dy_hat * sigmoid(layer_1, deriv=True)            # (m,1)
    dW = np.dot(layer_0.T, dLayer_1)                            # gradient on W
    
    # Update parameters
    W = W - alpha * dW

    if i % 2500 == 0:
        print ('Current loss :', L)

print ('Predicted values of y: ', y_hat)




