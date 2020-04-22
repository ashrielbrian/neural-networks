import numpy as np
from random import uniform

""" 
    Character-level LSTM built using numpy.
"""

def sigmoid(x):
    # sigmoid function
    return 1 / (1 + np.exp(-x))
    
np.random.seed(1)

# data I/O
data = open('datasets/shakespearean.txt').read()
chars = list(set(data))                                                                     # List of all unique characters in the data
data_size, vocab_size = len(data), len(chars)
print (f' {data_size} chars in data; {vocab_size} unique chars in vocab')

index_to_char = { i:ch for i, ch in enumerate(chars)}                                       # Dict to map character to its index, and vice versa
char_to_index = { ch:i for i, ch in enumerate(chars)}

# Initialising hyperparams
hidden_units = 120
seq_length = 25

# Initialising params
params = {}                                                                         # stores the parameters
grads = {}                                                                          # stores the parameters params

std = (1.0 / np.sqrt(vocab_size + hidden_units))                                    # Xavier Init
params['Wf'] = np.random.randn(hidden_units, hidden_units + vocab_size) * std       # Forget gate
params['bf'] = np.zeros((hidden_units, 1))

params['Wu'] = np.random.randn(hidden_units, hidden_units + vocab_size) * std       # Update gate
params['bu'] = np.zeros((hidden_units, 1))

params['Wc'] = np.random.randn(hidden_units, hidden_units + vocab_size) * std       # Candidate gate
params['bc'] = np.zeros((hidden_units, 1))

params['Wo'] = np.random.randn(hidden_units, hidden_units + vocab_size) * std       # Output gate
params['bo'] = np.zeros((hidden_units, 1))

params['Wy'] = np.random.randn(vocab_size, hidden_units) * std                      # pre-Softmax hidden layer
params['by'] = np.zeros((vocab_size, 1))

# initialise gradients
for key, p in params.items():
    grads['d' + str(key)] = np.zeros_like(p)

def reset_grads():
    for key in grads.keys():
        grads[key].fill(0)
    return

def computeModel(inputs, targets, c_prev, h_prev):
    """ 
        Arguments
            inputs      - a list of integer inputs corresponding to the sequence of input characters
            targets     - a list of integer inputs corresponding to the sequence of target (correct) characters
            h_prev      - the initial hidden activation state, or previous
            c_prev      - the initial memory cell state, or previous
        Returns
            The loss, the gradients, and the most recent hidden state (rightmost)
    """
    x_state, h_state, c_state, y_state, prob_state, concat_state  = {}, {}, {}, {}, {}, {}             # keeps track of the input, hidden, output, probability states at each timestep
    ft_gate, up_gate, out_gate, c_tilde = {}, {}, {}, {}
    c_tanh = {}                                                         # stores tanh computations that will be reused in backprop
    loss = 0
    h_state[-1] = np.copy(h_prev)                                       # copies the last hidden state from the previous sequence of inputs, as the new initial hidden state
    c_state[-1] = np.copy(c_prev)                                       # candidate state
    
    # Forward pass
    for t, char_index in enumerate(inputs):
        # initialise a one-hot vector for the given character
        x_state[t] = np.zeros((vocab_size, 1))
        x_state[t][char_index] = 1 

        # concat_state is a stacked column vector of the hidden activation and the input vectors, at timestep t        
        concat_state[t] = np.concatenate((h_state[t-1], x_state[t]), axis=0)
        
        ft_gate[t] = sigmoid(np.dot(params['Wf'], concat_state[t]) + params['bf']) 
        up_gate[t] = sigmoid(np.dot(params['Wu'], concat_state[t]) + params['bu'])
        out_gate[t] = sigmoid(np.dot(params['Wo'], concat_state[t]) + params['bo'])
        c_tilde[t] = np.tanh(np.dot(params['Wc'], concat_state[t]) + params['bc'])  # candidate memory cell
        
        c_state[t] = up_gate[t] * c_tilde[t] + ft_gate[t] * c_state[t-1]            # memory cell state

        c_tanh[t] = np.tanh(c_state[t])                                             # stores the tanh calculation to be reused in backprop
        h_state[t] = out_gate[t] * c_tanh[t]                                        # hidden activation state
        y_state[t] = np.dot(params['Wy'], h_state[t]) + params['by']                               
        prob_state[t] = np.exp(y_state[t]) / np.sum(np.exp(y_state[t]))             # softmax classification

        loss += -np.log(prob_state[t][targets[t],0])   

    reset_grads()
    # Backward pass
    dh_next = np.zeros_like(h_state[0])                                 # instantiate the "next" hidden state - required in backprop (rightmost hidden state)
    dc_next = np.zeros_like(c_state[0])
    
    for t in reversed(range(len(inputs))):
        dy = np.copy(prob_state[t])
        dy[targets[t]] -= 1                                             # Backprop into y. See derivation
        
        grads['dWy'] += np.dot(dy, h_state[t].T)

        dh = np.dot(params['Wy'].T, dy) + dh_next
        grads['dby'] += dy

        dc = dh * out_gate[t] * (1 - c_tanh[t] ** 2) + dc_next

        dout_gate = dh * c_tanh[t]
        dz_out = dout_gate * (out_gate[t] * (1 - out_gate[t]))
        grads['dWo'] += np.dot(dz_out, concat_state[t].T)
        grads['dbo'] += dz_out

        dup_gate = dc * c_tilde[t]
        dz_up = dup_gate * (up_gate[t] * (1 - up_gate[t]))
        grads['dWu'] += np.dot(dz_up, concat_state[t].T)
        grads['dbu'] += dz_up
        
        dc_tilde = dc * up_gate[t]
        dz_c_tilde = dc_tilde * (1 - c_tilde[t] ** 2)
        grads['dWc'] += np.dot(dz_c_tilde, concat_state[t].T)
        grads['dbc'] += dz_c_tilde

        dft_gate = dc * c_state[t-1]
        dz_ft = dft_gate * (ft_gate[t] * (1 - ft_gate[t]))
        grads['dWf'] += np.dot(dz_ft, concat_state[t].T)
        grads['dbf'] += dz_ft

        dc_next = dc * ft_gate[t]
        # leftmost hidden state, which will be the rightmost hidden state for the next time-step (reversed)
        dconcat_state = np.dot(params['Wc'].T, dz_c_tilde) + np.dot(params['Wu'].T, dz_up) \
                            + np.dot(params['Wf'].T, dz_ft) + np.dot(params['Wo'].T, dz_out)
        dh_next = dconcat_state[:hidden_units, :]           # unstacking the hidden activations and the inputs
    
    for key in grads.keys():                                # grad clipping to prevent exploding gradients
        np.clip(grads[key], -5, 5, out=grads[key])
    
    # h_state[len(inputs) - 1] returns the most recent hidden state; this will be used as the initial h_state for the next sequence of letters, similarly for c_state
    return loss, h_state[len(inputs) - 1], c_state[len(inputs) - 1]


def gradient_checking(inputs, targets, h_prev, c_prev, num_checks = 5, delta = 1e-5):    
    
    global params, grads
    _, _, _ = computeModel(inputs, targets, c_prev, h_prev)
    grads_original = grads

    text = ''

    for key in params.keys():
        gr_shape = grads['d' + key].shape
        assert (params[key].shape == grads['d' + key].shape), f'Error: dims do not match: {params[key].shape} and {gr_shape}'

        for _ in range(num_checks):
            random_index = int(uniform(0, params[key].size))
            # evaluate cost at (w + e) and (w - e)
            original_val = params[key].flat[random_index]                                         # the original param val at this index
            
            params[key].flat[random_index] = original_val + delta
            loss_plus, _, _ = computeModel(inputs, targets, c_prev, h_prev)
            
            params[key].flat[random_index] = original_val - delta
            loss_minus, _, _ = computeModel(inputs, targets, c_prev, h_prev)
            
            params[key].flat[random_index] = original_val
            grad_analytic = grads_original['d' + key].flat[random_index]
            
            grad_numerical = (loss_plus - loss_minus) / (2 * delta)
            
            relative_err = abs(grad_numerical - grad_analytic) / (abs(grad_numerical) + abs(grad_analytic))
            to_print = f'{key} with i = {random_index}: ({grad_analytic}, {grad_numerical}) => {relative_err} \n'
            text = text + to_print
    return text

def sample(h, c, seed_index, n):
    """ 
        Arguments
            h - most recent hidden state
            seed_index - the index of the seed character
            n - the number of characters to sample
        Returns
            A list of integers representing selected chracter indices
    """
    character_indices = []                                                      # Stores all the characters selected by the model
    x = np.zeros((vocab_size, 1))                                               # instantiate the one-hot vector of the first seed character
    x[seed_index] = 1

    for _ in range(n):
        # single forward pass to sample the next character, given the previous
        concat_state = np.concatenate((h, x), axis=0)

        ft_gate = sigmoid(np.dot(params['Wf'], concat_state) + params['bf'])    # (hidden_units, 1)
        up_gate = sigmoid(np.dot(params['Wu'], concat_state) + params['bu'])
        out_gate = sigmoid(np.dot(params['Wo'], concat_state) + params['bo'])
        c_tilde = np.tanh(np.dot(params['Wc'], concat_state) + params['bc'])    # candidate memory cell
        
        c = up_gate * c_tilde + ft_gate * c                     # memory cell state
        h = out_gate * np.tanh(c)                               # hidden activation state
        y = np.dot(params['Wy'], h) + params['by']                               
        prob = np.exp(y) / np.sum(np.exp(y))                    # softmax classification

        ix = np.random.choice(range(vocab_size), p=prob.ravel())        
        x = np.zeros((vocab_size, 1))                                           # instantiate the next character's one-hot vector
        x[ix] = 1
        character_indices.append(ix)                                            # saves the character selected in time step t, to the list
    
    return character_indices

def update_with_GD(learning_rate = 0.01):
    # Update params using standard GD
    for key in params.keys():
        params[key] += - learning_rate * grads['d' + key]

def update_with_Adam(n, learning_rate = 0.001, beta_1 = 0.9, beta_2 = 0.999, epsilon = 1e-8):
    # Update params using Adam Optimizer
    for key, val in params.items():
        adam_params['v_' + key] = beta_1 * adam_params['v_' + key] + (1 - beta_1) * grads['d' + key]
        v_grad_corrected = adam_params['v_' + key] / (1 - (beta_1 ** n))

        adam_params['s_' + key] = beta_2 * adam_params['s_' + key] + (1 - beta_2) * (grads['d' + key] ** 2)
        s_grad_corrected = adam_params['s_' + key] / (1 - (beta_2 ** n))

        params[key] -= (learning_rate * v_grad_corrected) / (np.sqrt(s_grad_corrected) + epsilon)

# initialise Adam params
adam_params = {}
for key, p in params.items():
    adam_params['v_' + str(key)], adam_params['s_' + str(key)] = np.zeros_like(p), np.zeros_like(p)

smooth_loss = -np.log(1.0/vocab_size) * seq_length                                  # loss at iteration 0

epochs = 10
batches = data_size // seq_length
data_trimmed = data[: batches * seq_length]
gradCheck = False                                                                   # Tracks whether gradcheck has been completed

for epoch in range(epochs):
    h_prev = np.zeros((hidden_units, 1))                                            # clear the RNN memory
    c_prev = np.zeros((hidden_units, 1))
    
    for j in range(0, len(data_trimmed) - seq_length, seq_length):

        # obtaining inputs
        inputs = [char_to_index[ch] for ch in data_trimmed[j: j + seq_length]]
        targets = [char_to_index[ch] for ch in data_trimmed[j + 1: j + seq_length + 1]]

        if (j % 10000 == 0) and (j != 0):
            # sample text
            sampled_indices = sample(h_prev, c_prev, inputs[0], 200)
            text = ''.join([index_to_char[ix] for ix in sampled_indices])
            print ('Sampled text: ', text, '\r')
        
        loss, h_prev, c_prev = computeModel(inputs, targets, c_prev, h_prev)
        smooth_loss = smooth_loss * 0.999 + loss * 0.001                            # weighted average on the loss to reduce volatility
        
        if (j % 100000 == 0) and (j != 0) and (gradCheck ==  True):
            # Gradient checking
            f = open('grad-check-results.txt', 'w')
            f.write(gradient_checking(inputs, targets, h_prev, c_prev))
            f.close()
            gradCheck = False

        if (j % 1000 == 0):
            print (f'Loss: {smooth_loss}, at epoch {epoch} sequence {j}')

        # Adamgrad update
        adam_iter = epoch * epochs + j /seq_length + 1
        update_with_Adam(n = adam_iter)

    






