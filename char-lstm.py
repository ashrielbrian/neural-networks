import numpy as np
from random import uniform

""" 
    Character-level RNN built with reference to @karpathy's gist: https://gist.github.com/karpathy/d4dee566867f8291f086
"""
def sigmoid(x, deriv=False):
    # sigmoid function
    sig = 1 / (1 + np.exp(-x))
    if (deriv == True):
        return sig * (1 - sig)
    return sig

np.random.seed(1)

# data I/O
data = open('datasets/shakespearean.txt').read()
chars = list(set(data))                                                                     # List of all unique characters in the data
data_size, vocab_size = len(data), len(chars)
print (f' {data_size} chars in data; {vocab_size} unique chars in vocab')

index_to_char = { i:ch for i, ch in enumerate(chars)}                                       # Dict to map character to its index, and vice versa
char_to_index = { ch:i for i, ch in enumerate(chars)}

# Initialising hyperparams
hidden_units = 100
seq_length = 25

# Initialising params
std = (1.0 / np.sqrt(vocab_size + hidden_units))
Wf = np.random.randn(hidden_units, hidden_units + vocab_size) * std
bf = np.zeros((hidden_units, 1))
Wu = np.random.randn(hidden_units, hidden_units + vocab_size) * std
bu = np.zeros((hidden_units, 1))
Wc = np.random.randn(hidden_units, hidden_units + vocab_size) * std
bc = np.zeros((hidden_units, 1))
Wo = np.random.randn(hidden_units, hidden_units + vocab_size) * std
bo = np.zeros((hidden_units, 1))
Wy = np.random.randn(vocab_size, hidden_units) * std
by = np.zeros((vocab_size, 1))

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
    c_tanh = {}                   # stores tanh computations that will be reused in backprop
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

        ft_gate[t] = sigmoid(np.dot(Wf, concat_state[t]) + bf)                  # (hidden_units, 1)
        up_gate[t] = sigmoid(np.dot(Wu, concat_state[t]) + bu)
        out_gate[t] = sigmoid(np.dot(Wo, concat_state[t]) + bo)
        c_tilde[t] = np.tanh(np.dot(Wc, concat_state[t]) + bc)                  # candidate memory cell
        
        c_state[t] = up_gate[t] * c_tilde[t] + ft_gate[t] * c_state[t-1]              # memory cell state

        c_tanh[t] = np.tanh(c_state[t])                                 # stores the tanh calculation to be reused in backprop
        h_state[t] = out_gate[t] * c_tanh[t]                             # hidden activation state
        y_state[t] = np.dot(Wy, h_state[t]) + by                               
        prob_state[t] = np.exp(y_state[t]) / np.sum(np.exp(y_state[t]))         # softmax classification
        loss += -np.log(prob_state[t][targets[t],0])   

    # Backward pass
    dWf, dWu, dWc, dWo, dWy = np.zeros_like(Wf), np.zeros_like(Wu), np.zeros_like(Wc), np.zeros_like(Wo), np.zeros_like(Wy)
    dbf, dbu, dbc, dbo, dby = np.zeros_like(bf), np.zeros_like(bu), np.zeros_like(bc), np.zeros_like(bo), np.zeros_like(by)
    dh_next = np.zeros_like(h_state[0])                                                         # instantiate the "next" hidden state - required in backprop (rightmost hidden state)
    dc_next = np.zeros_like(c_state[0])
    
    for t in reversed(range(len(inputs))):
        dy = np.copy(prob_state[t])
        dy[targets[t]] -= 1                                                                     # Backprop into y. See derivation
        
        dWy += np.dot(dy, h_state[t].T)
        dh = np.dot(Wy.T, dy) + dh_next
        dby += dy

        dout_gate = dh * c_tanh[t]                
        dc = dh * out_gate[t] * (1 - c_tanh[t] * c_tanh[t]) + dc_next
        
        dz_out = dout_gate * (out_gate[t] * (1 - out_gate[t]))
        dbo += dz_out
        dWo += np.dot(dz_out, concat_state[t].T)
        
        dft_gate = dc * c_state[t-1]
        dc_next = dc * ft_gate[t]
        dup_gate = dc * c_tilde[t]
        dc_tilde = dc * up_gate[t]

        dz_up = dup_gate * (up_gate[t] * (1 - up_gate[t]))
        dWu += np.dot(dz_up, concat_state[t].T)
        dbu += dz_up
        
        dz_c_tilde = dc_tilde * (1 - c_tilde[t] * c_tilde[t])
        dWc += np.dot(dz_c_tilde, concat_state[t].T)
        dbc += dz_c_tilde

        dz_ft = dft_gate * (ft_gate[t] * (1 - ft_gate[t]))
        dWf += np.dot(dz_ft, concat_state[t].T)
        dbf += dz_ft

        dconcat_state = np.dot(Wc.T, dz_c_tilde) + np.dot(Wu.T, dz_up) + np.dot(Wf.T, dz_ft) + np.dot(Wo.T, dz_out)

        # leftmost hidden state, which will be the rightmost hidden state for the next time-step (reversed)
        dh_next = dconcat_state[:hidden_units, :]           # unstacking between the hidden activations and the inputs
    for d in [dWf, dWu, dWc, dWo, dWy, dbf, dbu, dbc, dbo, dby]:                                # grad clipping to prevent exploding gradients
        np.clip(d, -5, 5, d)
    return loss, dWf, dWu, dWc, dWo, dWy, dbf, dbu, dbc, dbo, dby, h_state[len(inputs) - 1], c_state[len(inputs) - 1]                           # h_state[len(inputs) - 1] returns the most recent hidden state; this will be used as the initial h_state for the next sequence of letters

def update_with_Adam(param, grad, v_grad, s_grad, n, learning_rate = 0.001, beta_1 = 0.9, beta_2 = 0.999, epsilon = 1e-8):
    """ 
        v_grad - momentum weighted-grad
        s_grad - RMSprop weighted-grad
        Returns
            the updated parameter
    """
    v_grad = beta_1 * v_grad + (1 - beta_1) * grad
    v_grad_corrected = v_grad / (1 - (beta_1 ** n))

    s_grad = beta_2 * s_grad + (1 - beta_2) * (grad ** 2)
    s_grad_corrected = s_grad / (1 - (beta_2 ** n))

    param += - (learning_rate * v_grad_corrected) / (np.sqrt(s_grad_corrected) + epsilon)
    return param

def gradient_checking(inputs, targets, h_prev, c_prev):
    num_checks, delta = 15, 1e-5
    global Wf, Wu, Wc, Wo, Wy, bf, bu, bc, bo, by
    _, dWf, dWu, dWc, dWo, dWy, dbf, dbu, dbc, dbo, dby, _, _ = computeModel(inputs, targets, c_prev, h_prev)

    for param, grad, name in zip(
                                [Wf, Wu, Wc, Wo, Wy, bf, bu, bc, bo, by], 
                                [dWf, dWu, dWc, dWo, dWy, dbf, dbu, dbc, dbo, dby], 
                                ['Wf', 'Wu', 'Wc', 'Wo', 'Wy', 'bf', 'bu', 'bc', 'bo', 'by']):

        assert (param.shape == grad.shape), f'Error: dims do not match: {param.shape} and {grad.shape}'
        print (name)
        for _ in range(num_checks):
            random_index = int(uniform(0, param.size))
            # evaluate cost at (w + e) and (w = e)
            original_val = param.flat[random_index]             # the original param val at this index
            param.flat[random_index] = original_val + delta
            loss_gt, _, _, _, _, _, _, _, _,_, _, _, _ = computeModel(inputs, targets, c_prev, h_prev)
            param.flat[random_index] = original_val - delta
            loss_sm, _, _, _, _, _, _, _, _,_, _, _, _ = computeModel(inputs, targets, c_prev, h_prev)
            param.flat[random_index] = original_val

            grad_analytic = grad.flat[random_index]
            grad_numerical = (loss_gt - loss_sm) / (2 * delta)
            relative_err = abs(grad_numerical - grad_analytic) / (abs(grad_numerical + grad_analytic))
            print (f'{name} with i = {random_index}, {original_val}: ({grad_analytic}, {grad_numerical}) => {relative_err}')

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

        ft_gate = sigmoid(np.dot(Wf, concat_state) + bf)                  # (hidden_units, 1)
        up_gate = sigmoid(np.dot(Wu, concat_state) + bu)
        out_gate = sigmoid(np.dot(Wo, concat_state) + bo)
        c_tilde = np.tanh(np.dot(Wc, concat_state) + bc)                  # candidate memory cell
        
        c = up_gate * c_tilde + ft_gate * c              # memory cell state

        h = out_gate * np.tanh(c)                             # hidden activation state
        y = np.dot(Wy, h) + by                               
        prob = np.exp(y) / np.sum(np.exp(y))         # softmax classification

        ix = np.random.choice(range(vocab_size), p=prob.ravel())        
        x = np.zeros((vocab_size, 1))                                           # instantiate the next character's one-hot vector
        x[ix] = 1
        
        character_indices.append(ix)                                            # saves the character selected in time step t, to the list
    
    return character_indices

def update_with_GD(param, grad, learning_rate = 0.001):
    param += - learning_rate * grad
    return param
pointer, n = 0, 0                                                                   # tracks the current index of the dataset (used to determine subets of the sequences) and the number of iterations

v_Wf, v_Wu, v_Wc, v_Wo, v_Wy = np.zeros_like(Wf), np.zeros_like(Wu), np.zeros_like(Wc), np.zeros_like(Wo), np.zeros_like(Wy) # instantiate the weighted grads; to be used in the Adam optimization (Momentum)
v_bf, v_bu, v_bc, v_bo, v_by = np.zeros_like(bf), np.zeros_like(bu), np.zeros_like(bc), np.zeros_like(bo), np.zeros_like(by)
s_Wf, s_Wu, s_Wc, s_Wo, s_Wy = np.zeros_like(Wf), np.zeros_like(Wu), np.zeros_like(Wc), np.zeros_like(Wo), np.zeros_like(Wy) # instantiate the weighted grads; to be used in the Adam optimization (RMS prop)
s_bf, s_bu, s_bc, s_bo, s_by = np.zeros_like(bf), np.zeros_like(bu), np.zeros_like(bc), np.zeros_like(bo), np.zeros_like(by)

smooth_loss = -np.log(1.0/vocab_size) * seq_length                                  # loss at iteration 0


epochs = 10
batches = data_size // seq_length
data_trimmed = data[: batches * seq_length]

print (len(data_trimmed) - seq_length)
for epoch in range(epochs):
    h_prev = np.zeros((hidden_units, 1))                                    # clear the RNN memory
    c_prev = np.zeros((hidden_units, 1))
    pointer = 0                                                             # go back to the start of the data
    
    for j in range(0, len(data_trimmed) - seq_length, seq_length):
        # obtaining inputs
        inputs = [char_to_index[ch] for ch in data_trimmed[j: j + seq_length]]
        targets = [char_to_index[ch] for ch in data_trimmed[j + 1: j + seq_length + 1]]
        if (j % 10000 == 0):
            sampled_indices = sample(h_prev, c_prev, inputs[0], 200)
            text = ''.join([index_to_char[ix] for ix in sampled_indices])
            print ('Sampled text: ', text, '\r')
        loss, dWf, dWu, dWc, dWo, dWy, dbf, dbu, dbc, dbo, dby, h_prev, c_prev = computeModel(inputs, targets, c_prev, h_prev)
        smooth_loss = smooth_loss * 0.999 + loss * 0.001                            # weighted average on the loss to reduce volatility
        
        if (j % 1000 == 0):
            print (f'Loss: {smooth_loss}, at epoch {epoch} sequence {j}')
        
        # Adamgrad update
        adam_iter = (epoch * epochs) + 1
        for param, grad, v_grad, s_grad in zip(
                                [Wf, Wu, Wc, Wo, Wy, bf, bu, bc, bo, by], 
                                [dWf, dWu, dWc, dWo, dWy, dbf, dbu, dbc, dbo, dby], 
                                [v_Wf, v_Wu, v_Wc, v_Wo, v_Wy, v_bf, v_bu, v_bc, v_bo, v_by],
                                [s_Wf, s_Wu, s_Wc, s_Wo, s_Wy, s_bf, s_bu, s_bc, s_bo, s_by]):
            #param = update_with_GD(param, grad)
            param = update_with_Adam(param, grad, v_grad, s_grad, n = adam_iter)            # n + 1 to avoid beta_1 ** 0 on first iteration
        #if  (j % 2000 == 0) and (j != 0):
            #gradient_checking(inputs, targets, h_prev, c_prev)
    




""" while n <= 500:

    if (pointer + seq_length >= len(data)) or (n == 0):
        h_prev = np.zeros((hidden_units, 1))                                    # clear the RNN memory
        c_prev = np.zeros((hidden_units, 1))
        pointer = 0                                                             # go back to the start of the data

    # obtaining inputs
    inputs = [char_to_index[ch] for ch in data[pointer: pointer + seq_length]]
    targets = [char_to_index[ch] for ch in data[pointer + 1: pointer + seq_length + 1]]

    if n % 1000 == 0:
        # sample the model
        sampled_indices = sample(h_prev, c_prev, inputs[0], 200)
        text = ''.join([index_to_char[ix] for ix in sampled_indices])
        print ('Sampled text: ', text, '\r')

    loss, dWf, dWu, dWc, dWo, dWy, dbf, dbu, dbc, dbo, dby, h_prev, c_prev = computeModel(inputs, targets, c_prev, h_prev)
    smooth_loss = smooth_loss * 0.999 + loss * 0.001                            # weighted average on the loss to reduce volatility
    if n % 100 == 0:
        print (f'At iteration {n}, loss of {smooth_loss}. \r')
    
    # Adamgrad update
    for param, grad, v_grad, s_grad in zip(
                            [Wf, Wu, Wc, Wo, Wy, bf, bu, bc, bo, by], 
                            [dWf, dWu, dWc, dWo, dWy, dbf, dbu, dbc, dbo, dby], 
                            [v_Wf, v_Wu, v_Wc, v_Wo, v_Wy, v_bf, v_bu, v_bc, v_bo, v_by],
                            [s_Wf, s_Wu, s_Wc, s_Wo, s_Wy, s_bf, s_bu, s_bc, s_bo, s_by]):
        
        param = update_with_Adam(param, grad, v_grad, s_grad, n = n + 1)            # n + 1 to avoid beta_1 ** 0 on first iteration

    n += 1
    pointer += seq_length

    if (n % 500 == 0) and (n != 0):
        gradient_checking(inputs, targets, h_prev, c_prev) """





