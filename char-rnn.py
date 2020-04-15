import numpy as np

""" 
    Character-level RNN built with reference to @karpathy's gist: https://gist.github.com/karpathy/d4dee566867f8291f086
"""

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
alpha = 0.03

# Initialising params
Wxh = np.random.randn(hidden_units, vocab_size) * 0.01                  # parameters input to hidden; 0.01 is required, otherwise can cause overflow in activation func
Whh = np.random.randn(hidden_units, hidden_units) * 0.01                # hidden to hidden (L-R)
Why = np.random.randn(vocab_size, hidden_units) * 0.01                  # hidden to output
bh = np.zeros((hidden_units, 1))
by = np.zeros((vocab_size, 1))

def computeModel(inputs, targets, h_prev):
    """ 
        Arguments
            inputs      - a list of integer inputs corresponding to the sequence of input characters
            targets     - a list of integer inputs corresponding to the sequence of target (correct) characters
            h_prev      - the initial hidden state, or previous
        Returns
            The loss, the gradients, and the most recent hidden state (rightmost)
    """
    x_state, h_state, y_state, prob_state  = {}, {}, {}, {}             # keeps track of the input, hidden, output, probability states at each timestep
    loss = 0
    h_state[-1] = np.copy(h_prev)                                       # copies the last hidden state from the previous sequence of inputs, as the new initial hidden state
    
    # Forward pass
    for t, char_index in enumerate(inputs):
        # initialise a one-hot vector for the given character
        x_state[t] = np.zeros((vocab_size, 1))
        x_state[t][char_index] = 1

        h_state[t] = np.tanh(np.dot(Wxh, x_state[t]) + np.dot(Whh, h_state[t - 1]) + bh)        # computing hidden state (post-activation)
        y_state[t] = np.dot(Why, h_state[t]) + by
        prob_state[t] = np.exp(y_state[t]) / np.sum(np.exp(y_state[t]))                         # softmax classification
        loss += -np.log(prob_state[t][targets[t],0])                                            # compute cost

    # Backward pass
    dWxh, dWhh, dWhy = np.zeros_like(Wxh), np.zeros_like(Whh), np.zeros_like(Why)
    dbh, dby = np.zeros_like(bh), np.zeros_like(by)
    dh_next = np.zeros_like(h_state[0])                                                         # instantiate the "next" hidden state - required in backprop (rightmost hidden state)

    for t in reversed(range(len(inputs))):
        dy = np.copy(prob_state[t])
        dy[targets[t]] -= 1                                                                     # Backprop into y. See derivation
        dWhy += np.dot(dy, h_state[t].T)                                                        
        dby += dy
        dh = np.dot(Why.T, dy) + dh_next                                                        # add the rightmost hidden state since this also contributes to the gradient
        dhraw = (1 - h_state[t] * h_state[t]) * dh                                              # hraw = (Whh * h<t-1> + bh) + (Wxh * X<t>), i.e. right before tanh activation  

        dbh += dhraw
        dWxh += np.dot(dhraw, x_state[t].T)
        dWhh += np.dot(dhraw, h_state[t-1].T)
        dh_next = np.dot(Whh.T, dhraw)                                                          # leftmost hidden state, which will be the rightmost hidden state for the next time-step (reversed)
    
    for d in [dWxh, dWhh, dWhy, dbh, dby]:                                                      # grad clipping to prevent exploding gradients
        np.clip(d, -5, 5, d)
    return loss, dWxh, dWhh, dWhy, dbh, dby, h_state[len(inputs) - 1]                           # the last var returns the most recent hidden state; this will be used as the initial h_state for the next sequence of letters


def sample(h, seed_index, n):
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

    for t in range(n):
        # single forward pass to sample the next character, given the previous
        h = np.tanh(np.dot(Wxh, x) + np.dot(Whh, h) + bh)
        y = np.dot(Why, h) + by
        prob = np.exp(y) / np.sum(np.exp(y))
        ix = np.random.choice(range(vocab_size), p=prob.ravel())
        
        x = np.zeros((vocab_size, 1))                                           # instantiate the next character's one-hot vector
        x[ix] = 1
        
        character_indices.append(ix)                                            # saves the character selected in time step t, to the list
    
    return character_indices

pointer, n = 0, 0                                                               # tracks the current index of the dataset (used to determine subets of the sequences) and the number of iterations
mWxh, mWhh, mWhy = np.zeros_like(Wxh), np.zeros_like(Whh), np.zeros_like(Why)   # instantiate the memory for grads; to be used in the Adagrad optimization
mbh, mby = np.zeros_like(bh), np.zeros_like(by)
smooth_loss = -np.log(1.0/vocab_size) * seq_length                               # loss at iteration 0

while n <= 50000:

    if (pointer + seq_length >= len(data)) or (n == 0):
        h_prev = np.zeros((hidden_units, 1))                                    # clear the RNN memory
        pointer = 0                                                             # go back to the start of the data

    # obtaining inputs
    inputs = [char_to_index[ch] for ch in data[pointer: pointer + seq_length]]
    targets = [char_to_index[ch] for ch in data[pointer + 1: pointer + seq_length + 1]]

    if n % 1000 == 0:
        # sample the model
        sampled_indices = sample(h_prev, inputs[0], 200)
        text = ''.join([index_to_char[ix] for ix in sampled_indices])
        print ('Sampled text: ', text, '\r')

    loss, dWxh, dWhh, dWhy, dbh, dby, h_prev = computeModel(inputs, targets, h_prev)
    smooth_loss = smooth_loss * 0.999 + loss * 0.001                            # weighted average on the loss to reduce volatility
    if n % 100 == 0:
        print (f'At iteration {n}, loss of {smooth_loss}. \r')
    
    # Adagrad update
    for param, grad, mem in zip([Wxh, Whh, Why, bh, by], [dWxh, dWhh, dWhy, dbh, dby], [mWxh, mWhh, mWhy, mbh, mby]):

        mem += grad * grad
        param += - alpha * grad / np.sqrt(mem + 1e-8)

    if n % 50000 == 0 and n != 0:
        # Simple check on grads. This will show vanishing gradients on many parameters, esp on dWxh
        print ('Grads: ', dWxh, dWhh, dWhy)
    n += 1
    pointer += seq_length
