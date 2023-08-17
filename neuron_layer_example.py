import numpy as np

# x1,x2 -- w1,w2 -> N + b --> Output
# Output = n(x1 * w1 + x2 * w2 + b)

# this model represents a single neuron layer with 8 neurons which receives 5 inputs
# each neuron has a bias and a weight for each input
# 
# every parameter and input are being randomly generated and
# we're testing a batch of four input-sets  



def output_for(input,weights,bias):
    '''
    A function to calculate the output of the neuron layer for a batch of inputs
    using "for" loops instead of matrix dot product.
    '''
    output_for = []
    for input_n in input:
        out_n = []
        for neuron in range(len(weights)):
            out = sum(input_n*weights[neuron])
            out_n.append(out)
        output_for.append(out_n)
    output_for = output_for + bias
    return np.array(output_for)

def normalize_2d(matrix):
    '''
    A function to normalize a matrix
    '''
    norm = np.linalg.norm(matrix)
    matrix = matrix/norm
    return matrix

input = np.random.sample((4,5))  # a batch of 4 input-sets
weights = 2* np.random.sample((8,5)) - 1 # the weights for each connection for each neuron and input
bias = np.zeros((8)) # the biases of each neuron    
output = np.dot(input,weights.T) + bias # the output using dot product
output_f = output_for(input,weights,bias) # the output using loops
output_norm = normalize_2d(output) # normalizing the output

print('in:',input)
print('w:',weights)
print('output:',output)
print('output_for:',output_f)
print('output_norm:',output_norm)
