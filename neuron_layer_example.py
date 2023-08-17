import numpy as np

# x1,x2 -- w1,w2 -> N + b --> Output
# Output = n(x1 * w1 + x2 * w2 + b)

# this model represents a single neuron layer with 8 neurons which receives 5 inputs
# each neuron has a bias and a weight for each input
# 
# every parameter and input are being randomly generated and
# we're testing a batch of four input-sets  

input = np.array([np.random.sample(5) for i in range(4)]) # a batch of 4 input-sets
weights = np.array([np.random.sample(5) for i in range(8)]) # the weights for each connection for each neuron and input
bias = np.zeros((8))
print("in:",input)
print("w:",weights)

outputb = []
for input_n in input:
    out_n = []
    for neuron in range(len(weights)):
        out = sum(input_n*weights[neuron])
        out_n.append(out)
    outputb.append(out_n)

outputb = np.array(outputb)
print("outputb:",outputb)

output = np.dot(input,weights.T) + bias
norm = np.linalg.norm(output)
output_norm = output/norm
print("output:",output)
print("output_norm:",output_norm)