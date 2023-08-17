import numpy as np

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

def generate_random_input(size,count):
    '''
    Generate a random input batch with "size" inputs with "count" values each
    '''
    return np.random.sample((size,count))

class NeuronLayer():
    def __init__(self,input_batch,neuron_count,last=False,labels = []):

        self.input_batch = input_batch
        
        self.neuron_count = neuron_count
        
        self.last = last

        self.labels = labels
        

    def randomize_parameters(self):
        self.weights = 10 * np.random.sample((self.neuron_count,len(self.input_batch[0]))) - 5 
        self.bias = np.zeros((self.neuron_count))

    def output_generate(self):
        self.output = np.dot(self.input_batch,self.weights.T) + self.bias
        self.output_norm = normalize_2d(self.output)
        if self.last:
            self.output_labeled = []
            for i in self.output_norm:
                current_output = {}
                for j,label in zip(i,self.labels):
                    current_output[label] = j
                self.output_labeled.append(current_output)
            return self.output_labeled    
        return self.output_norm
    

input = generate_random_input(16,10)
hidden1 = NeuronLayer(input,20)
hidden1.randomize_parameters()
hidden2 = NeuronLayer(hidden1.output_generate(),10)
hidden2.randomize_parameters()
output_layer = NeuronLayer(hidden2.output_generate(),10,last=True,labels = ['0','1','2','3','4','5','6','7','8','9'])
output_layer.randomize_parameters()

print('in:',input)
print('hidden1:',hidden1.output_generate())
print('hidden2:',hidden2.output_generate())
print('output:',output_layer.output_generate())