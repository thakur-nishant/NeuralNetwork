import math
import numpy as np

class NeuralNetwork:

    def __init__(self, input_nodes, hidden_nodes, output_nodes):
        self.input_nodes = input_nodes
        self.hidden_nodes = hidden_nodes
        self.output_nodes = output_nodes

        self.weights_IH = np.random.rand(self.hidden_nodes, self.input_nodes)
        self.weights_HO = np.random.rand(self.output_nodes, self.hidden_nodes)

        self.bias_IH = np.random.rand(self.hidden_nodes, 1)
        self.bias_HO = np.random.rand(self.output_nodes, 1)


    def sigmoid(self, x):
        return 1 / (1 + math.exp(-x))


    def feedforward(self,input_list):

        input_matrix = np.reshape(np.array(input_list),(-1,1))

        # Generating the hidden outputs
        hidden_matrix = np.matmul(self.weights_IH, input_matrix)

        # hidden_matrix = np.add(hidden_matrix, self.bias_IH)
        hidden_matrix = hidden_matrix + self.bias_IH


        # Activation function i.e. sigmoid function in this case
        sigmoid_fun = np.vectorize(self.sigmoid)
        hidden_matrix = sigmoid_fun(hidden_matrix)

        # print("Hidden:",hidden_matrix)
        # Generate the output
        output_matrix = np.matmul(self.weights_HO, hidden_matrix)

        output_matrix = np.add(output_matrix, self.bias_HO)
        # print("Output:", output_matrix)

        #Activation function
        output_matrix = sigmoid_fun(output_matrix)

        return output_matrix

