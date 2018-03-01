import random
from neural_network import NeuralNetwork

def run():
    # create a neural network with 2 input neuron, 2 hidden neuron and 1 output
    nn = NeuralNetwork(2, 2, 1)

    training_data = [{
        'inputs' : [0, 1],
        'target' : [1]
    },
        {
        'inputs' : [1, 0],
        'target' : [1]
    },
        {
        'inputs' : [0, 0],
        'target' : [0]
    },
        {
        'inputs' : [1, 1],
        'target' : [0]
    }]

    for i in range(50000):
        data = random.choice(training_data)
        nn.train(data['inputs'], data['target'])

    print("Input: [0,0] | Output:",nn.feedforward([0,0])[1])
    print("Input: [0,1] | Output:",nn.feedforward([0,1])[1])
    print("Input: [1,0] | Output:",nn.feedforward([1,0])[1])
    print("Input: [1,1] | Output:",nn.feedforward([1,1])[1])


if __name__ == '__main__':
    run()