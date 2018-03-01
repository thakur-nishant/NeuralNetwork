from neural_network import NeuralNetwork

def run():
    nn = NeuralNetwork(2, 2, 1)
    ip = [1,0]
    op = nn.feedforward(ip)

    print(op)


if __name__ == '__main__':
    run()