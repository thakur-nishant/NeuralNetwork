import tensorflow as tf


class ThreeLayerNN:
    def __init__(self):
        self.alpha = 0.1  # "Alpha": (Learning rate)
        self.lmda = 0.01  # "Lambda": (Weight regularization)
        self.n_input_node = 2
        self.n_hidden_nodes = 100  # "Number of Nodes in Hidden Layer"
        self.sample_size = 200  # "Number of Samples"
        self.n_classes = 4  # "Number of Classes"
        self.epoch = 10
        self.n_features = 2  # 2 coordinated (x,y) of a point

        self.type_of_data = 's_curve'  # "Type of generated data"
        self.hidden_transfer_function = 'Relu'  # "Hidden Layer Transfer Function"

        self.X = tf.placeholder(tf.float32, [None, self.n_features], name='features')
        self.Y = tf.placeholder(tf.float32, name='labels')

        # now declare the weights connecting the input to the hidden layer
        self.W1 = tf.Variable(tf.random_normal([self.n_features, self.n_hidden_nodes], stddev=0.001), name='W1')
        self.b1 = tf.Variable(tf.random_normal([self.n_hidden_nodes]), name='b1')

        # and the weights connecting the hidden layer to the output layer
        self.W2 = tf.Variable(tf.random_normal([300, 10], stddev=0.03), name='W2')
        self.b2 = tf.Variable(tf.random_normal([10]), name='b2')

    def train(self):

        # calculate the output of the hidden layer
        hidden_out = tf.add(tf.matmul(self.X, self.W1), self.b1)
        if self.hidden_transfer_function == 'Relu':
            hidden_out = tf.nn.relu(hidden_out)
        else:
            hidden_out = tf.nn.sigmoid(hidden_out)

        # now calculate the hidden layer output
        op = tf.add(tf.matmul(hidden_out, self.W2), self.b2)

        # Regularization of weights
        regularizer = tf.contrib.layers.l2_regularizer(self.lmda)

        penalty = tf.contrib.layers.apply_regularization(regularizer, weights_list=[self.W1, self.W2])

        cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=op, labels=self.Y))
