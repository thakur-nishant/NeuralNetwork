import os

import numpy as np
import scipy.misc
from sklearn.model_selection import train_test_split

class NN:
    def __init__(self):
        self.learning_rate = 0.1
        self.weights = (2*np.random.rand(10, 785) - 1) * 0.001
        self.confusion_matrix = [[0 for i in range(10)] for j in range(10)]
        X, Y = self.convert_input_data()
        print(len(X), len(Y))
        self.X_train, self.X_test, self.Y_train, self.Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

    def read_one_image_and_convert_to_vector(self,file_name):
        img = scipy.misc.imread(file_name).astype(np.float32) # read image and convert to float
        img = np.append(img, [1]) # add bias to input vector
        return img.reshape(-1,1) # reshape to column vector and return it

    def convert_input_data(self):
        path = './Data/'
        X = []
        Y = []
        for filename in os.listdir(path):
            # convert each image to a vector and normalize each vector element to be in the range of -1 to 1
            temp_vector = self.read_one_image_and_convert_to_vector(path+filename) / 127.5 - 1
            X.append(temp_vector)
            Y.append(int(filename[0]))
        return X, Y

    def randomize_weights(self):
        self.weights = (2*np.random.rand(10, 785) - 1) * 0.001

    def hyperbolic_tangent(self, x):
        return np.tanh(x)

    def linear(self,x):
        return x

    def symmetrical_hard_limit(self, x):
        for i in range(len(x)):
            x[i] = 1 if x[i] > 0 else 0
        return x

    def train_test_data_split(self, X, Y):
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
        print(Y_train)
        print(Y_test)

    def predict(self, input_vector):
        output_vector = np.dot(self.weights, input_vector)
        # print(output_vector)
        # output_vector = self.symmetrical_hard_limit(output_vector)
        output_vector = self.hyperbolic_tangent(output_vector)

        return output_vector

    def delta_rule(self, input_vector, a, t):
        self.weights = self.weights + self.learning_rate * (t-a) * input_vector.T

    def smoothing(self, input_vector, t):
        self.weights = (1 - self.learning_rate) * self.weights + self.learning_rate * t * np.array(input_vector).T

    def unsupervised_hebb(self, input_vector, a):
        self.weights = self.weights + self.learning_rate * a * np.array(input_vector).T

    def confusion_matrix_and_accuracy(self):
        self.confusion_matrix = [[0 for i in range(10)] for j in range(10)]
        count = 0
        for t in range(len(self.Y_test)):
            a = self.predict(self.X_test[t])
            guess = np.argmax(a.T)
            # print(self.Y_test[t], guess, a)
            self.confusion_matrix[self.Y_test[t]][guess] += 1
            if guess == self.Y_test[t]:
                count += 1
        return count/len(self.Y_test)*100

    def train(self):
        input_vectors = self.X_train
        targets = self.Y_train
        print(self.weights)
        for j in range(1000):
            for i in range(len(targets)):
                # a = self.predict(input_vectors[i])
                # self.delta_rule(input_vectors[i], a, targets[i])
                self.smoothing(input_vectors[i], targets[i])
            print("Accuracy(", (j+1) ,"):", self.confusion_matrix_and_accuracy())
        print(self.weights)
        print("Confusion matrix: \n", self.confusion_matrix)


if __name__ == '__main__':
    nn = NN()
    print("Confusion matrix: \n", nn.confusion_matrix)
    nn.train()


