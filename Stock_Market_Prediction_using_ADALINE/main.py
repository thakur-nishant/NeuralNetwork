import os
import matplotlib.pyplot as plt
import numpy as np

import Thakur_04_02 as dh


class NN:
    def __init__(self):
        filename = "data_set_2.csv"
        data = self.read_normalize_data(filename)

        self.learning_rate = 0.1
        self.delay = 10
        self.train_test_split = 80
        self.stride = 1
        self.epoch = 10
        self.weights = np.zeros(2 * (self.delay + 1) + 1)
        self.train_data, self.test_data = self.data_split(data)
        self.figure, self.axes_array = plt.subplots(1, 2)
        self.figure.set_size_inches(5, 5)
        self.axes = self.figure.gca()

    def read_normalize_data(self, filename):
        res = dh.load_and_normalize_data(filename)
        return res

    def data_split(self, data):
        len_data = len(data)
        len_train = int(len_data * self.train_test_split / 100)
        return data[:len_train], data[len_train:]

    def train_direct(self):
        mse = []
        mae = []
        for _ in range(self.epoch):
            total = 0
            for i in range(0, len(self.train_data) - (self.delay + 2), self.stride):
                total += 1
                input_vector = self.train_data[i:i + (self.delay + 1)].T.flatten().tolist()
                input_vector.append(1)
                t = self.train_data[i + (self.delay + 2)][0]
                z = np.array(input_vector).reshape(-1,1)
                if i == 0:
                    h = t * z
                    R = z*z.T
                else:
                    h += t * z
                    R += z * z.T
            R = R / total
            h = h / total

            self.weights = np.matmul(np.linalg.inv(R), h)

            mse_t, mae_t = self.test()
            mse.append(mse_t)
            mae.append(mae_t)

        x_axis = [i for i in range(self.epoch)]
        self.plot_graph(x_axis, mse, mae, self.axes_array)

    def train(self):

        mse = []
        mae = []
        for _ in range(self.epoch):
            for i in range(len(self.train_data) - (self.delay + 2)):
                input_vector = self.train_data[i:i + (self.delay + 1)].T.flatten().tolist()
                input_vector.append(1)
                target = self.train_data[i + (self.delay + 2)][0]
                output = np.dot(self.weights.T, input_vector)
                error = target - output

                self.weights = self.weights + 2 * self.learning_rate * error * np.array(input_vector)

            mse_t, mae_t = self.test()
            mse.append(mse_t)
            mae.append(mae_t)

        x_axis = [i for i in range(self.epoch)]
        self.plot_graph(x_axis, mse, mae, self.axes_array)

    def test(self):
        mse = 0
        mae = 0
        for i in range(len(self.test_data) - (self.delay + 2)):
            input_vector = self.test_data[i:i + (self.delay + 1)].T.flatten().tolist()
            input_vector.append(1)

            target = self.train_data[i + (self.delay + 2)][0]
            output = np.dot(self.weights.T, input_vector)
            error = target - output
            mse += error ** 2
            mae += abs(error)

        return mse / len(self.test_data), mae / len(self.test_data)

    def plot_graph(self,x_axis, mse_price, mae_price, axes_array):
        print(mse_price,mae_price)
        axes_array[0].plot(x_axis, mse_price)
        axes_array[0].set_title("MSE for Price")
        axes_array[1].plot(x_axis, mae_price)
        axes_array[1].set_title("MAE for Price")
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    test = NN()
    test.train()
    # test.train_direct()