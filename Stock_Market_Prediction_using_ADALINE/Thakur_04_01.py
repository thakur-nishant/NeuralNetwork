# Thakur, Nishant
# 1001-544-591
# 2018-10-29
# Assignment-04-01
import os
import sys

import matplotlib
import numpy as np
import scipy.misc
from sklearn.model_selection import train_test_split

matplotlib.use('TkAgg')

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt

import Thakur_04_02 as dh

if sys.version_info[0] < 3:
    from Tkinter import *
else:
    from tkinter import *


class Window(Frame):
    def __init__(self, master=None):
        Frame.__init__(self, master)
        self.master = master
        self.init_window()

    # Creation of init_window
    def init_window(self):
        # initialization
        # filename = "data_set_2.csv"
        filename = "stock_data.csv"
        data = self.read_normalize_data(filename)

        self.learning_rate = 0.1
        self.delay = 10
        self.train_test_split = 80
        self.epoch_count = 0
        self.epoch = 10
        self.stride = 1
        self.weights = np.zeros(2 * (int(self.delay) + 1) + 1)
        self.train_data, self.test_data = self.data_split(data)
        self.mse = []
        self.mae = []

        # changing the title of our master widget
        self.master.title("Widrow-Huff learning and adaptive filters")

        self.left_frame = Frame(self.master)
        self.conrols = Frame(self.master)

        self.left_frame.grid(row=0, column=0)
        self.conrols.grid(row=1, columnspan=2)

        self.figure = plt.figure(figsize=(12, 5))
        self.mse_f = self.figure.add_subplot(211)
        # self.mse_f.axis([0, 100, 0, 2])
        self.mse_f.set_title("MSE for Price")

        self.mae_f = self.figure.add_subplot(212)
        # self.mae_f.axis([0, 100, 0, 2])
        self.mae_f.set_title("MAE for Price")

        self.canvas = FigureCanvasTkAgg(self.figure, master=self.left_frame)
        self.plot_widget = self.canvas.get_tk_widget()
        self.plot_widget.grid(row=0, column=0)

        self.delay_slider = Scale(self.conrols, variable=DoubleVar(), orient=HORIZONTAL,
                                  from_=0, to_=100, resolution=1, bg="#DDDDDD",
                                  activebackground="#FF0000", highlightcolor="#00FFFF",
                                  label="Number of Delayed Elements",
                                  command=lambda event: self.delay_slider_callback())
        self.delay_slider.set(self.delay)
        self.delay_slider.bind("<ButtonRelease-1>", lambda event: self.delay_slider_callback())
        self.delay_slider.grid(row=0, column=0)

        self.learning_rate_slider = Scale(self.conrols, variable=DoubleVar(), orient=HORIZONTAL,
                                          from_=0.001, to_=1, resolution=0.001, bg="#DDDDDD",
                                          activebackground="#FF0000", highlightcolor="#00FFFF",
                                          label="Learning Rate(Alpha)",
                                          command=lambda event: self.learning_rate_slider_callback())
        self.learning_rate_slider.set(self.learning_rate)
        self.learning_rate_slider.bind("<ButtonRelease-1>", lambda event: self.learning_rate_slider_callback())
        self.learning_rate_slider.grid(row=0, column=1)

        self.data_split_slider = Scale(self.conrols, variable=DoubleVar(), orient=HORIZONTAL,
                                       from_=0, to_=100, resolution=1, bg="#DDDDDD",
                                       activebackground="#FF0000", highlightcolor="#00FFFF",
                                       label="Training Sample Size(%)",
                                       command=lambda event: self.data_split_slider_callback())
        self.data_split_slider.set(self.train_test_split)
        self.data_split_slider.bind("<ButtonRelease-1>", lambda event: self.data_split_slider_callback())
        self.data_split_slider.grid(row=0, column=2)

        self.stride_slider = Scale(self.conrols, variable=DoubleVar(), orient=HORIZONTAL,
                                   from_=1, to_=100, resolution=1, bg="#DDDDDD",
                                   activebackground="#FF0000", highlightcolor="#00FFFF",
                                   label="Stride",
                                   command=lambda event: self.stride_slider_callback())
        self.stride_slider.set(self.stride)
        self.stride_slider.bind("<ButtonRelease-1>", lambda event: self.stride_slider_callback())
        self.stride_slider.grid(row=0, column=3)

        self.epoch_slider = Scale(self.conrols, variable=DoubleVar(), orient=HORIZONTAL,
                                  from_=1, to_=100, resolution=1, bg="#DDDDDD",
                                  activebackground="#FF0000", highlightcolor="#00FFFF",
                                  label="Number of Iterations",
                                  command=lambda event: self.epoch_slider_callback())
        self.epoch_slider.set(self.epoch)
        self.epoch_slider.bind("<ButtonRelease-1>", lambda event: self.epoch_slider_callback())
        self.epoch_slider.grid(row=0, column=4)

        self.adjust_weights = Button(self.conrols, text="Set Weights to Zero", command=self.reset_weights)
        self.adjust_weights.grid(row=1, column=0)

        self.adjust_weights = Button(self.conrols, text="Adjust Weights (LMS)", command=self.train)
        self.adjust_weights.grid(row=1, column=1)

        self.randomize_weights = Button(self.conrols, text="Adjust Weights (Direct)", command=self.train_direct)
        self.randomize_weights.grid(row=1, column=2)

    def learning_rate_slider_callback(self):
        self.learning_rate = np.float(self.learning_rate_slider.get())

    def epoch_slider_callback(self):
        self.epoch = np.int(self.epoch_slider.get())

    def stride_slider_callback(self):
        self.stride = np.int(self.stride_slider.get())

    def delay_slider_callback(self):
        self.reset_weights()
        self.delay = np.int(self.delay_slider.get())

    def data_split_slider_callback(self):
        self.train_test_split = np.int(self.data_split_slider.get())

    def read_normalize_data(self, filename):
        res = dh.load_and_normalize_data(filename)
        return res

    def data_split(self, data):
        len_data = len(data)
        len_train = int(len_data * self.train_test_split / 100)
        return data[:len_train], data[len_train:]

    def reset_weights(self):
        self.weights = np.zeros(2 * (int(self.delay) + 1) + 1)
        print(self.weights)

    def train_direct(self):
        self.epoch_count += self.epoch
        for _ in range(int(self.epoch)):
            total = 0
            for i in range(0, len(self.train_data) - (int(self.delay) + 2), int(self.stride)):
                total += 1
                input_vector = self.train_data[i:i + (int(self.delay) + 1)].T.flatten().tolist()
                input_vector.append(1)
                t = self.train_data[i + (int(self.delay) + 2)][0]
                z = np.array(input_vector).reshape(-1, 1)
                if i == 0:
                    h = t * z
                    R = z * z.T
                else:
                    h += t * z
                    R += z * z.T
            R = R / total
            h = h / total

            self.weights = np.matmul(np.linalg.inv(R), h).T
            mse_t, mae_t = self.test()
            self.mse.append(mse_t[0])
            self.mae.append(mae_t[0])
        self.plot_graph()

    def train(self):
        self.epoch_count += self.epoch
        for _ in range(int(self.epoch)):
            for i in range(len(self.train_data) - (int(self.delay) + 2)):
                input_vector = np.append(self.train_data[i:i + (int(self.delay) + 1)].T.flatten().tolist(), [1]).reshape(-1,1)

                target = self.train_data[i + (int(self.delay) + 2)][0]
                output = np.matmul(self.weights, input_vector)
                error = target - output
                self.weights = self.weights + 2 * self.learning_rate * error * input_vector.T

            mse_t, mae_t = self.test()
            self.mse.append(mse_t[0][0])
            self.mae.append(mae_t[0][0])
        self.plot_graph()

    def test(self):
        mse = 0
        mae = 0
        for i in range(len(self.test_data)-(int(self.delay) + 2)):
            input_vector = np.append(self.train_data[i:i+(int(self.delay)+1)].T.flatten().tolist(), [1]).reshape(-1, 1)
            target = self.train_data[i+(int(self.delay)+2)][0]
            output = np.matmul(self.weights, input_vector)
            error = target - output
            mse += error ** 2
            mae += abs(error)

        return mse / len(self.test_data), mae / len(self.test_data)

    def plot_graph(self):
        x_axis = [i for i in range(int(self.epoch_count))]
        print("############## epoch count:", self.epoch_count )
        print(len(self.mae),self.mse)
        print(len(self.mae),self.mae)
        self.mse_f.plot(x_axis, self.mse)
        self.mae_f.plot(x_axis, self.mae)

        self.figure.canvas.draw()
        # self.mse_f.plot(x_axis, mse_price)
        # self.mae_f.plot(x_axis, mae_price)
        # plt.tight_layout()
        # plt.show()


if __name__ == "__main__":
    root = Tk()
    # size of the window
    root.geometry("1200x650")
    app = Window(root)
    root.mainloop()
