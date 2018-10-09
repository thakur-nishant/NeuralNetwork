# Thakur, Nishant
# 1001-544-591
# 2018-10-08
# Assignment-03-01
import os
import sys

import matplotlib
import numpy as np
import scipy.misc
from sklearn.model_selection import train_test_split

matplotlib.use('TkAgg')

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt

if sys.version_info[0] < 3:
    from Tkinter import *
else:
    from tkinter import *


def read_image_and_convert_to_vector(file_name):
    img = scipy.misc.imread(file_name).astype(np.float32)  # read image and convert to float
    return img.reshape(-1, 1)  # reshape to column vector and return it


class Window(Frame):
    def __init__(self, master=None):
        Frame.__init__(self, master)
        self.master = master
        self.init_window()

    # Creation of init_window
    def init_window(self):
        # initialization
        self.learning_rate = 0.1
        self.weights = (2 * np.random.rand(10, 785) - 1) * 0.001
        self.confusion_matrix = [[0 for i in range(10)] for j in range(10)]
        self.confusion_matrix_variable = StringVar()
        self.learning_rule = "Filtered Learning"
        self.transfer_function = "Symmetrical Hard limit"
        self.epoch_count = 0
        X, Y = self.convert_input_data()
        print(len(X), len(Y))
        self.X_train, self.X_test, self.Y_train, self.Y_test = train_test_split(X, Y, test_size=0.2)

        # changing the title of our master widget
        self.master.title("Hebbian Learning Rules on MNIST")

        self.left_frame = Frame(self.master)
        self.right_frame = Frame(self.master)
        self.conrols = Frame(self.master)

        self.left_frame.grid(row=0, column=0)
        self.right_frame.grid(row=0, column=1)
        self.conrols.grid(row=1, columnspan=2)

        self.messageVar = Message(self.right_frame, textvariable=self.confusion_matrix_variable)
        self.messageVar.config(bg='lightgreen')
        self.messageVar.pack()

        self.figure = plt.figure(figsize=(9,4))
        self.axes = self.figure.add_axes([0.15, 0.15, 0.70, 0.82])
        # self.axes = self.figure.add_axes()
        self.axes = self.figure.gca()
        self.axes.set_xlabel('Epoch(0-1000)')
        self.axes.set_ylabel('Accuracy(%)')
        self.plot = self.axes.scatter([], [], s=0.5)
        self.axes.set_title("")
        self.axes.set_aspect('auto')
        plt.xlim(0, 1000)
        plt.ylim(0, 100)
        self.canvas = FigureCanvasTkAgg(self.figure, master=self.left_frame)
        self.plot_widget = self.canvas.get_tk_widget()
        self.plot_widget.grid(row=0, column=0)

        self.learning_rate_slider = Scale(self.conrols, variable=DoubleVar(), orient=HORIZONTAL,
                                          from_=0.001, to_=1, resolution=0.001, bg="#DDDDDD",
                                          activebackground="#FF0000", highlightcolor="#00FFFF",
                                          label="Learning Rate(Alpha)",
                                          command=lambda event: self.learning_rate_slider_callback())
        self.learning_rate_slider.set(self.learning_rate)
        self.learning_rate_slider.bind("<ButtonRelease-1>", lambda event: self.learning_rate_slider_callback())
        self.learning_rate_slider.grid(row=0, column=0)

        self.adjust_weights = Button(self.conrols, text="Adjust Weights", command=self.train)
        self.adjust_weights.grid(row=0, column=1)

        self.randomize_weights = Button(self.conrols, text="Randomize Weights", command=self.randomize_weights)
        self.randomize_weights.grid(row=0, column=2)

        self.disp_confusion_matrix = Button(self.conrols, text="Display Confusion Matrix",
                                               command=self.display_confusion_matrix)
        self.disp_confusion_matrix.grid(row=0, column=3)

        self.label_for_learning_method = Label(self.conrols, text="Select Learning Method:",
                                               justify="center")
        self.label_for_learning_method.grid(row=0, column=4)
        self.learning_method_variable = StringVar()
        self.learning_method_dropdown = OptionMenu(self.conrols, self.learning_method_variable,
                                                   "Filtered Learning", "Delta Rule", "Unsupervised Hebb",
                                                   command=lambda
                                                       event: self.learning_method_dropdown_callback())
        self.learning_method_variable.set("Filtered Learning")
        self.learning_method_dropdown.grid(row=0, column=5)

        self.label_for_activation_function = Label(self.conrols, text="Transfer Functions:",
                                                   justify="center")
        self.label_for_activation_function.grid(row=0, column=6)
        self.transfer_function_variable = StringVar()
        self.transfer_function_dropdown = OptionMenu(self.conrols, self.transfer_function_variable,
                                                     "Symmetrical Hard limit", "Linear", "Hyperbolic Tangent",
                                                     command=lambda
                                                         event: self.transfer_function_dropdown_callback())
        self.transfer_function_variable.set("Symmetrical Hard limit")
        self.transfer_function_dropdown.grid(row=0, column=7)

    def learning_method_dropdown_callback(self):
        self.learning_rule = self.learning_method_variable.get()

    def transfer_function_dropdown_callback(self):
        self.transfer_function = self.transfer_function_variable.get()

    def learning_rate_slider_callback(self):
        self.learning_rate = np.float(self.learning_rate_slider.get())

    def select_learning_rule(self, input_vector, a, t):
        if self.learning_rule == "Filtered Learning":
            self.weights = (1 - self.learning_rate) * self.weights + self.learning_rate * np.dot(t, np.array(input_vector).T)
        elif self.learning_rule == "Delta Rule":
            self.weights = self.weights + self.learning_rate * (t - a) * input_vector.T
        else:
            self.weights = self.weights + self.learning_rate * a * np.array(input_vector).T

    def select_transfer_function(self, x):
        if self.transfer_function == "Symmetrical Hard limit":
            for i in range(len(x)):
                x[i] = 1 if x[i] > 0 else 0
        elif self.transfer_function == "Linear":
            pass
        else:
            x = np.tanh(x)
        return x

    def predict(self, input_vector):
        output_vector = np.dot(self.weights, input_vector)
        output_vector = self.select_transfer_function(output_vector)
        return output_vector

    def train(self):
        print("Training started")
        input_vectors = self.X_train
        targets = self.Y_train
        for j in range(100):
            for i in range(len(targets)):
                a = self.predict(input_vectors[i])
                # self.delta_rule(input_vectors[i], a, targets[i])
                target_vector = np.array([0 for k in range(10)])
                target_vector[targets[i]] = 1
                target_vector = target_vector.reshape(-1,1)
                self.select_learning_rule(input_vectors[i], a, target_vector)
            self.plot_accuracy()
        print(self.weights)
        self.confusion_matrix_fun()
        print("100 epochs complete")
        print("Confusion matrix: \n", self.confusion_matrix)
        count = 0
        for i in range(10):
            count += self.confusion_matrix[i][i]
        print("Accuracy:", count/len(self.Y_test))

    def plot_accuracy(self):
        self.confusion_matrix_fun()
        count = 0
        for i in range(10):
            count += self.confusion_matrix[i][i]
        acc = count/len(self.Y_test)*100
        point = [self.epoch_count, acc]
        array = self.plot.get_offsets()
        # add the points to the plot
        array = np.append(array, point)
        self.plot.set_offsets(array.reshape(-1,2))
        self.epoch_count += 1
        self.figure.canvas.draw()

    def display_confusion_matrix(self):
        res = " Confusion Matrix: \n\n"
        for i in self.confusion_matrix:
            res += str(i) + "\n"
        print(res)
        self.confusion_matrix_variable.set(res)

    def randomize_weights(self):
        self.weights = (2 * np.random.rand(10, 785) - 1) * 0.001
        print(self.weights)

    def confusion_matrix_fun(self):
        self.confusion_matrix = [[0 for i in range(10)] for j in range(10)]
        for t in range(len(self.Y_test)):
            a = self.predict(self.X_test[t])
            guess = np.argmax(a.T)
            # print(self.Y_test[t], guess, a)
            self.confusion_matrix[self.Y_test[t]][guess] += 1

    def read_one_image_and_convert_to_vector(self, file_name):
        img = scipy.misc.imread(file_name).astype(np.float32)  # read image and convert to float
        img = np.append(img, [1])  # add bias to input vector
        return img.reshape(-1, 1)  # reshape to column vector and return it

    def convert_input_data(self):
        path = './Data/'
        X = []
        Y = []
        for filename in os.listdir(path):
            # convert each image to a vector and normalize each vector element to be in the range of -1 to 1
            temp_vector = self.read_one_image_and_convert_to_vector(path + filename) / 127.5 - 1
            X.append(temp_vector)
            Y.append(int(filename[0]))
        return X, Y


if __name__ == "__main__":
    root = Tk()
    # size of the window
    root.geometry("1200x650")
    app = Window(root)
    root.mainloop()
