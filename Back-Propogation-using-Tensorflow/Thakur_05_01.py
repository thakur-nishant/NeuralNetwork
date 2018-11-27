# Thakur, Nishant
# 1001-544-591
# 2018-11-26
# Assignment-05-01
import sys

import sklearn.datasets
from sklearn.cluster import AgglomerativeClustering
import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt
import tensorflow as tf
if sys.version_info[0] < 3:
    from Tkinter import *
else:
    from tkinter import *


def generate_data(dataset_name, n_samples, n_classes):
    if dataset_name == 'swiss_roll':
        data = sklearn.datasets.make_swiss_roll(n_samples, noise=1.5, random_state=99)[0]
        data = data[:, [0, 2]] / 10
    if dataset_name == 'moons':
        data = sklearn.datasets.make_moons(n_samples=n_samples, noise=0.15)[0] / 1.2
    if dataset_name == 'blobs':
        data = sklearn.datasets.make_blobs(n_samples=n_samples, centers=n_classes * 2, n_features=2,
                                           cluster_std=0.85 * np.sqrt(n_classes), random_state=100)
        return data[0] / 10., [i % n_classes for i in data[1]]
    if dataset_name == 's_curve':
        data = sklearn.datasets.make_s_curve(n_samples=n_samples, noise=0.15, random_state=100)[0]
        data = data[:, [0, 2]] / 2.0

    ward = AgglomerativeClustering(n_clusters=n_classes * 2, linkage='ward').fit(data)
    return data[:] + np.random.randn(*data.shape) * 0.03, [i % n_classes for i in ward.labels_]


class Window(Frame):
    def __init__(self, master=None):
        Frame.__init__(self, master)
        self.master = master
        self.init_window()

    # Creation of init_window
    def init_window(self):
        # initialization
        self.learning_rate = 0.1
        self.regularization_rate = 0.01
        self.epochs = 10
        self.epochs_count = 1
        self.batch_size = 100
        self.input_dimension = 2
        self.number_of_nodes_in_first_hidden_layer = 100
        self.number_of_classes = 4
        self.number_of_samples = 200

        self.type_of_generated_data = 's_curve'
        self.hidden_layer_transfer_function = 'Relu'

        # generate data
        self.X_data, Y, self.yy = self.get_data()
        self.Y_data = np.array(Y)
        # tensorflow part
        self.x = tf.placeholder(tf.float32, [None, self.input_dimension])
        self.one_hot_true_label = tf.placeholder(tf.float32, [None, self.number_of_classes])
        self.W1 = tf.Variable(tf.random_normal([self.input_dimension, self.number_of_nodes_in_first_hidden_layer], stddev=0.01),
                         name='W1')
        self.b1 = tf.Variable(tf.random_normal([self.number_of_nodes_in_first_hidden_layer]), name='b1')
        self.W2 = tf.Variable(tf.random_normal([self.number_of_nodes_in_first_hidden_layer, self.number_of_classes], stddev=0.01),
                         name='W2')
        self.b2 = tf.Variable(tf.random_normal([self.number_of_classes]), name='b2')

        init_operator = tf.global_variables_initializer()

        self.sess = tf.Session()
        self.sess.run(init_operator)

        # changing the title of our master widget
        self.master.title("TensorFlow Back propagation")

        self.left_frame = Frame(self.master)
        self.right_frame = Frame(self.master)
        self.conrols = Frame(self.master)

        self.left_frame.grid(row=0, column=0)
        self.right_frame.grid(row=0, column=1)
        self.conrols.grid(row=1, columnspan=2)

        self.figure = plt.figure(figsize=(5, 5))
        # self.axes = self.figure.add_axes([0.15, 0.15, 0.80, 0.80])
        self.axes = self.figure.add_axes()
        self.axes = self.figure.gca()
        self.axes.set_title("")
        self.axes.set_aspect('auto')
        plt.xlim(-2, 2)
        plt.ylim(-2, 2)
        self.axes.scatter(self.X_data[:, 0], self.X_data[:, 1], c=self.yy, cmap=plt.cm.Accent)
        self.canvas = FigureCanvasTkAgg(self.figure, master=self.left_frame)
        self.plot_widget = self.canvas.get_tk_widget()
        self.plot_widget.grid(row=0, column=0)

        self.learning_rate_slider = Scale(self.conrols, variable=DoubleVar(), orient=HORIZONTAL,
                                          from_=0.000, to_=1, resolution=0.001, bg="#DDDDDD",
                                          activebackground="#FF0000", highlightcolor="#00FFFF",
                                          label="Learning Rate(Alpha)",
                                          command=lambda event: self.learning_rate_slider_callback())
        self.learning_rate_slider.set(self.learning_rate)
        self.learning_rate_slider.bind("<ButtonRelease-1>", lambda event: self.learning_rate_slider_callback())
        self.learning_rate_slider.grid(row=0, column=0)

        self.weight_regularization_slider = Scale(self.conrols, variable=DoubleVar(), orient=HORIZONTAL,
                                                  from_=0, to_=1, resolution=0.01, bg="#DDDDDD",
                                                  activebackground="#FF0000", highlightcolor="#00FFFF",
                                                  label="Lambda",
                                                  command=lambda event: self.weight_regularization_slider_callback())
        self.weight_regularization_slider.set(self.regularization_rate)
        self.weight_regularization_slider.bind("<ButtonRelease-1>",
                                               lambda event: self.weight_regularization_slider_callback())
        self.weight_regularization_slider.grid(row=0, column=1)

        self.hidden_layer_nodes_count_slider = Scale(self.conrols, variable=DoubleVar(), orient=HORIZONTAL,
                                                     from_=1, to_=500, resolution=1, bg="#DDDDDD",
                                                     activebackground="#FF0000", highlightcolor="#00FFFF",
                                                     label="Num. nodes in Hidden Layer",
                                                     command=lambda
                                                         event: self.hidden_layer_nodes_count_slider_callback())
        self.hidden_layer_nodes_count_slider.set(self.number_of_nodes_in_first_hidden_layer)
        self.hidden_layer_nodes_count_slider.bind("<ButtonRelease-1>",
                                                  lambda event: self.hidden_layer_nodes_count_slider_callback())
        self.hidden_layer_nodes_count_slider.grid(row=0, column=2)

        self.number_of_samples_slider = Scale(self.conrols, variable=DoubleVar(), orient=HORIZONTAL,
                                                     from_=4, to_=1000, resolution=1, bg="#DDDDDD",
                                                     activebackground="#FF0000", highlightcolor="#00FFFF",
                                                     label="Num.of samples",
                                                     command=lambda
                                                         event: self.number_of_samples_slider_callback())
        self.number_of_samples_slider.set(self.number_of_samples)
        self.number_of_samples_slider.bind("<ButtonRelease-1>",
                                                  lambda event: self.number_of_samples_slider_callback())
        self.number_of_samples_slider.grid(row=0, column=3)

        self.number_of_classes_slider = Scale(self.conrols, variable=DoubleVar(), orient=HORIZONTAL,
                                              from_=0, to_=100, resolution=1, bg="#DDDDDD",
                                              activebackground="#FF0000", highlightcolor="#00FFFF",
                                              label="Num. of Classes",
                                              command=lambda
                                                  event: self.number_of_classes_slider_callback())
        self.number_of_classes_slider.set(self.number_of_classes)
        self.number_of_classes_slider.bind("<ButtonRelease-1>",
                                           lambda event: self.number_of_classes_slider_callback())
        self.number_of_classes_slider.grid(row=0, column=4)

        self.adjust_weights = Button(self.conrols, text="Adjust Weights(Train)", command=self.train)
        self.adjust_weights.grid(row=1, column=0)

        self.reset_weights = Button(self.conrols, text="Reset Weights", command=self.reset_weights_fun)
        self.reset_weights.grid(row=1, column=1)

        #########################################################################
        #  Set up the frame for drop down selection
        #########################################################################
        self.label_for_activation_function = Label(self.conrols, text="Hidden Layer Transfer Function",
                                                      justify="center")
        self.label_for_activation_function.grid(row=1, column=2)
        self.activation_function_variable = StringVar()
        self.activation_function_dropdown = OptionMenu(self.conrols, self.activation_function_variable,
                                                          "Sigmoid", "Relu",
                                                          command=lambda
                                                              event: self.activation_function_dropdown_callback())
        self.activation_function_variable.set(self.hidden_layer_transfer_function)
        self.activation_function_dropdown.grid(row=1, column=3)

        self.label_for_generate_data = Label(self.conrols, text="Hidden Layer Transfer Function",
                                                   justify="center")
        self.label_for_generate_data.grid(row=1, column=4)
        self.generate_data_variable = StringVar()
        self.generate_data_dropdown = OptionMenu(self.conrols, self.generate_data_variable,
                                                       "s_curve", "blobs", "swiss_roll", "moons",
                                                       command=lambda
                                                           event: self.generate_data_dropdown_callback())
        self.generate_data_variable.set(self.type_of_generated_data)
        self.generate_data_dropdown.grid(row=1, column=5)

    def reset_weights_fun(self):
        tf.reset_default_graph()
        self.sess.close()
        # tensorflow part
        self.x = tf.placeholder(tf.float32, [None, self.input_dimension])
        self.one_hot_true_label = tf.placeholder(tf.float32, [None, self.number_of_classes])
        self.W1 = tf.Variable(
            tf.random_normal([self.input_dimension, self.number_of_nodes_in_first_hidden_layer], stddev=0.01),
            name='W1')
        self.b1 = tf.Variable(tf.random_normal([self.number_of_nodes_in_first_hidden_layer]), name='b1')
        self.W2 = tf.Variable(
            tf.random_normal([self.number_of_nodes_in_first_hidden_layer, self.number_of_classes], stddev=0.01),
            name='W2')
        self.b2 = tf.Variable(tf.random_normal([self.number_of_classes]), name='b2')

        init_operator = tf.global_variables_initializer()

        self.sess = tf.Session()
        self.sess.run(init_operator)

    def activation_function_dropdown_callback(self):
        self.hidden_layer_transfer_function = self.activation_function_variable.get()

    def generate_data_dropdown_callback(self):
        self.type_of_generated_data = self.generate_data_variable.get()
        # generate data
        self.X_data, Y, self.yy = self.get_data()
        self.Y_data = np.array(Y)

        self.axes.scatter(self.X_data[:, 0], self.X_data[:, 1], c=self.yy, cmap=plt.cm.Accent)

        self.train()

    def learning_rate_slider_callback(self):
        self.learning_rate = np.float(self.learning_rate_slider.get())

    def weight_regularization_slider_callback(self):
        self.regularization_rate = np.float(self.weight_regularization_slider.get())

    def hidden_layer_nodes_count_slider_callback(self):
        self.number_of_nodes_in_first_hidden_layer = np.int(self.hidden_layer_nodes_count_slider.get())

        tf.reset_default_graph()
        self.sess.close()
        # tensorflow part
        self.x = tf.placeholder(tf.float32, [None, self.input_dimension])
        self.one_hot_true_label = tf.placeholder(tf.float32, [None, self.number_of_classes])
        self.W1 = tf.Variable(
            tf.random_normal([self.input_dimension, self.number_of_nodes_in_first_hidden_layer], stddev=0.01),
            name='W1')
        self.b1 = tf.Variable(tf.random_normal([self.number_of_nodes_in_first_hidden_layer]), name='b1')
        self.W2 = tf.Variable(
            tf.random_normal([self.number_of_nodes_in_first_hidden_layer, self.number_of_classes], stddev=0.01),
            name='W2')
        self.b2 = tf.Variable(tf.random_normal([self.number_of_classes]), name='b2')

        init_operator = tf.global_variables_initializer()

        self.sess = tf.Session()
        self.sess.run(init_operator)

    def number_of_samples_slider_callback(self):
        self.number_of_samples = np.int(self.number_of_samples_slider.get())
        # generate data
        self.X_data, Y, self.yy = self.get_data()
        self.Y_data = np.array(Y)
        tf.reset_default_graph()
        self.sess.close()
        # tensorflow part
        self.x = tf.placeholder(tf.float32, [None, self.input_dimension])
        self.one_hot_true_label = tf.placeholder(tf.float32, [None, self.number_of_classes])
        self.W1 = tf.Variable(
            tf.random_normal([self.input_dimension, self.number_of_nodes_in_first_hidden_layer], stddev=0.01),
            name='W1')
        self.b1 = tf.Variable(tf.random_normal([self.number_of_nodes_in_first_hidden_layer]), name='b1')
        self.W2 = tf.Variable(
            tf.random_normal([self.number_of_nodes_in_first_hidden_layer, self.number_of_classes], stddev=0.01),
            name='W2')
        self.b2 = tf.Variable(tf.random_normal([self.number_of_classes]), name='b2')

        init_operator = tf.global_variables_initializer()

        self.sess = tf.Session()
        self.sess.run(init_operator)

    def number_of_classes_slider_callback(self):
        self.number_of_classes = np.int(self.number_of_classes_slider.get())
        # generate data
        self.X_data, Y, self.yy = self.get_data()
        self.Y_data = np.array(Y)
        tf.reset_default_graph()
        self.sess.close()
        # tensorflow part
        self.x = tf.placeholder(tf.float32, [None, self.input_dimension])
        self.one_hot_true_label = tf.placeholder(tf.float32, [None, self.number_of_classes])
        self.W1 = tf.Variable(
            tf.random_normal([self.input_dimension, self.number_of_nodes_in_first_hidden_layer], stddev=0.01),
            name='W1')
        self.b1 = tf.Variable(tf.random_normal([self.number_of_nodes_in_first_hidden_layer]), name='b1')
        self.W2 = tf.Variable(
            tf.random_normal([self.number_of_nodes_in_first_hidden_layer, self.number_of_classes], stddev=0.01),
            name='W2')
        self.b2 = tf.Variable(tf.random_normal([self.number_of_classes]), name='b2')

        init_operator = tf.global_variables_initializer()

        self.sess = tf.Session()
        self.sess.run(init_operator)

    def get_data(self):
        X, y = generate_data(self.type_of_generated_data, self.number_of_samples, self.number_of_classes)
        Y = []
        for idx in y:
            row = np.zeros(self.number_of_classes)
            row[idx] = 1
            Y.append(row)
        return X, Y, y

    def nn_model(self):
        # regularizer = tf.contrib.layers.l2_regularizer(self.regularization_rate)
        # first hidden layer
        first_hidden_layer_net = tf.add(tf.matmul(self.x, self.W1), self.b1)
        if self.hidden_layer_transfer_function == "Relu":
            first_hidden_layer_out = tf.nn.relu(first_hidden_layer_net)
        else:
            first_hidden_layer_out = tf.nn.sigmoid(first_hidden_layer_net)

        # output layer
        output_layer_net = tf.add(tf.matmul(first_hidden_layer_out, self.W2), self.b2)
        # Calculate class probabilities by using softmax
        output_predicted_probabilities = tf.nn.softmax(output_layer_net)
        # tf.contrib.layers.apply_regularization(regularizer, weights_list=[self.W1, self.W2])
        return output_predicted_probabilities

    def train(self):
        batch_x = self.X_data
        batch_y = self.Y_data

        output_predicted_probabilities = self.nn_model()
        predicted = tf.argmax(output_predicted_probabilities, 1)
        # calculate cross-entropy
        ouput_clipped = tf.clip_by_value(output_predicted_probabilities, 1e-10, 0.9999999)
        cross_entropy_loss = -tf.reduce_mean(tf.reduce_sum(self.one_hot_true_label * tf.log(ouput_clipped)
                                                                + (1 - self.one_hot_true_label) * tf.log(
            1 - ouput_clipped),
                                                                axis=1))
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=self.learning_rate).minimize(
            cross_entropy_loss)

        correct_prediction = tf.equal(tf.argmax(self.one_hot_true_label, 1), tf.argmax(output_predicted_probabilities, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        for epoch in range(self.epochs):
            _, loss = self.sess.run([optimizer, cross_entropy_loss],
                               feed_dict={self.x: batch_x, self.one_hot_true_label: batch_y})
            print("Epoch", self.epochs_count, "loss:", loss)
            self.epochs_count += 1
        resolution = 100
        xs = np.linspace(-2., 2., resolution)
        ys = np.linspace(-2., 2., resolution)
        xx, yy = np.meshgrid(xs, ys)
        xx = xx.flatten()
        yy = yy.flatten()
        batch_x = np.vstack((xx,yy)).T
        # batch_y = np.ones(len(batch_x))
        prediction = self.sess.run([predicted],
                      feed_dict={self.x: batch_x})
        print(prediction)
        print(np.unique(prediction))
        prediction = np.reshape(prediction,(100,100))
        # self.plot_graph()
        quad = self.axes.pcolormesh(xs, ys, prediction)
        self.axes.scatter(self.X_data[:, 0], self.X_data[:, 1], c=self.yy, cmap=plt.cm.Accent)
        # self.axes.suptitle(self.type_of_generated_data, fontsize=20)

        self.figure.canvas.draw()


if __name__ == "__main__":
    root = Tk()
    # size of the window
    root.geometry("1200x650")
    app = Window(root)
    root.mainloop()

