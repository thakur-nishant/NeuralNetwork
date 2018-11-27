import tensorflow as tf

learning_rate = 0.5
epochs = 10
batch_size = 100
input_dimension = 2
number_of_nodes_in_first_hidden_layer = 100
number_of_classes = 4
x = tf.placeholder(tf.float32, [None, input_dimension])
one_hot_true_label = tf.placeholder(tf.float32, [None, number_of_classes])
W1 = tf.Variable(tf.random_normal([input_dimension, number_of_nodes_in_first_hidden_layer], stddev=0.01), name='W1')
b1 = tf.Variable(tf.random_normal([number_of_nodes_in_first_hidden_layer]), name='b1')
W2 = tf.Variable(tf.random_normal([number_of_nodes_in_first_hidden_layer, number_of_classes], stddev=0.01), name='W2')
b2 = tf.Variable(tf.random_normal([number_of_classes]), name='b2')

# first hidden layer
first_hidden_layer_net = tf.add(tf.matmul(x, W1), b1)
first_hidden_layer_out = tf.nn.relu(first_hidden_layer_net)

# output layer
output_layer_net = tf.add(tf.matmul(first_hidden_layer_out, W2), b2)
# Calculate class probabilities by using softmax
output_predicted_probabilities = tf.nn.softmax(output_layer_net)

# calculate cross-entropy
ouput_clipped = tf.clip_by_value(output_predicted_probabilities, 1e-10, 0.9999999)
cross_entropy_loss = -tf.reduce_mean(tf.reduce_sum(one_hot_true_label * tf.log(ouput_clipped)
                         + (1 - one_hot_true_label) * tf.log(1 - ouput_clipped), axis=1))

# optimizer for gradient descent
optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(cross_entropy_loss)

# initialize operators
init_operator = tf.global_variables_initializer()

# measure accuracy
correct_prediction = tf.equal(tf.argmax(one_hot_true_label, 1), tf.argmax(output_predicted_probabilities, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# train
# start the session
with tf.Session() as sess:
    sess.run(init_operator)
    total_batch = int(len(mnist.train.labels) / batch_size)
    for epoch in range(epochs):
        avg_loss = 0
        for i in range(total_batch):
            batch_x, batch_y = mnist.train.next_batch(batch_size=batch_size)
            merge = tf.summary.merge_all()
            _, loss = sess.run([optimizer, cross_entropy_loss],
                               feed_dict={x: batch_x, one_hot_true_label: batch_y})
            avg_loss += loss / total_batch
            # print("Batch #",i,  "Avg. Loss : ", avg_loss)

        print("Epoch #", (epoch + 1), "Loss =", "{:.3f}".format(avg_loss))
    accuracy = sess.run(accuracy, feed_dict={x: mnist.test.images, one_hot_true_label: mnist.test.labels})
    print("Accuracy: ", accuracy)
