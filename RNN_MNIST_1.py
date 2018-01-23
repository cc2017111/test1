import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

# hyperparameters
lr = 0.001
training_iter = 100000
batch_size = 128
display_step = 10

n_inputs = 28  # image_size 28x28
n_step = 28    # time steps
n_hidden_units = 128  # neurons in hidden layer
n_classes = 10  # (0~9 digits)

# tf Graph input
x = tf.placeholder(tf.float32, [None, n_step, n_inputs])
y = tf.placeholder(tf.float32, [None, n_classes])

# define weights
weights = {
    # [28, 128]
    'in': tf.Variable(tf.random_normal([n_inputs, n_hidden_units])),
    # [128, 10]
    'out': tf.Variable(tf.random_normal([n_hidden_units, n_classes]))
}
biases = {
    # [128, ]
    'in': tf.Variable(tf.constant(0.1, tf.float32)),
    # [10, ]
    'out': tf.Variable(tf.constant(0.1, tf.float32))
}


def RNN(X, weights, biases):
    # hidden layer for input to cell
    # X [128 batch, 28 step, 28 inputs] --> X[128*28, 28 inputs]
    X = tf.reshape(X, [-1, n_inputs])
    X_in = tf.matmul(X, weights['in']) + biases['in']      # X_in [128 batch*28 step, 128 hidden_units]
    X_in = tf.reshape(X_in, [-1, n_step, n_hidden_units])  # X_in [128 batch, 28 step, 128 hidden_units]

    # cell
    lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(n_hidden_units, forget_bias=1.0, state_is_tuple=True)
    # lstm cell is divided into two parts (c_state(long-term memory), m_state(short-term memory)
    _init_state = lstm_cell.zero_state(batch_size, dtype=tf.float32)
    outputs, states = tf.nn.dynamic_rnn(lstm_cell, X_in, initial_state=_init_state, time_major=False)

    # hidden layer for output as final results
    results = tf.matmul(states[1], weights['out']) + biases['out']
    # or
    # unstack to list [(batch, outputs)..] * steps
    # outputs = tf.unstack(tf.transpose(outputs, [1, 0, 2]))  # state is the last outputs
    # results = tf.matmul(outputs[-1], weights['out']) + biases['out']
    return results

pred = RNN(x, weights, biases)
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
train_op = tf.train.AdamOptimizer(lr).minimize(cost)

correct_pred = tf.equal(tf.arg_max(pred, 1), tf.arg_max(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

init = tf.global_variables_initializer()
sess = tf.Session()

sess.run(init)
step = 0
while step*batch_size < training_iter:
    batch_xs, batch_ys = mnist.train.next_batch(batch_size)
    batch_xs = batch_xs.reshape([batch_size, n_step, n_inputs])
    sess.run([train_op], feed_dict={x: batch_xs, y: batch_ys})
    if step % display_step == 0:
        print(sess.run(accuracy, feed_dict={x: batch_xs, y: batch_ys}))
    step += 1

sess.close()  # accuracy = 0.9765
