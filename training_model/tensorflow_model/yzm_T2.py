#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import os
import tensorflow as tf
sys.path.append(os.getcwd())
from utils import TrainData, TestData


train_data = TrainData()
test_data = TestData()


def add_layer(inputs, in_size, out_size, activation_function=None,):
    # add one more layer and return the output of this layer
    Weights = tf.Variable(tf.random_normal([in_size, out_size]))
    biases = tf.Variable(tf.zeros([1, out_size]) + 0.1,)
    Wx_plus_b = tf.matmul(inputs, Weights) + biases
    if activation_function is None:
        outputs = Wx_plus_b
    else:
        outputs = activation_function(Wx_plus_b,)
    return outputs


def compute_accuracy(v_xs, v_ys):
    global prediction
    y_pre = sess.run(prediction, feed_dict={xs: v_xs})
    correct_prediction = tf.equal(tf.argmax(y_pre, 1), tf.argmax(v_ys, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    result = sess.run(accuracy, feed_dict={xs: v_xs, ys: v_ys})
    return result


# define placeholder for inputs to network
xs = tf.placeholder(tf.float32, [None, train_data.in_size])
ys = tf.placeholder(tf.float32, [None, train_data.out_size])

# add output layer
prediction = add_layer(xs, train_data.in_size, train_data.out_size,  activation_function=tf.nn.softmax)

# the error between prediction and real data
cross_entropy = tf.reduce_mean(-tf.reduce_sum(ys * tf.log(prediction),
                                              reduction_indices=[1]))       # loss
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

sess = tf.Session()
# important step
init = tf.global_variables_initializer()
sess.run(init)


for i in range(400):
    train_xs, train_ys = train_data.next_batch(100)
    test_xs, test_ys = test_data.next_batch(100)
    sess.run(train_step, feed_dict={xs: train_xs, ys: train_ys})
    if i % 50 == 0:
        print(compute_accuracy(test_xs, test_ys))
