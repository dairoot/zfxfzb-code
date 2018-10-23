#!/usr/bin/env python
# -*- coding: utf-8 -*-


import sys
import os
sys.path.append(os.getcwd())
from utils import TrainData, TestData


train_data = TrainData()
test_data = TestData()

import tensorflow as tf

x = tf.placeholder("float", [None, train_data.in_size])
W = tf.Variable(tf.zeros([train_data.in_size, train_data.out_size]))
b = tf.Variable(tf.zeros([train_data.out_size]))
y = tf.nn.softmax(tf.matmul(x, W) + b)
y_ = tf.placeholder("float", [None, train_data.out_size])
cross_entropy = -tf.reduce_sum(y_*tf.log(y))
train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

for i in range(1000):
    batch_xs, batch_ys = train_data.next_batch(100)
    sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})


correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

print(sess.run(accuracy, feed_dict={x: train_data.test_xs, y_: train_data.test_ys}))
