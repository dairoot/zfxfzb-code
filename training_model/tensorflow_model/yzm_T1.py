#!/usr/bin/env python
# -*- coding: utf-8 -*-


import tensorflow as tf
import sys
import os
sys.path.append(os.getcwd())
from utils import TrainData


zfdata = TrainData()

x = tf.placeholder("float", [None, zfdata.in_size])
W = tf.Variable(tf.zeros([zfdata.in_size, zfdata.out_size]))
b = tf.Variable(tf.zeros([zfdata.out_size]))
y = tf.nn.softmax(tf.matmul(x, W) + b)
y_ = tf.placeholder("float", [None, zfdata.out_size])
cross_entropy = -tf.reduce_sum(y_*tf.log(y))
train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

for i in range(200):
    batch_xs, batch_ys = zfdata.next_batch(100)
    sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})


correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
print(sess.run(accuracy, feed_dict={x: zfdata.test_xs, y_: zfdata.test_ys}))
