# coding: utf-8
import tensorflow as tf
import sys
import os
sys.path.append(os.getcwd())
from utils import TrainData
train_data = TrainData()


def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                          strides=[1, 2, 2, 1], padding='SAME')


sess = tf.InteractiveSession()


x = tf.placeholder(tf.float32, shape=[None, train_data.in_size])  # 16x21
y_ = tf.placeholder(tf.float32, shape=[None, train_data.out_size])

# 第一层卷积
W_conv1 = weight_variable([5, 5, 1, 32])
b_conv1 = bias_variable([32])
x_image = tf.reshape(x, [-1, 16, 21, 1])  # 16x21
h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)  # 16x21x32
h_pool1 = max_pool_2x2(h_conv1)  # 8x11x32

# 第二层卷积
W_conv2 = weight_variable([5, 5, 32, 64])
b_conv2 = bias_variable([64])

h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)  # 8x11x64
h_pool2 = max_pool_2x2(h_conv2)  # 4x6x64

# 密集连接层
W_fc1 = weight_variable([4 * 6 * 64, 1024])  # 4x6x64
b_fc1 = bias_variable([1024])

h_pool2_flat = tf.reshape(h_pool2, [-1, 4*6*64])  # 4x6x64
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)


# Dropout
keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

# 输出层
W_fc2 = weight_variable([1024, train_data.out_size])
b_fc2 = bias_variable([train_data.out_size])

y_conv = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)


# 训练和评估模型
cross_entropy = -tf.reduce_sum(y_*tf.log(y_conv))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
sess.run(tf.global_variables_initializer())


print('开始训练')
for i in range(1000):
    batch_xs, batch_ys = train_data.next_batch(50)
    if i % 100 == 0:
        train_accuracy = accuracy.eval(feed_dict={
            x: batch_xs, y_: batch_ys, keep_prob: 1.0})
        print "step %d, training accuracy %g" % (i, train_accuracy)
    train_step.run(feed_dict={x: batch_xs, y_: batch_ys, keep_prob: 0.5})

print "test accuracy %g" % accuracy.eval(feed_dict={
    x: train_data.test_xs, y_: train_data.test_ys, keep_prob: 1.0})
