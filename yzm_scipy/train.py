#!/usr/bin/env python
# -*- coding: utf-8 -*-
# coding: utf-8
import numpy as np
from scipy.optimize import fmin_bfgs
import sys
import os
sys.path.append(os.path.dirname(sys.path[0]))


def sigmoid(z):
    g = 1.0/(1.0+np.exp(-z))
    return g


def lrGD(theta, *args):
    theta = np.matrix(theta).transpose()
    X, y, the_lambda = args
    m = y.shape[0]
    grad = np.zeros(theta.shape)

    htheta = sigmoid(X * theta)
    grad[0] = 1.0/m * np.sum(np.array((htheta-y))*np.array(X[:, 0:1]))
    for i in range(1, theta.shape[0]):
        grad[i] = 1.0/m * np.sum(np.array((htheta-y)) *
                                 np.array(X[:, i:i+1])) + the_lambda/m*theta[i]
    return grad.flatten()


def lrCostFunction(theta, *args):
    theta = np.matrix(theta).transpose()
    X, y, the_lambda = args
    m = y.shape[0]
    htheta = sigmoid(X*theta)
    J = -np.sum(np.array(y)*np.array(np.log(htheta)) + np.array((1-y))
                * np.array(np.log(1-htheta)))/m + the_lambda*np.sum(
        np.array(theta[1:]) * np.array(theta[1:]))/(2*m)
    return J


def oneVsAll(X, y, num_labels, the_lambda):
    m, n = np.shape(X)
    all_theta = np.matrix(np.zeros((num_labels, n+1)))
    X = np.hstack((np.ones((m, 1)), X))
    for c in range(num_labels):
        print 'Training for %d/%d' % (c+1, num_labels)
        initial_theta = np.zeros((n+1, 1))
        args = (X, (y == c), the_lambda)
        theta = fmin_bfgs(lrCostFunction, initial_theta,
                          fprime=lrGD, args=args, maxiter=50)
        all_theta[c, :] = theta.transpose()

    return all_theta


def train():
    '''开始训练'''
    num_labels = 36
    the_lambda = 0.1
    data = np.matrix(np.loadtxt('data.dat'))
    y = data[:, 336]
    X = data[:, :336]
    all_theta = oneVsAll(X, y, num_labels, the_lambda)
    ''' 生成训练后的特征文本 '''
    np.savetxt('yzm_scipy/theta.dat', all_theta)


train()
