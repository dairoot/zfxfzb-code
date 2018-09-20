# coding: utf-8

import numpy as np
import sys
import os
sys.path.append(os.path.dirname(sys.path[0]))
from utils import get_img_data
from PIL import Image


def sigmoid(z):
    g = 1.0/(1.0+np.exp(-z))
    return g


def predictOneVsAll(all_theta, X):
    m = X.shape[0]
    X = np.hstack((np.ones((m, 1)), X))
    real_all_theta = all_theta.transpose()
    all_predict = sigmoid(np.dot(X, real_all_theta))
    Accuracy = all_predict.max(1)
    p = np.argmax(all_predict, axis=1)
    return Accuracy, p


def verify():
    # 加载图片
    img = Image.open('img/CheckCode.gif').convert("L")
    all_theta = np.matrix(np.loadtxt('yzm_scipy/theta.dat'))
    X = np.matrix(get_img_data(img))
    acc, pred = predictOneVsAll(all_theta, X)
    answers = map(chr, map(lambda x: x + 48 if x <= 9 else x + 87, pred))
    return ''.join(answers)


if __name__ == "__main__":
    print verify()
