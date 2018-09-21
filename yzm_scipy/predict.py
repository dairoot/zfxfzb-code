# coding: utf-8

import numpy as np
import sys
import os
sys.path.append(os.path.dirname(sys.path[0]))
from utils import get_img_data, num_turn_chr
from PIL import Image

all_theta = np.matrix(np.loadtxt('yzm_scipy/theta.dat'))


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


def verify(file_name):
    # 加载图片
    img = Image.open(file_name).convert("L")

    X = np.matrix(get_img_data(img))
    acc, pred = predictOneVsAll(all_theta, X)
    answers = map(lambda x: num_turn_chr(x), pred)
    return ''.join(answers)


if __name__ == "__main__":
    print verify('img/CheckCode.gif')
