#!/usr/bin/env python
# -*- coding: utf-8 -*-
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cross_validation import cross_val_score
from sklearn.externals import joblib
import numpy as np
from PIL import Image
import sys
import os
sys.path.append(os.path.dirname(sys.path[0])) 
from utils import num_turn_chr, get_img_data


if __name__ == "__main__":
    # 读取训练样本数据
    data = np.loadtxt('data.dat')
    y_train = [num_turn_chr(int(i)) for i in data[:, 336]]
    X_train = data[:, :336]

    # 进行训练
    knn = KNeighborsClassifier(n_neighbors=1)
    knn.fit(X_train, y_train)

    # 导出训练函数
    # joblib.dump(knn, 'yzm_sklearn/knn.pkl')
    # knn = joblib.load('yzm_sklearn/knn.pkl')

    # 精度统计
    score = cross_val_score(knn, X_train, y_train, cv=5, scoring='accuracy')
    print score.mean()

    # 加载图片
    img = Image.open('img/CheckCode.gif').convert("L")

    # 识别验证码
    print knn.predict(get_img_data(img))
