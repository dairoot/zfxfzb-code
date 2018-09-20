#!/usr/bin/env python
# -*- coding: utf-8 -*-
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
from sklearn.externals import joblib
import numpy as np

import sys
import os
sys.path.append(os.path.dirname(sys.path[0]))
from utils import num_turn_chr

# 读取训练样本数据
data = np.loadtxt('data.dat')
y_train = [num_turn_chr(int(i)) for i in data[:, 336]]
X_train = data[:, :336]

# 进行训练
knn = KNeighborsClassifier(n_neighbors=1)
knn.fit(X_train, y_train)

# 导出训练函数
joblib.dump(knn, 'yzm_sklearn/knn.pkl')


if __name__ == "__main__":
    # 精度统计
    score = cross_val_score(knn, X_train, y_train, cv=5, scoring='accuracy')
    print score.mean()
