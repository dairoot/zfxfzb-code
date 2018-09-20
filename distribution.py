#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt  
import string


# 读取训练样本数据
data = np.loadtxt('data.dat', dtype='int')
y_data = data[:, 336]
x = [i for i in string.digits+string.ascii_lowercase]
y = [0]*36
for value in y_data:
    y[value] += 1

# 绘制数据分布图
y_pos = np.arange(36)
plt.bar(y_pos, y, align='center', alpha=0.5)
plt.xticks(y_pos, x)
plt.ylabel('count')
plt.show()

