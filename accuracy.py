#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import random

from yzm_scipy.predict import verify as scipy_verify
from yzm_sklearn.predict import verify as sklearn_verify

folder = 'zfxfzb-code-data/img/'
imgs_folder = folder+'test_img/'

img_files = os.listdir(imgs_folder)
num = 1000
test_img_files = random.sample(img_files, num)

data = {'scipy': scipy_verify, 'sklearn': sklearn_verify}
for k, verify in data.items():
    error = 0.0
    for img in test_img_files:
        result = verify(imgs_folder+img)
        if img[:4] != result:
            error += 1
    print k+'\t模型正确率：', 1 - error / num
