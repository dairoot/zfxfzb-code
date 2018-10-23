# coding: utf-8
from PIL import Image
import numpy as np


def str_turn_num(c):
    if c in ['9', 'o', 'z']:
        print('存在错误数据')
    return ord(c) - 87 if ord(c) >= 97 else ord(c) - 48


def num_turn_chr(n):
    if n in [9, 24, 35]:
        print('存在错误数据')
    return chr(n + 48) if n < 10 else chr(n + 87)


def denoise_img(img):
    '''图片降噪处理'''
    img2 = Image.new("L", img.size, 255)
    for x in range(img.size[1]):
        for y in range(img.size[0]):
            pix = img.getpixel((y, x))
            if pix == 17:  # these are the numbers to get
                img2.putpixel((y, x), 0)
    return img2


def get_img_data(img):
    img = denoise_img(img)
    x_size, y_size = img.size
    y_size -= 5
    piece = (x_size-22) // 8
    centers = [4+piece*(2*i+1) for i in range(4)]
    X = []
    for i, center in enumerate(centers):
        split_img = img.crop((center-(piece+2), 1, center+(piece+2), y_size))
        width, height = split_img.size
        X_a = []
        for h in range(0, height):
            for w in range(0, width):
                pixel = split_img.getpixel((w, h))
                if pixel == 255:
                    X_a.append(1)
                else:
                    X_a.append(0)
        X.append(X_a)
    return X


class Data(object):
    in_size = 336
    out_size = 36

    def __init__(self, file="data.dat"):
        self.cursor = 0
        self.data = np.loadtxt(file)

        y = self.data[:, self.in_size].reshape((-1, 1))
        self.g_y = np.rint(y == range(self.out_size))
        self.g_X = self.data[:, :self.in_size]

    def next_batch(self, amount):
        X_train = self.g_X[self.cursor:self.cursor+amount]
        y_train = self.g_y[self.cursor:self.cursor+amount]
        self.cursor += amount
        return X_train, y_train


class TrainData(Data):
    """docstring for TrainData"""

    def __init__(self, file="data.dat"):
        super(TrainData, self).__init__(file)
        print('Train data size: %d' % len(self.data))

    @property
    def test_xs(self):
        return self.g_X[-100:]

    @property
    def test_ys(self):
        return self.g_y[-100:]


class TestData(Data):
    def __init__(self, file="test.dat"):
        super(TestData, self).__init__(file)
        print('Test data size: %d' % len(self.data))
