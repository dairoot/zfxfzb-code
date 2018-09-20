#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
from PIL import Image
from utils import denoise_img, str_turn_num

folder = 'zfxfzb-code-data/img/'
# imgs_folder = folder+'zfxfzb_code/'
imgs_folder = folder+'train_img/'
single_img_name = folder + 'single_imgs/%s_%s.png'


class ProcePhoto():
    def __init__(self, imgs_folder, img_name, data_file_handle):
        self.img_name = img_name.split('.')[0]
        self.data_file_handle = data_file_handle
        img = Image.open(imgs_folder+img_name).convert("L")
        # 降噪
        self.im = denoise_img(img)

    def split_img(self):
        ''' 分割图片'''
        x_size, y_size = self.im.size
        y_size -= 5
        piece = (x_size-22) / 8
        centers = [4+piece*(2*i+1) for i in range(4)]
        for i, center in enumerate(centers):

            single_img = self.im.crop(
                (center-(piece+2), 1, center+(piece+2), y_size))
            # 存储分割后的图片
            # single_img_path = single_img_name % (self.img_name[i], self.img_name)
            # single_img.save(single_img_path)
            # 存储分割后的图片像素点
            self.save_img_data(single_img, self.img_name[i])

    def save_img_data(self, img, c):
        ''' 存储分割后的图片像素点 '''
        for h in range(0, img.size[1]):
            for w in range(0, img.size[0]):
                pixel = img.getpixel((w, h))
                if pixel == 255:
                    print >> self.data_file_handle, 1,
                else:
                    print >> self.data_file_handle, 0,
        print >> self.data_file_handle, str_turn_num(c), ''


if __name__ == "__main__":
    img_files = os.listdir(imgs_folder)
    with open('data.dat', 'w') as data_file:
        for img_name in img_files:
            im = ProcePhoto(imgs_folder, img_name, data_file)
            im.split_img()
