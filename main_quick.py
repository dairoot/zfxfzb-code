#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import ray
from PIL import Image
from utils import denoise_img, str_turn_num

folder = 'zfxfzb-code-data/img/'


class ProcePhoto(object):
    def __init__(self, img_path):
        self.img_name = img_path.split('/')[-1][:4]
        img = Image.open(img_path).convert("L")
        self.img = denoise_img(img)

    def photo_to_text(self):
        ''' 图片转数据 '''
        x_size, y_size = self.img.size
        y_size -= 5
        piece = (x_size - 22) // 8
        centers = [4 + piece * (2 * i + 1) for i in range(4)]
        photo_data = []
        data_str = ''
        for i, center in enumerate(centers):
            single_img = self.img.crop((center - (piece + 2), 1, center + (piece + 2), y_size))
            width, height = single_img.size
            photo_data_x = []
            for h_index in range(0, height):
                for w_index in range(0, width):
                    pixel = single_img.getpixel((w_index, h_index))
                    data_str += '1 ' if pixel == 255 else '0 '
            data_str = '%s%s\n' % (data_str, str_turn_num(self.img_name[i]))
        return data_str
    

@ray.remote
def run(img_path):
    im = ProcePhoto(img_path)
    data = im.photo_to_text()
    return data


if __name__ == "__main__":
    ray.init()
    dat = 'data.dat'
    imgs_folder = folder+'zfxfzb_code/'
    if not os.path.exists(imgs_folder):
        print ("请运行样本生成程序\nUser: ./checkcode.exe")
    else:
        img_files = os.listdir(imgs_folder)
        with open(dat, 'w') as data_file:
            task = []
            for img_name in img_files:
                img_path = imgs_folder+img_name
                task.append(run.remote(img_path))
            data = ray.get(task)
            data_file.write(''.join(data))
