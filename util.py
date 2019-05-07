# -*- coding:utf-8 -*-
# description:辅助功能，图片预处理，转向量，生成测试集训练集

from PIL import Image
import numpy as np
import os

NUMBER = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']

CAPTCHA_LIST = NUMBER
CAPTCHA_LEN = 4
CAPTCHA_HEIGHT = 60
CAPTCHA_WIDTH = 160

IMG_PATH = "./img"
IMG_POSTFIX = ".png"
IMG_COUNT_INDEX = 0

def convert2gray(img):
    """
    图片转为黑白，3维转1维
    :param img: np
    :return:  灰度图的np
    """
    img = np.mean(img, -1)
    return img

def text2vec(text, captcha_len=CAPTCHA_LEN, captcha_list=CAPTCHA_LIST):
    """
    验证码文本转为向量
    :param text:
    :param captcha_len:
    :param captcha_list:
    :return: vector 文本对应的向量形式
    """
    text_len = len(text)    # 欲生成验证码的字符长度
    if text_len > captcha_len:
        raise ValueError('验证码最长4个字符')
    vector = np.zeros(captcha_len * len(captcha_list))      # 生成一个一维向量 验证码长度*字符列表长度
    for i in range(text_len):
        vector[captcha_list.index(text[i])+i*len(captcha_list)] = 1     # 找到字符对应在字符列表中的下标值+字符列表长度*i 的 一维向量 赋值为 1
    return vector

def getFileNameList(path):
    fileList = []
    for file in os.listdir(path):
        fileArray = os.path.splitext(file)
        if fileArray[1] == IMG_POSTFIX: 
            fileList.append(fileArray[0])
    return fileList

def getImageByName(path, fileName):
    imageFilePath = os.path.join(path, fileName + IMG_POSTFIX)
    image = Image.open(imageFilePath)
    image = image.resize((CAPTCHA_WIDTH, CAPTCHA_HEIGHT), Image.ANTIALIAS)
    return image

def get_next_batch(batch_count=60, width=CAPTCHA_WIDTH, height=CAPTCHA_HEIGHT):
    """
    获取训练图片组
    :param batch_count: default 60
    :param width: 验证码宽度
    :param height: 验证码高度
    :return: batch_x, batch_yc
    """
    batch_x = np.zeros([batch_count, width * height])
    batch_y = np.zeros([batch_count, CAPTCHA_LEN * len(CAPTCHA_LIST)])

    global IMG_COUNT_INDEX
    endIndex = IMG_COUNT_INDEX + batch_count
    fileList = getFileNameList(IMG_PATH)
    for i in range(IMG_COUNT_INDEX, endIndex, 1):  
        text = fileList[i]  
        image = getImageByName(IMG_PATH, text)
        image = convert2gray(image)     # 转灰度numpy

        # 将图片数组一维化 同时将文本也对应在两个二维组的同一行
        batch_x[(i-IMG_COUNT_INDEX), :] = image.flatten() / 255
        batch_y[(i-IMG_COUNT_INDEX), :] = text2vec(text)  # 验证码文本的向量形式

    IMG_COUNT_INDEX = endIndex
    return batch_x, batch_y

if __name__ == '__main__':
    # batch_x, batch_y = get_next_batch(1)
    # print(batch_x)
    # print(batch_y)

    # batch_x2, batch_y2 = get_next_batch(1)
    # print(batch_x2)
    # print(batch_y2)

    image = getImageByName(IMG_PATH, "0032")
    image.show()


