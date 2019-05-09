# -*- coding:utf-8 -*-
# description:辅助功能，图片预处理，转向量，生成测试集训练集

from PIL import Image, ImageDraw
import numpy as np
import os
import time

NUMBER = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']

CAPTCHA_LIST = NUMBER
CAPTCHA_LEN = 4
CAPTCHA_HEIGHT = 60
CAPTCHA_WIDTH = 160

IMG_PATH = "./img"
IMG_POSTFIX = ".png"
IMG_COUNT_INDEX = 0

IMG_RES_PATH = "./img_result"

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

def getImageByName(path, fileName):
    imageFilePath = os.path.join(path, fileName + IMG_POSTFIX)
    image = Image.open(imageFilePath)
    return image

#二值判断,如果确认是噪声,用改点的上面一个点的灰度进行替换
#该函数也可以改成RGB判断的,具体看需求如何
def getPixel(image,x,y,G,N):
    L = image.getpixel((x,y))
    if L > G:
        L = True
    else:
        L = False
 
    nearDots = 0
    if L == (image.getpixel((x - 1,y - 1)) > G):
        nearDots += 1
    if L == (image.getpixel((x - 1,y)) > G):
        nearDots += 1
    if L == (image.getpixel((x - 1,y + 1)) > G):
        nearDots += 1
    if L == (image.getpixel((x,y - 1)) > G):
        nearDots += 1
    if L == (image.getpixel((x,y + 1)) > G):
        nearDots += 1
    if L == (image.getpixel((x + 1,y - 1)) > G):
        nearDots += 1
    if L == (image.getpixel((x + 1,y)) > G):
        nearDots += 1
    if L == (image.getpixel((x + 1,y + 1)) > G):
        nearDots += 1
    if nearDots < N:
        return image.getpixel((x,y-1))
    else:
        return None
 
# 降噪 
# 根据一个点A的RGB值，与周围的8个点的RBG值比较，设定一个值N（0 <N <8），当A的RGB值与周围8个点的RGB相等数小于N时，此点为噪点 
# G: Integer 图像二值化阀值 
# N: Integer 降噪率 0 <N <8 
# Z: Integer 降噪次数 
# 输出 
#  0：降噪成功 
#  1：降噪失败 
def clearNoise(image,G,N,Z):
    draw = ImageDraw.Draw(image)
    for i in range(0,Z):
        for x in range(1,image.size[0] - 1):
            for y in range(1,image.size[1] - 1):
                color = getPixel(image,x,y,G,N)
                if color != None:
                    draw.point((x,y),color)

def binarizing(img, threshold): #input: gray image
    pixdata = img.load()
    w, h = img.size
    for y in range(h):
        for x in range(w):
            if pixdata[x, y] < threshold:
                pixdata[x, y] = 0
            else:
                pixdata[x, y] = 255
    return img

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

def splitAllImage():
    fileList = getFileNameList(IMG_PATH)
    for tempFile in fileList:
        image = getImageByName(IMG_PATH, tempFile)
        image = image.convert("L")
        image = binarizing(image, 160)
        clearNoise(image, 160, 4, 12)

        image1 = image.crop((0,0,30,40)).save(os.path.join(IMG_RES_PATH, list(tempFile)[0],str(time.time()) + IMG_POSTFIX))
        image2 = image.crop((30,0,60,40)).save(os.path.join(IMG_RES_PATH, list(tempFile)[1],str(time.time()) + IMG_POSTFIX))
        image3 = image.crop((60,0,90,40)).save(os.path.join(IMG_RES_PATH, list(tempFile)[2],str(time.time()) + IMG_POSTFIX))
        image4 = image.crop((90,0,120,40)).save(os.path.join(IMG_RES_PATH, list(tempFile)[3],str(time.time()) + IMG_POSTFIX))


if __name__ == '__main__':
    # batch_x, batch_y = get_next_batch(1)
    # print(batch_x)
    # print(batch_y)

    # batch_x2, batch_y2 = get_next_batch(1)
    # print(batch_x2)
    # print(batch_y2)

    splitAllImage()



