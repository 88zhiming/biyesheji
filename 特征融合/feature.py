#coding=utf-8
import cv2
import numpy as np
import matplotlib.pyplot as plt
import sys
import os
import math
from skimage.transform import rotate
from skimage.feature import local_binary_pattern
from skimage import data, io,data_dir,filters, feature
from skimage.color import label2rgb
import skimage
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import cv2
# from dark_night.Tselect import findT_out
import torch
import datetime


#计算图像的对比度
def cal_contrast(img):
    """
    代码和理论来源
    https://blog.csdn.net/zsc201825/article/details/89645190:
    """
    img1 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # 彩色转为灰度图片
    m, n = img1.shape
    # 图片矩阵向外扩展一个像素
    img1_ext = cv2.copyMakeBorder(img1, 1, 1, 1, 1, cv2.BORDER_REPLICATE) / 1.0  # 除以1.0的目的是uint8转为float型，便于后续计算
    rows_ext, cols_ext = img1_ext.shape
    b = 0.0
    for i in range(1, rows_ext - 1):
        for j in range(1, cols_ext - 1):
            b += ((img1_ext[i, j] - img1_ext[i, j + 1]) ** 2 + (img1_ext[i, j] - img1_ext[i, j - 1]) ** 2 +
                  (img1_ext[i, j] - img1_ext[i + 1, j]) ** 2 + (img1_ext[i, j] - img1_ext[i - 1, j]) ** 2)

    fco = b / (4 * (m - 2) * (n - 2) + 3 * (2 * (m - 2) + 2 * (n - 2)) + 2 * 4)  # 对应上面48的计算公式
    t = m*n
    fco = fco/t
    return round(fco,6)

#计算饱和度
def cal_bright_and_saturation(img):
    m, n, c = img.shape
    size = m * n
    img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    H,S,V = cv2.split(img)
    return S, V

#统计直方图并进行归一化。
def Histogram_cal(img):
    hist = cv2.calcHist([img], [0], None, [256], [0, 255])
    sum =hist.sum()
    # 直方图归一化
    hist /= sum

    # plt.figure()
    # plt.title("Grayscale Histogram hc")
    # plt.xlabel("Bins")
    # plt.ylabel("% of Pixels")
    # plt.plot(hist)
    # plt.xlim([0, 256])
    # plt.show()

    return hist


#统计直方图并进行归一化。
def Histogram_cal_t(img):

    hist = cv2.calcHist([img], [0], None, [256], [0, 1])
    sum =hist.sum()
    # # 直方图归一化
    # hist /= sum
    # plt.figure()
    # plt.title("Grayscale Histogram t")
    # plt.xlabel("Bins")
    # plt.ylabel("% of Pixels")
    # plt.plot(hist)
    # plt.xlim([0, 256])
    # plt.show()

    return hist

#计算图像信息熵
def shang(image):
    tmp = []
    for i in range(256):
        tmp.append(0)
    val = 0
    k = 0
    res = 0
    img = np.array(image)
    for i in range(len(img)):
        for j in range(len(img[i])):
            val = img[i][j]
            tmp[val] = float(tmp[val] + 1)
            k = float(k + 1)
    for i in range(len(tmp)):
        tmp[i] = float(tmp[i] / k)
    for i in range(len(tmp)):
        if (tmp[i] == 0):
            res = res
        else:
            res = float(res - tmp[i] * (math.log(tmp[i]) / math.log(2.0)))
    return res

#梯度
def Gradient(image):
    """
        功能：计算图像的梯度图
        输入：(n, n, 1)的矩阵,即输入单通道图片
        输出：平均梯度数值。
        https://www.cnblogs.com/qianxia/p/11096993.html
    """
    #它是soble增强版，从直方图上看，它的细节更多点。
    grad_x = cv2.Scharr(image, cv2.CV_32F, 1, 0)
    grad_y = cv2.Scharr(image, cv2.CV_32F, 0, 1)
    gradx = cv2.convertScaleAbs(grad_x)
    grady = cv2.convertScaleAbs(grad_y)
    gradxy = cv2.addWeighted(gradx, 0.5, grady, 0.5, 0)

    return gradxy

#计算图像透射率图
# def cal_t(img):
#     gaussBlur = cv2.GaussianBlur(img, (3, 3), 2.5)
#     t_dark , t_img= findT_out(gaussBlur)
#     return t_dark , t_img

def feature_extra(img):
    # print(Dark_tensor[i].shape)
    # print(Dark_tensor[i])
    # 调整维度 变为numpy后维度：([256, 256, 3])
    img2 = img.cpu().numpy().transpose(1, 2, 0)
    # 反Normalize 与 归一化操作
    img2 = (img2 * [0.229, 0.224, 0.225] + [0.485, 0.456, 0.406]) * 255
    img2 = img2.astype('uint8')
    #转化为opencv图像
    img2 = cv2.cvtColor(np.asarray(img2), cv2.COLOR_RGB2BGR)
    image = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    # #透射率图的信息
    # dark, t_guid = cal_t(img2)
    # ft = Histogram_cal_t(t_guid)
    # ft = np.squeeze(ft)

    # 计算亮度和饱和度直方图
    S, V = cal_bright_and_saturation(img2)
    fs = Histogram_cal(S)
    fv = Histogram_cal(V)
    fs = np.squeeze(fs)
    fv = np.squeeze(fv)

    # 计算图像对比度信息
    fco = cal_contrast(img2)
    fco = np.array(fco)
    fco = np.expand_dims(fco, 0)

    # 平均梯度信息
    gd_img = Gradient(image)
    fmgd = Histogram_cal(gd_img * 255)
    fmgd = np.squeeze(fmgd)

    feature = np.concatenate(( fs, fv, fco, fmgd), axis=0)
    feature = np.expand_dims(feature, 1)
    # print('特征图的形状',feature.shape)
    #feature = [769,1]

    feature_tensor = torch.from_numpy(feature).transpose(1, 0)
    feature_tensor = feature_tensor.to(torch.float32)

    return feature_tensor

























# def fun():
#     img2 = cv2.imread('D:\\Graduation project\\img\\darknight\\img2\\5.5-7803_QL.jpg')
#     # img2 = cv2.resize(img2,(600,600))
#     image = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
#     # 计算透射率图的直方图
#     dark, t_guid = cal_t(img2)
#     ft = Histogram_cal_t(t_guid)
#     ft = np.squeeze(ft)
#     # print(ft.shape)
#     # print(type(ft))
#     #计算亮度信息
#     # 计算亮度信息
#     S, V = cal_bright_and_saturation(img2)
#     fs = Histogram_cal(S)
#     fv = Histogram_cal(V)
#     fs = np.squeeze(fs)
#     fv = np.squeeze(fv)
#     # print(fbr)
#     # print(type(fbr))
#     #计算图像对比度信息
#     fco = cal_contrast(img2)
#     fco = np.array (fco)
#     fco = np.expand_dims(fco, 0)
#     # print(fco)
#     # print(type(fco))
#     #平均梯度信息
#     gd_img = meanGradient(image)
#     fmgd = Histogram_cal(gd_img*255)
#     fmgd = np.squeeze(fmgd)
#     # print(fmgd.shape)
#     # print(type(fmgd))
#     feature = np.concatenate((ft, fs, fv, fco,fmgd), axis= 0)
#     feature = np.expand_dims(feature,1)
#     print(feature.shape)
#     # print(type(feature))
#
#
#
# import time
# time_start = time.time()  # 记录开始时间
# fun()
# time_end = time.time()  # 记录结束时间
# time_sum = time_end - time_start  # 计算的时间差为程序的执行时间，单位为秒/s
# print(time_sum)


# #对比度测试，符合。
# img0 = cv2.imread('D:\\Graduation project\\img\\darknight\\img2\\NH_236.JPG')
# contrast(img0)
# img1 = cv2.imread('D:\\Graduation project\\img\\darknight\\img2\\21.jpg')
# contrast(img1)
# img2 = cv2.imread('D:\\Graduation project\\img\\darknight\\img2\\221.jpg')
# contrast(img2)
# img3 = cv2.imread('D:\\Graduation project\\img\\darknight\\img2\\13.jpg')
# contrast(img3)


