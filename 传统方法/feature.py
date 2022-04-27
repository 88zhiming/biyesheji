import cv2
import numpy as np
import imageio
import csv
import math
import os
# 定义最大灰度级数
gray_level = 16

#灰度共生矩阵特征提取
def maxGrayLevel(img):
    max_gray_level = 0
    (height, width) = img.shape
    # print (height, width)
    for y in range(height):
        for x in range(width):
            if img[y][x] > max_gray_level:
                max_gray_level = img[y][x]
    return max_gray_level + 1


def getGlcm(input, d_x, d_y):
    srcdata = input.copy()
    ret = [[0.0 for i in range(gray_level)] for j in range(gray_level)]
    (height, width) = input.shape

    max_gray_level = maxGrayLevel(input)

    # 若灰度级数大于gray_level，则将图像的灰度级缩小至gray_level，减小灰度共生矩阵的大小
    if max_gray_level > gray_level:
        for j in range(height):
            for i in range(width):
                srcdata[j][i] = srcdata[j][i] * gray_level / max_gray_level

    for j in range(height - d_y):
        for i in range(width - d_x):
            rows = srcdata[j][i]
            cols = srcdata[j + d_y][i + d_x]
            ret[rows][cols] += 1.0

    for i in range(gray_level):
        for j in range(gray_level):
            ret[i][j] /= float(height * width)

    return ret


def feature_computer(p):
    Con = 0.0
    Eng = 0.0
    Asm = 0.0
    Idm = 0.0
    for i in range(gray_level):
        for j in range(gray_level):
            Con += (i - j) * (i - j) * p[i][j]
            Asm += p[i][j] * p[i][j]
            Idm += p[i][j] / (1 + (i - j) * (i - j))
            if p[i][j] > 0.0:
                Eng += p[i][j] * math.log(p[i][j])
    return Asm, Con, -Eng, Idm


# def glcm(image_name):
def glcm(feature, filename):
    img = cv2.imread(filename)
    try:
        img_shape = img.shape
    except:
        print('imread error')

        return

    img = cv2.resize(img, (int(img_shape[1] / 2), int(img_shape[0] / 2)), interpolation=cv2.INTER_CUBIC)

    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    glcm_0 = getGlcm(img_gray, 1, 0)
    # glcm_1=getGlcm(src_gray, 0,1)
    # glcm_2=getGlcm(src_gray, 1,1)
    # glcm_3=getGlcm(src_gray, -1,1)

    asm, con, eng, idm = feature_computer(glcm_0)
    feature.extend([asm, con, eng, idm])
    return feature



def hsv(feature, filename):
    img = cv2.imread(filename)  # 读一张彩色图片
    if img is None:
        return

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)  # RGB空间转换为HSV空间
    h, s, v = cv2.split(hsv)
    color_feature = []  # 初始化颜色特征
    # 一阶矩（均值 mean）
    h_mean = np.mean(h)  # np.sum(h)/float(N)
    s_mean = np.mean(s)  # np.sum(s)/float(N)
    v_mean = np.mean(v)  # np.sum(v)/float(N)
    feature.extend([h_mean, s_mean, v_mean])  # 一阶矩放入特征数组
    return feature


def duibidu(feature, filename):
    img = imageio.imread(filename)

    max = (img[:, :, 0].max() + img[:, :, 1].max() + img[:, :, 1].max())
    min = (img[:, :, 0].min() + img[:, :, 1].min() + img[:, :, 1].min())
    result = (max - min) / img.size
    feature.extend([result])  # 对比度放入特征数组
    return feature


def white(feature, filename):
    img = cv2.imread(filename)
    width, height = img.shape[:2][::-1]
    # 将图片缩小便于显示观看
    img_resize = cv2.resize(img,
                            (int(width * 0.5), int(height * 0.5)), interpolation=cv2.INTER_CUBIC)

    # 将图片转为灰度图
    img_gray = cv2.cvtColor(img_resize, cv2.COLOR_RGB2GRAY)

    white = 0;
    len = np.shape(img_gray)[0]
    wid = np.shape(img_gray)[1]
    for i in range(0, len):
        for j in range(0, wid):
            if img_gray[i][j] > 200:
                white += 1
    feature.extend([white / (len * wid)])
    return feature


def lbp(feature, filename):
    img = cv2.imread(filename)
    # 将图片转为灰度图
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    x = olbp(gray)
    feature.extend([np.std(x)])
    return feature


def olbp(src):
    dst = np.zeros(src.shape, dtype=src.dtype)
    for i in range(1, src.shape[0] - 1):
        for j in range(1, src.shape[1] - 1):
            pass
            center = src[i][j]
            code = 0;
            code |= (src[i - 1][j - 1] >= center) << 7;
            code |= (src[i - 1][j] >= center) << 6;
            code |= (src[i - 1][j + 1] >= center) << 5;
            code |= (src[i][j + 1] >= center) << 4;
            code |= (src[i + 1][j + 1] >= center) << 3;
            code |= (src[i + 1][j] >= center) << 2;
            code |= (src[i + 1][j - 1] >= center) << 1;
            code |= (src[i][j - 1] >= center) << 0;

            dst[i - 1][j - 1] = code;
    return dst


def HOG(feature, filename):
    img = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
    if (img is None):
        print('picture con not load')
    resizeimg = cv2.resize(img, (128, 64), interpolation=cv2.INTER_CUBIC)
    cell_w = 8
    cell_x = int(resizeimg.shape[0] / cell_w)  # cell行数
    cell_y = int(resizeimg.shape[1] / cell_w)  # cell列数
    gammaimg = gamma(resizeimg) * 255
    feature_hog = hog(gammaimg, cell_x, cell_y, cell_w)
    feature.extend(feature_hog)
    return feature


# 计算图像HOG特征向量
def hog(img, cell_x, cell_y, cell_w):
    height, width = img.shape
    gradient_values_x = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=5)  # x方向梯度
    gradient_values_y = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=5)  # y方向梯度
    gradient_magnitude = np.sqrt(np.power(gradient_values_x, 2) + np.power(gradient_values_y, 2))
    gradient_angle = np.arctan2(gradient_values_x, gradient_values_y)
    print(gradient_magnitude.shape, gradient_angle.shape)
    # plt.figure()
    # plt.subplot( 1, 2, 1 )
    # plt.imshow(gradient_angle)
    # 角度转换至（0-180）
    gradient_angle[gradient_angle > 0] *= 180 / 3.14
    gradient_angle[gradient_angle < 0] = (gradient_angle[gradient_angle < 0] + 3.14) * 180 / 3.14
    # plt.subplot( 1, 2, 2 )
    # plt.imshow( gradient_angle )
    # plt.show()

    grad_cell = div(gradient_magnitude, cell_x, cell_y, cell_w)
    ang_cell = div(gradient_angle, cell_x, cell_y, cell_w)
    bins = get_bins(grad_cell, ang_cell)
    feature = []
    for i in range(cell_x - 1):
        for j in range(cell_y - 1):
            tmp = []
            tmp.append(bins[i, j])
            tmp.append(bins[i + 1, j])
            tmp.append(bins[i, j + 1])
            tmp.append(bins[i + 1, j + 1])
            tmp -= np.mean(tmp)
            feature.append(tmp.flatten())
    return np.array(feature).flatten()


# 灰度图像gamma校正
def gamma(img):
    # 不同参数下的gamma校正
    # img1 = img.copy()
    # img2 = img.copy()
    # img1 = np.power( img1 / 255.0, 0.5 )
    # img2 = np.power( img2 / 255.0, 2.2 )
    return np.power(img / 255.0, 1)


# 获取梯度值cell图像，梯度方向cell图像
def div(img, cell_x, cell_y, cell_w):
    cell = np.zeros(shape=(cell_x, cell_y, cell_w, cell_w))
    img_x = np.split(img, cell_x, axis=0)
    for i in range(cell_x):
        img_y = np.split(img_x[i], cell_y, axis=1)
        for j in range(cell_y):
            cell[i][j] = img_y[j]
    return cell


# 获取梯度方向直方图图像，每个像素点有9个值
def get_bins(grad_cell, ang_cell):
    bins = np.zeros(shape=(grad_cell.shape[0], grad_cell.shape[1], 9))
    for i in range(grad_cell.shape[0]):
        for j in range(grad_cell.shape[1]):
            binn = np.zeros(9)
            grad_list = np.int8(grad_cell[i, j].flatten())  # 每个cell中的64个梯度值展平，并转为整数
            ang_list = ang_cell[i, j].flatten()  # 每个cell中的64个梯度方向展平)
            ang_list = np.int8(ang_list / 20.0)  # 0-9
            ang_list[ang_list >= 9] = 0
            for m in range(len(ang_list)):
                binn[ang_list[m]] += int(grad_list[m])  # 不同角度对应的梯度值相加，为直方图的幅值
            # 每个cell的梯度方向直方图可视化
            # N = 9
            # x = np.arange( N )
            # str1 = ( '0-20', '20-40', '40-60', '60-80', '80-100', '100-120', '120-140', '140-160', '160-180' )
            # plt.bar( x, height = binn, width = 0.8, label = 'cell histogram', tick_label = str1 )
            # for a, b in zip(x, binn):
            # plt.text( a, b+0.05, '{}'.format(b), ha = 'center', va = 'bottom', fontsize = 10 )
            # plt.show()
            bins[i][j] = binn
    return bins
def white(feature, filename):
    img=cv2.imread(filename)
    # print(img.shape)
    width, height = img.shape[:2][::-1]
    # 将图片缩小便于显示观看
    img_resize = cv2.resize(img,
                            (int(width * 0.5), int(height * 0.5)), interpolation=cv2.INTER_CUBIC)

    # 将图片转为灰度图
    img_gray = cv2.cvtColor(img_resize, cv2.COLOR_RGB2GRAY)

    white = 0;
    len = np.shape(img_gray)[0]
    wid = np.shape(img_gray)[1]
    for i in range(0, len):
        for j in range(0, wid):
            if 260>img_gray[i][j] > 200:
                white += 1
    print([white / (len * wid)])
    # cv2.imshow('11',img)
    # cv2.waitKey(0)
    feature.extend([white / (len * wid)])
    return feature

def sharpness(feature, filename):
    # image = cv2.cvtColor(filename, cv2.COLOR_RGB2GRAY)  # to grayscale
    image = cv2.cvtColor(filename,cv2.COLOR_BGR2GRAY)
    array = np.asarray(image, dtype=np.int32)

    dx = np.diff(array)[1:, :]  # remove the first row
    dy = np.diff(array, axis=0)[:, 1:]  # remove the first column
    dnorm = np.sqrt(dx ** 2 + dy ** 2)

    sharpness = np.average(dnorm)
    print(sharpness)
    normalized_sharpness = sharpness / 40
    feature.extend(normalized_sharpness)
    return feature


def MAIN(feature, filename):
    feature = hsv(feature, filename)  #hsv特征
    feature = duibidu(feature, filename) #对比度特征
    feature = white(feature, filename)   #白色占比特征
    feature = lbp(feature, filename)    #纹理特征
    feature = glcm(feature,filename)    #灰度共生矩阵特征

    return feature


# 迭代读取文件夹下的所有文件
def read_file_all(data_dir_path):
    for f in os.listdir(data_dir_path):
        image_path = os.path.join(data_dir_path, f)
        print(f)
        if os.path.isfile(image_path):
            pass


if __name__ == '__main__':
    data_dir_path = r'/home/ubuntu/毕业设计/源代码/raininess/数据集/test_byoneself'
    # f = open('feature1.txt', 'w', encoding='utf-8')
    f = open('test_feature2.txt', 'w', encoding='utf-8')
    i = 1
    for file in os.listdir(data_dir_path):
        image_path = os.path.join(data_dir_path, file)
        for file1 in os.listdir(image_path): #打开指定路径下的所有的文件
            image_path1 = os.path.join(image_path, file1)#表示每一张图片的路径，此时表示的买一张图片的路径
            if os.path.isfile(image_path1):
                feature = []  # 初始化数组
                try:
                    feature = MAIN(feature, image_path1)
                    feature.extend([int(str(file))])
                    for j in range(len(feature)):
                        f.write(str(feature[j]))
                        if j < len(feature)-1:
                            f.write(',')
                    f.write('\r')
                    f.flush()
                    print(image_path1)
                    print(i)
                    i += 1
                except Exception as e:
                    print(e)
    f.close()
