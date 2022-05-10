import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms, datasets
from torchvision.transforms import Compose
import os
import cv2
import os
import time
import random
from PIL import Image,ImageChops,ImageEnhance
# from albumentations import *
from skimage.util import random_noise

#------------cutmix--start--------------------------------------------------
def cutmix(batch, alpha):
    data, targets = batch

    indices = torch.randperm(data.size(0))
    shuffled_data = data[indices]
    shuffled_targets = targets[indices]

    lam = np.random.beta(alpha, alpha)

    image_h, image_w = data.shape[2:]
    cx = np.random.uniform(0, image_w)
    cy = np.random.uniform(0, image_h)
    w = image_w * np.sqrt(1 - lam)
    h = image_h * np.sqrt(1 - lam)
    x0 = int(np.round(max(cx - w / 2, 0)))
    x1 = int(np.round(min(cx + w / 2, image_w)))
    y0 = int(np.round(max(cy - h / 2, 0)))
    y1 = int(np.round(min(cy + h / 2, image_h)))

    data[:, :, y0:y1, x0:x1] = shuffled_data[:, :, y0:y1, x0:x1]
    targets = (targets, shuffled_targets, lam)

    return data, targets


class CutMixCollator:
    def __init__(self, alpha):
        self.alpha = alpha

    def __call__(self, batch):
        batch = torch.utils.data.dataloader.default_collate(batch)
        batch = cutmix(batch, self.alpha)
        return batch


class CutMixCriterion:
    def __init__(self, reduction):
        self.criterion = nn.CrossEntropyLoss(reduction=reduction)

    def __call__(self, preds, targets):
        targets1, targets2, lam = targets
        return lam * self.criterion(
            preds, targets1) + (1 - lam) * self.criterion(preds, targets2)
#-----------------------------------------------------------------------------------------
def rand_bbox(size, lam):
    W = size[2]
    H = size[3]
    cut_rat = np.sqrt(1. - lam)
    cut_w = np.int(W * cut_rat)
    cut_h = np.int(H * cut_rat)

    # uniform
    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2

def cutmix_data(x, y, alpha=1., use_cuda=True):
    """
    Args:
        x: 图像
        y: 标签
        alpha: cutmix的比例，比如0.2，则这个框占整个图的0.8
        use_cuda:

    Returns:

    """

    if alpha > 0.:

        lam = np.random.beta(alpha, alpha)

    else:

        lam = 1.

    batch_size = x.size()[0]

    if use_cuda:

        index = torch.randperm(batch_size).cuda()

    else:

        index = torch.randperm(batch_size)



    size=x.size()

    bbx1, bby1, bbx2, bby2=rand_bbox(size,lam)

    x[:, :, bbx1:bbx2, bby1:bby2] = x[index, :, bbx1:bbx2, bby1:bby2]

    # adjust lambda to exactly match pixel ratio

    lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (x.size()[-1] *x.size()[-2]))

    y_a, y_b = y, y[index]

    return x, y_a, y_b, lam

def cutmix_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)
#-----------------cutmix--end-----------------------------------------------------------

#-------------------mixup-start--------------------------------------------------------
def mixup_data(x, y, alpha=1.0, use_cuda=True):
    '''Returns mixed inputs, pairs of targets, and lambda'''
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]
    if use_cuda:
        index = torch.randperm(batch_size).cuda()
    else:
        index = torch.randperm(batch_size)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)
#-------------------mixup-end--------------------------------------------------------


#-----------------------Gridmask-start----------------------------------------------------
#https://blog.csdn.net/weixin_42990464/article/details/107687284

class Grid(object):
    def __init__(self, use_h, use_w, rotate=1, offset=True, ratio=0.005, mode=0, prob=1.):
        self.use_h = use_h
        self.use_w = use_w
        self.rotate = rotate
        self.offset = offset
        self.ratio = ratio
        self.mode = mode
        self.st_prob = prob
        self.prob = prob

    def set_prob(self, epoch, max_epoch):
        self.prob = self.st_prob * epoch / max_epoch

    def __call__(self, img):
        if np.random.rand() > self.prob:
            return img
        h = img.size(1)
        w = img.size(2)
        self.d1 = 2
        self.d2 = min(h, w)
        hh = int(1.5 * h)
        ww = int(1.5 * w)
        d = np.random.randint(self.d1, self.d2)
        # d = self.d
        #        self.l = int(d*self.ratio+0.5)
        if self.ratio == 1:
            self.l = np.random.randint(1, d)
        else:
            self.l = min(max(int(d * self.ratio + 0.5), 1), d - 1)
        mask = np.ones((hh, ww), np.float32)
        st_h = np.random.randint(d)
        st_w = np.random.randint(d)
        if self.use_h:
            for i in range(hh // d):
                s = d * i + st_h
                t = min(s + self.l, hh)
                mask[s:t, :] *= 0
        if self.use_w:
            for i in range(ww // d):
                s = d * i + st_w
                t = min(s + self.l, ww)
                mask[:, s:t] *= 0

        r = np.random.randint(self.rotate)
        mask = Image.fromarray(np.uint8(mask))
        mask = mask.rotate(r)
        mask = np.asarray(mask)
        #        mask = 1*(np.random.randint(0,3,[hh,ww])>0)
        mask = mask[(hh - h) // 2:(hh - h) // 2 + h, (ww - w) // 2:(ww - w) // 2 + w]

        mask = torch.from_numpy(mask).float()
        if self.mode == 1:
            mask = 1 - mask

        mask = mask.expand_as(img)
        if self.offset:
            offset = torch.from_numpy(2 * (np.random.rand(h, w) - 0.5)).float()
            offset = (1 - mask) * offset
            img = img * mask + offset
        else:
            img = img * mask

        return img


class GridMask(nn.Module):
    def __init__(self, use_h, use_w, rotate=1, offset=False, ratio=0.5, mode=0, prob=1.):
        super(GridMask, self).__init__()
        self.use_h = use_h
        self.use_w = use_w
        self.rotate = rotate
        self.offset = offset
        self.ratio = ratio
        self.mode = mode
        self.st_prob = prob

    def set_prob(self, epoch, max_epoch):
        self.prob = self.st_prob * epoch / max_epoch  # + 1.#0.5

    def forward(self, x):
        if np.random.rand() > self.prob or not self.training:
            return x
        n, c, h, w = x.size()
        x = x.view(-1, h, w)
        hh = int(1.5 * h)
        ww = int(1.5 * w)
        d = np.random.randint(2, h)
        # d = self.d
        # self.l = int(d*self.ratio+0.5)
        self.l = min(max(int(d * self.ratio + 0.5), 1), d - 1)
        mask = np.ones((hh, ww), np.float32)
        st_h = np.random.randint(d)
        st_w = np.random.randint(d)
        if self.use_h:
            for i in range(hh // d):
                s = d * i + st_h
                t = min(s + self.l, hh)
                mask[s:t, :] *= 0
        if self.use_w:
            for i in range(ww // d):
                s = d * i + st_w
                t = min(s + self.l, ww)
                mask[:, s:t] *= 0

        r = np.random.randint(self.rotate)
        mask = Image.fromarray(np.uint8(mask))
        mask = mask.rotate(r)
        mask = np.asarray(mask)
        #        mask = 1*(np.random.randint(0,3,[hh,ww])>0)
        mask = mask[(hh - h) // 2:(hh - h) // 2 + h, (ww - w) // 2:(ww - w) // 2 + w]

        mask = torch.from_numpy(mask).float().cuda()
        if self.mode == 1:
            mask = 1 - mask
        mask = mask.expand_as(x)
        if self.offset:
            offset = torch.from_numpy(2 * (np.random.rand(h, w) - 0.5)).float().cuda()
            x = x * mask + offset * (1 - mask)
        else:
            x = x * mask

        return x.view(n, c, h, w)

#------------------------Gridmask-end------------------------------------------------------


#-----------------------------图像增强--start---------------------------------------------------------------
# 水平翻转
def horizontalFlip(img):
    return np.fliplr(img)


# 垂直翻转
def verticalFlip(img):
    return np.flipud(img)


#随机旋转
def random_rotation(image):
    RR = transforms.RandomRotation(degrees=(10, 80))
    rr_image = RR(image)
    return rr_image

#保持图像中心不变的随机仿射变换
def RandomAffine(image):
    RR =  transforms.RandomAffine(degrees=30, translate=(0, 0.2), scale=(0.9, 1), shear=(6, 9), fillcolor=66)
    rr_image = RR(image)
    return rr_image


# 转换图像灰度。
def Grayscale(image):
    RR =  transforms.Grayscale(num_output_channels=3)
    rr_image = RR(image)
    return rr_image

def color_enhancement(img):#颜色增强
    index = None
    enh_col = ImageEnhance.Color(img)
    color = 1.5
    image_colored = enh_col.enhance(color)
    return [image_colored, index]

def image_enhanced(img):
    """图像随机扩充"""
    rand = random.randrange(-60, 60, 10)    # 选择旋转角度-60~60（逆时针旋转）
    randbri = random.choice([0.6, 0.8, 1.0, 1.2])   # 选择亮度，大于1增强，小于1减弱
    randcol = random.choice([0.7, 0.9, 1.1, 1.3])   # 选择色度，大于1增强，小于1减弱
    randcon = random.choice([0.7, 0.9, 1.1, 1.3])   # 选择对比度，大于1增强，小于1减弱
    randsha = random.choice([0.5, 1.0, 1.5])        # 选择锐度，大于1增强，小于1减弱

    lr = img.transpose(Image.FLIP_LEFT_RIGHT)
    bri = ImageEnhance.Brightness(lr)
    bri_img1 = bri.enhance(randbri)
    col = ImageEnhance.Color(bri_img1)
    col_img1 = col.enhance(randcol)
    con = ImageEnhance.Contrast(col_img1)
    con_img1 = con.enhance(randcon)
    sha = ImageEnhance.Sharpness(con_img1)
    sha_img1 = sha.enhance(randsha)






#-----------------------------图像增强--end---------------------------------------------------------------

image_path = "D:\\Graduation project\\cnn\\data\\fog_data\\fog_photos\\data\\vgood\\"                 #输入图像文件夹
save_path = "D:\\Graduation project\\cnn\\data\\fog_data\\fog_photos\\dataenhac\\vgood\\"

#批量运行
def betch_run(inputPath, outPath):
    """
    Args:
        inputPath: 输入图片文件夹路径
        outPath:  保存处理结果路径
        singleRun: 是否单独测试图片，默认否
        singImgeName: 单独测试图片名字,默认空
    Returns:批量运算结果
    """
    img_name = []
    for name in os.listdir(inputPath):
        if name.endswith("JPG") or name.endswith("jpg"):
            img_name.append(name)
    # nameList = fileNameSort(img_name)

    if os.path.exists(outPath) is False:
        os.makedirs(outPath)

    aug_img1 = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5), #以0.5的概率对PIL图片进行水平翻转
        transforms.RandomCrop((720,1080), padding = 4, padding_mode="constant"),  #随机裁剪，120是裁剪后的大小
        # transforms.RandomResizedCrop(1080),  # 随机裁剪
        transforms.Pad(padding=(8, 16, 32, 64), fill=(255, 0, 0), padding_mode='symmetric'),
        # transforms.RandomCrop((1080,1920), padding=1080, padding_mode='symmetric'),
        transforms.RandomAffine(degrees=(20,30), translate=(0, 0.2), scale=(0.9, 1), shear=(6, 9)),
        transforms.Grayscale(num_output_channels=3),
        # transforms.RandomRotation(degrees=(10, 80)),#旋转degrees，代表旋转角度范围
        transforms.RandomVerticalFlip(p=0.5), #概率0.5垂直翻转
    ])

    aug_img2 = transforms.Compose([

        transforms.RandomChoice([transforms.RandomVerticalFlip(p=1), transforms.RandomHorizontalFlip(p=1)]),
        # transforms.RandomCrop((720,1080), padding=1080, padding_mode='symmetric'),
        # transforms.RandomCrop((720, 1080), padding=4, pad_if_needed = True,padding_mode="constant"),  # 随机裁剪，120是裁剪后的大小
        transforms.RandomAffine(degrees=(20, 80), translate=(0, 0.4), scale=(0.9, 1), shear=(6, 9)),
        # transforms.Grayscale(num_output_channels=3),
        transforms.ColorJitter(brightness=(0.5,1.5), contrast=(0.5,1.5), saturation=(0.5,1.5), hue=(-0.5,0.5)),#brightness亮度  contrast对比度 saturation饱和度, huesude1

    ])

    for name in img_name:
        img_path = inputPath + name
        print(f"{img_path} start")
        img = Image.open(img_path)
        img = img.convert("RGB")
        image = aug_img2(img)
        image.save(save_path+"2-"+name)
        print(f"{name} finish")




if __name__ == '__main__':
    betch_run(image_path, save_path)