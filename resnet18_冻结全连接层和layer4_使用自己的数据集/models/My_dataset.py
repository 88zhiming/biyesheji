from PIL import Image
import torch
from torch.utils.data import Dataset
# import utilis1
from torchvision import transforms
import cv2
import numpy


class MyDataSet(Dataset):
    """自定义数据集"""

    def __init__(self, images_path: list, images_class: list, transform=None, transform_sky=None):
        self.images_path = images_path[0]
        self.images_path_new = images_path[1]
        self.images_class = images_class
        self.transform = transform
        self.transform_sky = transform_sky

    # 计算数据集下所有样本个数，注意传入的数据是哪种方式。
    def __len__(self):
        return len(self.images_path)

    # 每次传入索引返回该索引所对应图片和标签信息
    def __getitem__(self, item):
        # self.images_path[item]得到一个路径，item是一个索引，索引是我们的batch_size产生的。
        # 注意这里的图片是使用PIL处理的，如果需要opencv需要格式转换
        img = Image.open(self.images_path[item])
        img_new =  Image.open(self.images_path_new[item])


        # RGB为彩色图片，L为灰度图片
        img = Image.open(self.images_path[item]).convert('RGB')
        img_new = Image.open(self.images_path_new[item]).convert('RGB')

        label = self.images_class[item]

        #这部分是处理官方的数据，输出的图片已经是tensor形式
        if self.transform is not None:
            img = self.transform(img)

        if self.transform_sky is not None:
            img_new = self.transform_sky(img_new)

        #这里可以处理非官方数据，图片预处理
        #PIL->tensor



        return img, img_new, label

    @staticmethod
    def collate_fn(batch):#打包方式，这里batch是外在输入，就是上面得到img和label构成的元组。如果batch_size为8，则此时batch是有8组数据，每组都是[图片tensor,label]
        # 官方实现的default_collate可以参考
        # https://github.com/pytorch/pytorch/blob/67b7e751e6b5931a9f45274653f4f653a4e6cdf6/torch/utils/data/_utils/collate.py
        images, images_new, labels = tuple(zip(*batch))  # *batch非关键字参数，生成三组数据，每组有batch_size个

        images = torch.stack(images, dim=0)  # torch.stack拼接，增加一个维度，在0维度拼接（batch_size, c, w, h）
        images_new = torch.stack(images_new, dim=0)  # torch.stack拼接，增加一个维度，在0维度拼接（batch_size, c, w, h）
        labels = torch.as_tensor(labels)  # 将label转化为tensor
        return images, images_new, labels
