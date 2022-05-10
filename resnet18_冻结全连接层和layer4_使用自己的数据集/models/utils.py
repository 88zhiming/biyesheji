import os
import json
import pickle
import random
from PIL import Image
import cv2

import matplotlib.pyplot as plt

#root文件路径， val_rate验证集比例
#两个文件 root:整幅图， rootNew:仅天空区域
def read_split_data(root: str, rootNew: str, val_rate: float = 0.3):
    random.seed(0)  # 保证随机结果可复现
    assert os.path.exists(root), "dataset root: {} does not exist.".format(root)
    assert os.path.exists(rootNew), "dataset root: {} does not exist.".format(rootNew)

    # 遍历文件夹，一个文件夹对应一个类别。
    #os.listdir(root)；列出所有文件， os.path.isdir(os.path.join(root, cla))判断os.path.join(root, cla)是否是文件夹，是则该文件夹名字保存到flower_class
    flower_class = [cla for cla in os.listdir(root) if os.path.isdir(os.path.join(root, cla))]
    #flower_class：[每个文件夹的名字]

    # 排序，保证顺序一致
    flower_class.sort()

    # 生成类别名称以及对应的数字索引
    #（key,val） = (类别名称，索引)
    class_indices = dict((k, v) for v, k in enumerate(flower_class))

    #dict((val, key) for key, val in class_indices.items()，key,val值反过来，现在字典键 = 索引，值=类别
    json_str = json.dumps(dict((val, key) for key, val in class_indices.items()), indent=4)
    #写入json
    with open('class_indices.json', 'w') as json_file:
        json_file.write(json_str)

    train_images_path = []  # 存储训练集的所有图片路径
    train_images_path_new = []  # 存储训练集的所有图片路径
    train_images_label = []  # 存储训练集图片对应索引信息

    val_images_path = []  # 存储验证集的所有图片路径
    val_images_path_new = []
    val_images_label = []  # 存储验证集图片对应索引信息
    every_class_num = []  # 存储每个类别的样本总数
    supported = [".jpg", ".JPG", ".png", ".PNG"]  # 支持的文件后缀类型

    # 遍历每个文件夹下的文件flower_class存储的是类别名称，也就是类别文件夹名称
    for cla in flower_class:
        #根目录核+类别名称，拼成一个文件夹完整路径
        cla_path = os.path.join(root, cla)
        cla_path_new = os.path.join(rootNew, cla)

        # 遍历获取supported支持的所有文件路径
        #1、for i in os.listdir(cla_path)：遍历文件夹中文件
        #2、 if os.path.splitext(i)[-1]，对文件名字i进行分割，得两个元素，名字和后缀。取后缀判断是否在我们支持得后缀文件里面
        #3、将文件夹中我们支持得后缀图片文件，拼接成一个完整得路径。放到images中
        #images是一个存储一个个图片文件路径得路径列表，这里需要修改，改成两个文件列表归为一类。
        #root 根目录，cla 类别名称， i图片文件名称
        images = []
        root_pic_name = os.listdir(cla_path)
        rootNew_pic_name = os.listdir(cla_path_new)
        for i in root_pic_name:
            #确保两个文件夹都存在该图
            # 确保两个文件夹都存在该图
            if i[1] == '-':
                i = i.split("-")[-1]

            if ((os.path.splitext(i)[-1]) in supported) and (i in rootNew_pic_name):
                images.append(os.path.join(root, cla, i))
            else:
                print(f"{cla_path_new}中找不到{i}")


        # 获取该类别对应的索引
        image_class = class_indices[cla]
        # 记录该类别的样本数量
        every_class_num.append(len(images))
        print("every_class_num", every_class_num)
        # print("images", len(images))
        # 按比例随机采样验证样本， k是选取个数，放测试集列表
        val_path = random.sample(images, k=int(len(images) * val_rate))
        # print("val_path", len(val_path))
        val_images_two_path1 = []
        train_images_two_path1 = []
        for img_path in images:
            if img_path in val_path:  # 如果该路径在采样的验证集样本中则存入验证集
                val_images_path.append(img_path)
                val_images_label.append(image_class)

                imgName = img_path.split("\\")[-1]
                img_path_new = os.path.join(rootNew, cla, imgName)
                val_images_path_new.append(img_path_new)
                # val_images_two_path1.append([img_path,img_path_new ])

            else:  # 否则存入训练集
                train_images_path.append(img_path)
                train_images_label.append(image_class)

                imgName = img_path.split("\\")[-1]
                img_path_new = os.path.join(rootNew, cla, imgName)
                train_images_path_new.append(img_path_new)
                # train_images_two_path1.append([img_path, img_path_new])


    train_images_two_path = [train_images_path,train_images_path_new]
    val_images_two_path = [val_images_path,val_images_path_new]



        #val_images_path_two：[原图列表，天空列表]  ，train_images_two_path1：[[原图， 天空图]，。。。， [原图，天空图]]

    print("{} images were found in the dataset.".format(sum(every_class_num)))
    print("{} images for training.".format(len(train_images_path)))
    print("{} images for validation.".format(len(val_images_path)))

    plot_image = False
    #这个是绘制训练图像数量直方图
    if plot_image:
        # 绘制每种类别个数柱状图
        plt.bar(range(len(flower_class)), every_class_num, align='center')
        # 将横坐标0,1,2,3,4替换为相应的类别名称
        plt.xticks(range(len(flower_class)), flower_class)
        # 在柱状图上添加数值标签
        for i, v in enumerate(every_class_num):
            plt.text(x=i, y=v + 5, s=str(v), ha='center')
        # 设置x坐标
        plt.xlabel('image class')
        # 设置y坐标
        plt.ylabel('number of images')
        # 设置柱状图的标题
        plt.title('flower class distribution')
        plt.show()

    return train_images_two_path, train_images_label, val_images_two_path, val_images_label


#打印图片和标签
def plot_data_loader_image(data_loader):
    batch_size = data_loader.batch_size
    #绘制图片个数
    plot_num = min(batch_size, 4)

    json_path = './class_indices.json'
    assert os.path.exists(json_path), json_path + " does not exist."
    json_file = open(json_path, 'r')
    # 载入标签文件 key是数字索引，val是类别
    class_indices = json.load(json_file)

    # 遍历一个baitc_size中的图片
    for data in data_loader:
        images, images_news, labels = data

        #找到签plotnum个图像和标签数据
        for i in range(plot_num):
            # [C, H, W] -> [H, W, C]
            #调整图像顺序
            img = images[i].numpy().transpose(1, 2, 0)
            # 反Normalize操作，得到得是随机裁剪后得图像，此时图片是float类型
            img = (img * [0.229, 0.224, 0.225] + [0.485, 0.456, 0.406]) * 255

            images_new = images_news[i].numpy().transpose(1, 2, 0)
            # 反Normalize操作，得到得是随机裁剪后得图像，此时图片是float类型
            images_new = (images_new * [0.229, 0.224, 0.225] + [0.485, 0.456, 0.406]) * 255

            #因为label是tensor了，通过item获得数据
            label = labels[i].item()
            #下面是绘制，1行，plot_num列， 第i+1个图像
            plt.subplot(2, plot_num, i+1)
            plt.xlabel(class_indices[str(label)])
            plt.xticks([])  # 去掉x轴的刻度
            plt.yticks([])  # 去掉y轴的刻度
            #绘制图片
            plt.imshow(img.astype('uint8')) #图片需要转换为int8类型

            plt.subplot(2, plot_num, i+3)
            plt.xlabel("img1-"+class_indices[str(label)])
            plt.xticks([])  # 去掉x轴的刻度
            plt.yticks([])  # 去掉y轴的刻度
            # 绘制图片
            plt.imshow(images_new.astype('uint8'))  # 图片需要转换为int8类型

        plt.show()

def write_pickle(list_info: list, file_name: str):
    with open(file_name, 'wb') as f:
        pickle.dump(list_info, f)


def read_pickle(file_name: str) -> list:
    with open(file_name, 'rb') as f:
        info_list = pickle.load(f)
        return info_list


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

class FocalLoss_1(nn.Module):

    def __init__(self, weight=None, reduction='mean', gamma=0, eps=1e-7):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.eps = eps
        self.ce = torch.nn.CrossEntropyLoss(weight=weight, reduction=reduction)

    def forward(self, input, target):
        logp = self.ce(input, target)
        p = torch.exp(-logp)
        loss = (1 - p) ** self.gamma * logp
        return loss.mean()


class FocalLoss(nn.Module):
    r"""
        This criterion is a implemenation of Focal Loss, which is proposed in
        Focal Loss for Dense Object Detection.

            Loss(x, class) = - \alpha (1-softmax(x)[class])^gamma \log(softmax(x)[class])

        The losses are averaged across observations for each minibatch.

        Args:
            alpha(1D Tensor, Variable) : the scalar factor for this criterion
            gamma(float, double) : gamma > 0; reduces the relative loss for well-classiﬁed examples (p > .5),
                                   putting more focus on hard, misclassiﬁed examples
            size_average(bool): By default, the losses are averaged over observations for each minibatch.
                                However, if the field size_average is set to False, the losses are
                                instead summed for each minibatch.


    """

    def __init__(self, class_num, alpha=None, gamma=2, size_average=True):
        super(FocalLoss, self).__init__()
        if alpha is None:
            self.alpha = Variable(torch.ones(class_num, 1))
        else:
            if isinstance(alpha, Variable):
                self.alpha = alpha
            else:
                self.alpha = Variable(alpha)
        self.gamma = gamma
        self.class_num = class_num
        self.size_average = size_average

    def forward(self, inputs, targets):
        N = inputs.size(0)
        C = inputs.size(1)
        P = F.softmax(inputs)

        class_mask = inputs.data.new(N, C).fill_(0)
        class_mask = Variable(class_mask)
        ids = targets.view(-1, 1)
        class_mask.scatter_(1, ids.data, 1.)
        # print(class_mask)

        if inputs.is_cuda and not self.alpha.is_cuda:
            self.alpha = self.alpha.cuda()
        alpha = self.alpha[ids.data.view(-1)]

        probs = (P * class_mask).sum(1).view(-1, 1)

        log_p = probs.log()
        # print('probs size= {}'.format(probs.size()))
        # print(probs)

        batch_loss = -alpha * (torch.pow((1 - probs), self.gamma)) * log_p
        # print('-----bacth_loss------')
        # print(batch_loss)

        if self.size_average:
            loss = batch_loss.mean()
        else:
            loss = batch_loss.sum()
        return loss


# 注释版
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from torch.autograd import Variable
#
# class FocalLoss(nn.Module):
#     def __init__(self, class_num, alpha=None, gamma=2, size_average=True):
#         super(FocalLoss, self).__init__()
#         if alpha is None:  # alpha 是平衡因子
#             self.alpha = Variable(torch.ones(class_num, 1))
#         else:
#             if isinstance(alpha, Variable):
#                 self.alpha = alpha
#             else:
#                 self.alpha = Variable(alpha)
#         self.gamma = gamma  # 指数
#         self.class_num = class_num  # 类别数目
#         self.size_average = size_average  # 返回的loss是否需要mean一下
#
#     def forward(self, inputs, targets):
#         # target : N, 1, H, W
#         inputs = inputs.permute(0, 2, 3, 1)
#         targets = targets.permute(0, 2, 3, 1)
#         num, h, w, C = inputs.size()
#         N = num * h * w
#         inputs = inputs.reshape(N, -1)   # N, C
#         targets = targets.reshape(N, -1)  # 待转换为one hot label
#         P = F.softmax(inputs, dim=1)  # 先求p_t
#         class_mask = inputs.data.new(N, C).fill_(0)
#         class_mask = Variable(class_mask)
#         ids = targets.view(-1, 1)
#         class_mask.scatter_(1, ids.data, 1.)  # 得到label的one_hot编码
#
#         if inputs.is_cuda and not self.alpha.is_cuda:
#             self.alpha = self.alpha.cuda()  # 如果是多GPU训练 这里的cuda要指定搬运到指定GPU上 分布式多进程训练除外
#         alpha = self.alpha[ids.data.view(-1)]
#         # y*p_t  如果这里不用*， 还可以用gather提取出正确分到的类别概率。
#         # 之所以能用sum，是因为class_mask已经把预测错误的概率清零了。
#         probs = (P * class_mask).sum(1).view(-1, 1)
#         # y*log(p_t)
#         log_p = probs.log()
#         # -a * (1-p_t)^2 * log(p_t)
#         batch_loss = -alpha * (torch.pow((1 - probs), self.gamma)) * log_p
#
#         if self.size_average:
#             loss = batch_loss.mean()
#         else:
#             loss = batch_loss.sum()
#         return loss

