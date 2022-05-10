import os
import json

import torch
from torchvision import transforms, datasets
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from prettytable import PrettyTable

from models.resnet_model import resnet34, resnet50, resnext50_32x4d
# from models.moblie_v3_model import mobilenet_v3_large,mobilenet_v3_small
from models.My_resnet_model_mix import  resnet18
from models.shufflenet_model import shufflenet_v2_x1_0
from models.vggnet_model import vgg
use_cbam = True

from sklearn import metrics
from models.vggnet_model import vgg
from models.resnet_model import resnet18
# from  models.My_resnet_model_mix import  resnet18
activate_silu = False
num_classes = 3  # 分的类别


# modelName = "resnet50"
# model_name = "vgg16"
# net = vgg(model_name=model_name, num_classes=3, init_weights=False)
# create_model = model = vgg(model_name=model_name, num_classes=3, init_weights=False)
# create_model = model =shufflenet_v2_x1_0(num_classes=num_classes)
# create_model =resnet18(num_classes=num_classes, use_cbam= use_cbam, activate_silu = activate_silu)
model_name = "vgg16"
create_model = vgg(model_name=model_name, num_classes=3, init_weights=False)
model_weight_path ="/tmp/pycharm_project_773/vgg16模型/models/训练结果/权重存储2022.05.09-22.02.10/model-54.pth"
image_path ="/home/amax/YSPACK/TEST_DATE_HUNXIAO/"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
class ConfusionMatrix(object):
    """
    注意，如果显示的图像不全，是matplotlib版本问题
    本例程使用matplotlib-3.2.1(windows and ubuntu)绘制正常
    需要额外安装prettytable库  使输出能够打印成一个列表的形式
    num_classes:分类类别个数
    labels：分类标签列表
    """
    def __init__(self, num_classes: int, labels: list):
        self.matrix = np.zeros((num_classes, num_classes))  #初始化混淆矩阵
        self.num_classes = num_classes
        self.labels = labels

    #形成混淆矩阵
    def update(self, preds, labels):     #预测值与真实标签输入。
        for p, t in zip(preds, labels):   #zip(preds, labels)，预测和真实标签，打包组合后进行遍历，p预测，t真实
            self.matrix[p, t] += 1        #因为混淆矩阵：行是预测值，列是真实值

    #打印指标信息
    def summary(self):
        # calculate accuracy，计算准确率，准确率为对角线上所有预测对的样本除以总的样本数
        sum_TP = 0
        for i in range(self.num_classes):
            sum_TP += self.matrix[i, i]       #统计对角线
        acc = sum_TP / np.sum(self.matrix)     #np.sum(self.matrix)，求验证集样本个数。
        print("模型的总的正确率为： ", acc)

        # precision, recall, specificity
        table = PrettyTable()    #初始化一个列表
        table.field_names = ["", "Precision", "Recall", "Specificity"]  #给列表添加信息
        for i in range(self.num_classes):       #遍历每一个类别。
            TP = self.matrix[i, i]              #每一类对角线元素，表示的是预测对的样本数
            FP = np.sum(self.matrix[i, :]) - TP #将假预测为真，一行元素上所有的数值和减去TP（预测对的样本数）
            FN = np.sum(self.matrix[:, i]) - TP #把真预测为假，一列元素上的所有数值和减去TP（预测对的样本数）
            TN = np.sum(self.matrix) - TP - FP - FN  #假的预测，总的减去其他的。
            Precision = round(TP / (TP + FP), 3) if TP + FP != 0 else 0.  #每一类的精确值
            Recall = round(TP / (TP + FN), 3) if TP + FN != 0 else 0.      #每一类的召回率
            Specificity = round(TN / (TN + FP), 3) if TN + FP != 0 else 0.  #每一类的特异值
            table.add_row([self.labels[i], Precision, Recall, Specificity])  #在列表中添加相关元素
        print(table)  #打印表格

    #绘制混淆矩阵
    def plot(self):
        matrix = self.matrix
        print(matrix)  #打印混淆矩阵
        plt.imshow(matrix, cmap=plt.cm.Reds)  #颜色从白色到蓝色

        # 设置x轴坐标label，将原本x轴信息，替换为labels（原本是从0到numclass -1的数字）,将x轴旋转45度
        plt.xticks(range(self.num_classes), self.labels, rotation=45)  #横轴替换成自己的标签
        # 设置y轴坐标label
        plt.yticks(range(self.num_classes), self.labels)    #纵轴替换成自己的标签
        # 显示colorbar
        plt.colorbar()  #打开calor bar,显示数值的密集程度，颜色越深。数值分布越密集。
        plt.xlabel('True Labels')
        plt.ylabel('Predicted Labels')
        plt.title(' Confusion matrix')

        # 在图中标注数量/概率信息。每个区域的数值标注。
        thresh = matrix.max() / 2  #设置阈值，来确定数值颜色，取最大值的一半
        for x in range(self.num_classes): #显示图像，原点在左上角，x轴从左向右，y轴从上到下。
            for y in range(self.num_classes):
                # 注意这里的matrix[y, x]不是matrix[x, y]
                info = int(matrix[y, x])   #这里行是y坐标，列是x坐标。取整，得到当前位置的统计个数。
                plt.text(x, y, info,   #这里绘制数字和颜色。
                         verticalalignment='center',
                         horizontalalignment='center',
                         color="white" if info > thresh else "black")  #数值大于阈值，数字字体是白色。因为混淆矩阵，数值越大颜色越深，所以颜色深的地方选择白色数值。
        plt.tight_layout() #让图形显示更加紧凑，让图形完整显示出来。
        plt.savefig('resnet18原模型.jpg')
        plt.show()


if __name__ == '__main__': #下面是pytorch
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)
    # os.environ["CUDA_VISIBLE_DEVICES"] = "3,4"
    data_transform = transforms.Compose([transforms.Resize(256),
                                         transforms.CenterCrop(224),
                                         transforms.ToTensor(),
                                         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])


    validate_dataset = datasets.ImageFolder(root=os.path.join(image_path, "val"),
                                            transform=data_transform)

    batch_size = 8
    #载入验证集
    validate_loader = torch.utils.data.DataLoader(validate_dataset,
                                                  batch_size=batch_size, shuffle=False,
                                                  num_workers=2)
    net = create_model
    # model_name = "vgg16"
    # net = vgg(model_name=model_name, num_classes=4, init_weights=False)
    # load pretrain weights

    assert os.path.exists(model_weight_path), "cannot find {} file".format(model_weight_path)
    net.load_state_dict(torch.load(model_weight_path, map_location=device))
    net.to(device)

    # 读取json文件，为了后面提取label做准备
    json_label_path = '/tmp/pycharm_project_773/resnet18_冻结参数_使用自己的数据集/models/class_indices.json'
    assert os.path.exists(json_label_path), "cannot find {} file".format(json_label_path)
    json_file = open(json_label_path, 'r')
    class_indict = json.load(json_file)

    labels = [label for _, label in class_indict.items()] #提取label信息

    confusion = ConfusionMatrix(num_classes=3, labels=labels) #混淆矩阵
    net.eval()
    with torch.no_grad():
        for val_data in tqdm(validate_loader):#遍历数据集。
            val_images, val_labels = val_data #将数据分为图片和标签。
            outputs = net(val_images.to(device))
            outputs = torch.softmax(outputs, dim=1)  #表示的最大的预测最大的概率
            outputs = torch.argmax(outputs, dim=1)   #返回最大概率所对应的下标
            confusion.update(outputs.to("cpu").numpy(), val_labels.to("cpu").numpy())

    confusion.plot()
    confusion.summary()


