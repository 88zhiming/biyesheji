# encoding=gbk
import os
import json
#交叉熵损失函数已经包含softmax函数
import math
import datetime
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torchvision import transforms, datasets
from tqdm import tqdm  # 进度条包
from torch.utils.tensorboard import SummaryWriter
import torch.optim.lr_scheduler as lr_scheduler

# from models.resnet_model import resnet18
from shufflenet_model import shufflenet_v2_x1_0
#from data_enhancement import mixup_data, mixup_criterion, cutmix_data, CutMixCollator, CutMixCriterion
import numpy as np
import matplotlib.pyplot as plt
from torchvision.utils import make_grid

# 选择'ggplot'样式
plt.style.use("classic")

current_time = datetime.datetime.now().strftime("%Y.%m.%d-%H.%M.%S")
# 每训练
#freeze_layers = True  # 2冻结权重
num_classes = 3  # 分的类别
lr = 0.01
gamma = 0.95

lrf = 0.01  # 学习率衰减
epochs = 80 # 训练论述
batch_size = 32


log_path = "/tmp/pycharm_project_773/shufflenet网络/训练结果/tensorboard" + current_time  # tensorboard + current_time  # tensorboard
save_path ="/tmp/pycharm_project_773/shufflenet网络/训练结果/权重存储" + current_time  # 权重存储
# model_weight_path = "/home/amax/fgq/Graduation project/cnn/pre-weights/resnet18-pre.pth"  # 预训练权重
image_path ="/home/amax/YSPACK/NEW_DATA_BYONSELF_SPIT_GENGXIN/"# 输入图像文件夹
save_weitht_txt =  "/tmp/pycharm_project_773/shufflenet网络/训练结果/文本记录" + current_time  # 文本记录训练过程 #文本记录训练过程

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("using {} device.".format(device))

    if os.path.exists(save_path) is False:
        os.makedirs(save_path)

    if os.path.exists(log_path) is False:
        os.makedirs(log_path)

    if not os.path.exists(save_weitht_txt):
        os.makedirs(save_weitht_txt)

    tb_writer = SummaryWriter(log_dir=log_path)

    data_transform = {
        "train": transforms.Compose([transforms.RandomResizedCrop(224),  # 随机裁剪
                                     transforms.RandomHorizontalFlip(),  # 对折变换
                                     transforms.ToTensor(),
                                     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]),

        "val": transforms.Compose([transforms.Resize(256),  # 将长宽比不动，缩放到256
                                   transforms.CenterCrop(224),  # 采用中心裁剪。
                                   transforms.ToTensor(),
                                   transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])}

    # 加载图片和预处理图片
    train_dataset = datasets.ImageFolder(root=os.path.join(image_path, "train"),
                                         transform=data_transform["train"])
    train_num = len(train_dataset)  # 表示有多少张图片

    # {'daisy':0, 'dandelion':1, 'roses':2, 'sunflower':3, 'tulips':4}
    flower_list = train_dataset.class_to_idx
    cla_dict = dict((val, key) for key, val in flower_list.items())
    # write dict into json file
    json_str = json.dumps(cla_dict, indent=3)
    with open('class_indices.json', 'w') as json_file:
        json_file.write(json_str)

    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])  # number of workers，多线程预处理
    print('Using {} dataloader workers every process'.format(nw))

    # 迭代器
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=batch_size, shuffle=True,
                                               num_workers=nw)

    validate_dataset = datasets.ImageFolder(root=os.path.join(image_path, "val"),
                                            transform=data_transform["val"])
    val_num = len(validate_dataset)
    validate_loader = torch.utils.data.DataLoader(validate_dataset,
                                                  batch_size=batch_size, shuffle=False,
                                                  num_workers=nw)

    print("using {} images for training, {} images for validation.".format(train_num,
                                                                           val_num))

    # net = model  # 此时最后一个全连接层有1000个节点。
    model_name = "vgg16"
    net =shufflenet_v2_x1_0(num_classes=num_classes)

    # 预训练权重路径

    # if os.path.exists(model_weight_path):
    #     weights_dict = torch.load(model_weight_path, map_location=device)
    #     model_dict = net.state_dict()
    # #
    # # #     # 筛除不加载的层结构
    # # #
    #     load_weights_dict = {k: v for k, v in weights_dict.items()
    #                          if (k in model_dict) and (model_dict[k].numel() == v.numel())}
    #     model_dict.update(load_weights_dict)
    #
    #     missing_keys, unexpected_keys = net.load_state_dict(model_dict, strict=False)
    #     print(f"missing_keys:{missing_keys}")


    # # freeze features weights
    # # 这里遍历net.features下的所有参数
    # for param in net.features.parameters():
    #     param.requires_grad = False
    #
    # par = []
    # for name, para in net.named_parameters():
    #     # 除最后的全连接层外，其他权重全部冻结
    #     if "fc" not in name:# and "layer4" not in name:  #
    #         para.requires_grad_(False)
    # #         par.append(name)
    # print("冻结参数", par)

    net.to(device)
    # define loss function
    loss_function = nn.CrossEntropyLoss()

    # construct an optimizer
    params = [p for p in net.parameters() if p.requires_grad]
    optimizer = optim.Adam(params, lr)#加入正则化
    scheduler = lr_scheduler.ExponentialLR(optimizer, gamma=gamma)

    best_acc = 0.0
    # save_path = './models/resNet34.pth'
    train_steps = len(train_loader)  # 一共330张图片，每次8张，需要42步，train_steps：就是42

    train_loss_tmp = {}
    train_acc_tmp = {}
    val_acc_tmp = {}
    val_loss_tmp = {}

    for epoch in range(epochs):
        start_t  = datetime.datetime.now()
        # train
        net.train()  # 训练模式
        running_loss = 0.0
        val_loss = 0.0
        train_bar = tqdm(train_loader)
        train_acc = torch.zeros(1).to(device)
        for step, data in enumerate(train_bar):  # 也就是说这里面要循环42次,也就是说一个epoch，要加载完所有图片，把所有图片进行训练。
            images, labels = data

            optimizer.zero_grad()
            logits = net(images.to(device))
            loss = loss_function(logits, labels.to(device))
            loss.backward()  # loss反向传播
            optimizer.step()  # 参数更新

            train_y = torch.max(logits, dim=1)[1]  # 找到最大预测结果，dim表示在第一个维度里面寻找最大值，[1]这里是找索引
            train_acc += torch.eq(train_y, labels.to(device)).sum().item()

                # print statistics
            running_loss += loss.item()  # 累加损失

            train_bar.desc = "train epoch[{}/{}] loss:{:.3f}".format(epoch,
                                                                         epochs,
                                                                         loss)
        scheduler.step()
        # validate
        net.eval()
        val_acc = torch.zeros(1).to(device)  # accumulate accurate number / epoch
        with torch.no_grad():  # 在接下来过程中，不要计算每个节点的误差损失梯度，如果不用这个函数，则在测试过程中也会计算损失误差梯度，它会消耗算力和内存资源。
            val_bar = tqdm(validate_loader)
            for val_data in val_bar:
                val_images, val_labels = val_data
                outputs = net(val_images.to(device))  # 正向传播
                loss = loss_function(outputs, val_labels.to(device))
                predict_y = torch.max(outputs, dim=1)[1]  # 找到最大预测结果，dim表示在第一个维度里面寻找最大值，[1]这里是找索引
                # torch.eq(predict_y, val_label).sum().这个求和可以计算出一共有多少个样本被正确预测，这里求出的结果是一个tensor，使用item()将数值取出
                val_acc += torch.eq(predict_y, val_labels.to(device)).sum().item()

                val_bar.desc = "valid epoch[{}/{}]".format(epoch,
                                                           epochs)
                val_loss += loss.item()  # 累加损失

        # train_acc：所有训练图片的准确率之和，train_num：所有的训练图片的数量。
        # running_loss：是每一个batch_size之后的loss之和。就是说330张图片，以8 batch喂入，则需要喂42次，才能算一个epoch。每次喂入得一个loss,那么总共的loss就是running_loss
        # 所以需要计算平均loss的时候就是除以train_steps，训练论述
        end_t = datetime.datetime.now()
        print("运行一次时间", (end_t - start_t).seconds)

        val_accurate = val_acc.item() / val_num
        train_accurate = train_acc.item() / train_num
        mean_loss = running_loss / train_steps
        val_loss_final = val_loss / val_num
        print('[epoch %d] train_loss: %.3f  train_accuracy: %.3f val_accuracy: %.3f  val_loss: %.3f' %
              (epoch, mean_loss, train_accurate, val_accurate, val_loss_final))

        tags = ["train_loss","train_accuracy", "val_accuracy", "leaning_rate", "val_loss"]
        tb_writer.add_scalar(tags[0], mean_loss, epoch)
        tb_writer.add_scalar(tags[1], train_accurate, epoch)
        tb_writer.add_scalar(tags[2], val_accurate, epoch)
        tb_writer.add_scalar(tags[3], optimizer.param_groups[0]["lr"], epoch)
        tb_writer.add_scalar(tags[4], val_loss_final, epoch)

        train_loss_tmp[epoch] =round(mean_loss,3)
        train_acc_tmp[epoch] = round(train_accurate,3)
        val_acc_tmp[epoch] = round(val_accurate,3)
        val_loss_tmp[epoch] = round(val_loss_final,3)

        # 将Loss,train_prec1,train_prec5,val_prec1,val_prec5用.txt的文件存起来
        data_save(save_weitht_txt + '/train_loss.txt', train_loss_tmp)
        data_save(save_weitht_txt + '/train_acc.txt', train_acc_tmp)
        data_save(save_weitht_txt + '/val_acc.txt', val_acc_tmp)
        data_save(save_weitht_txt + '/val_loss.txt', val_loss_tmp)

        # data_save(directory + 'train_loss.txt', Loss_plot)

        print(f"best_acc:{round(best_acc, 3)}")
        accurate = train_accurate + val_accurate
        if accurate > best_acc:
            best_acc = accurate
            print(
                f"save model epoch:{epoch},train_accurate:{round(train_accurate, 3)}, val_accurate:{round(val_accurate, 3)}")
            torch.save(net.state_dict(), save_path + "/model-{}.pth".format(epoch))

    draw_picture(save_weitht_txt)
    print('Finished Training')

def data_save(root, file):
    if not os.path.exists(root):
        with open(root, 'w') as _:
            pass

    file_temp = open(root, 'r')
    lines = file_temp.readlines()
    if not lines:
        epoch = -1
    else:
        epoch = lines[-1][:lines[-1].index(' ')]
    epoch = int(epoch)
    file_temp.close()

    file_temp = open(root, 'a')
    for line in file:
        if line > epoch:
            file_temp.write(str(line) + " " + str(file[line]) + '\n')
    file_temp.close()


def draw_picture(path):
#画训练结果图
    train_acc = []
    train_loss = []
    val_acc = []
    for name in os.listdir(path):
        imfile = os.path.join(path, name)
        # print(imfile)
        file_temp = open(imfile, 'r')
        lines = file_temp.readlines()

        if name == "train_acc.txt":
            for line in lines:
                tmp = line.split(" ")[1]
                train_acc.append(float(tmp.split("\\")[0]))

        if name == "val_acc.txt":
            for line in lines:
                tmp = line.split(" ")[1]
                val_acc.append(float(tmp.split("\\")[0]))
        epoch = lines[-1][:lines[-1].index(' ')]
        file_temp.close()


    fig, ax = plt.subplots(figsize=(7, 5))
    x = range(int(epoch)+1)

    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    #linestyle = "solid"  "dashed" "dashdot" "dotted"; linewidth=3.0
    ax.plot(x, train_acc, color="red", label="train_acc")  # 用名字代表颜色, 'red'代表红色
    ax.plot(x, val_acc, color="b",  label="val_acc")  # 颜色代码，(rgbcmyk)

    ax.set_title("结果对比", fontdict={"fontsize": 15})
    ax.set_xlabel("轮数")  # 添加横轴标签
    ax.set_ylabel("准确率")  # 添加纵轴标签
    ax.legend(loc="best")  # 展示图例
    plt.title('Training and validation accuracy')
    plt.savefig('Training and validation accuracy.jpg')
    # plt.title('Training and validation loss')
    # plt.savefig('densenet_download_model.jpg')
    plt.show()





if __name__ == '__main__':
    main()


# log_path = "/tmp/pycharm_project_773/vgg16做迁移学习/models/权重存储/22" + current_time  # tensorboard
# save_path = "/tmp/pycharm_project_773/vgg16做迁移学习/models/权重存储" + current_time  # 权重存储
# model_weight_path = "/home/amax/fgq/Graduation project/cnn/pre-weights/vgg16-pre.pth"  # 预训练权重
# image_path = "/home/amax/YSPACK/HECHENGDATA_10%"# 输入图像文件夹
# # image_path_new = '/home/amax/fgq/Graduation project/cnn/data/fog_data/fog_photos_sky/data'
# save_weitht_txt =  "/tmp/pycharm_project_773/vgg16做迁移学习/models/log" + current_time  #文本记录训练过程