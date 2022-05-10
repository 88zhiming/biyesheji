# encoding=gbk
import os
import json
import math
import datetime

import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import torch.optim.lr_scheduler as lr_scheduler
import numpy as np

from torch.autograd import Variable
from torchvision import transforms, datasets
from tqdm import tqdm  # ��������
from torch.utils.tensorboard import SummaryWriter
from My_resnet_model_mix import resnet18
# from models.My_resnet_model2_nomix_1 import resnet18
# from models.moblie_v3_model import mobilenet_v3_large, mobilenet_v3_small
# from models.shuffle_model import shufflenet_v2_x1_0
# from models.vggnet_model import vgg
# from data_enhancement import mixup_data, mixup_criterion, cutmix_data, CutMixCollator, CutMixCriterion
from torchvision.utils import make_grid
from utils import read_split_data, plot_data_loader_image
from utils import read_split_data, plot_data_loader_image

# ѡ��'ggplot'��ʽ
plt.style.use('classic')

current_time = datetime.datetime.now().strftime("%Y.%m.%d-%H.%M.%S")
# ÿѵ��
freeze_layers = True  # 2����Ȩ��
num_classes = 3  # �ֵ����
lr = 0.01
gamma = 0.95

lrf = 0.01  # ѧϰ��˥��
epochs = 80  # ѵ������
batch_size = 64




use_cbam = True
activate_silu = False

logs = True   #�Ƿ�����־
log_path = "/tmp/pycharm_project_773/resnet18_�������_ʹ���Լ������ݼ�/models/ѵ�����/11" + current_time  # tensorboard
save_path = "/tmp/pycharm_project_773/resnet18_�������_ʹ���Լ������ݼ�/models/ѵ�����/Ȩ�ش洢" + current_time  # Ȩ�ش洢
model_weight_path = "/home/amax/fgq/Graduation project/cnn/pre-weights/resnet18-pre.pth"  # Ԥѵ��Ȩ��
image_path = "/home/amax/YSPACK/NEW_DATA_BYONSELF_SPIT_GENGXIN/"# ����ͼ���ļ���

save_weitht_txt ="/tmp/pycharm_project_773/resnet18_�������_ʹ���Լ������ݼ�/models/ѵ�����/�ı���¼" + current_time
os.environ["CUDA_VISIBLE_DEVICES"] = "3"

model = resnet18(num_classes=num_classes, use_cbam= use_cbam, activate_silu = activate_silu)

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
        "train": transforms.Compose([transforms.RandomResizedCrop(224),  # ����ü�
                                     transforms.RandomHorizontalFlip(),  # ���۱任
                                     transforms.ToTensor(),
                                     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]),
                                    #��׼���������Թ���������Ԥѵ����������
        "val": transforms.Compose([transforms.Resize(256),  # ������Ȳ�������С�����ŵ�256
                                   transforms.CenterCrop(224),  # �������Ĳü���
                                   transforms.ToTensor(),
                                   transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])}

    # ����ͼƬ��Ԥ����ͼƬ
    train_dataset = datasets.ImageFolder(root=os.path.join(image_path, "train"),
                                         transform=data_transform["train"])
    train_num = len(train_dataset)  # ��ʾ�ж�����ͼƬ

    # {'daisy':0, 'dandelion':1, 'roses':2, 'sunflower':3, 'tulips':4}
    flower_list = train_dataset.class_to_idx
    cla_dict = dict((val, key) for key, val in flower_list.items())
    # write dict into json file
    json_str = json.dumps(cla_dict, indent=4)
    with open('class_indices.json', 'w') as json_file:
        json_file.write(json_str)

    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])  # number of workers�����߳�Ԥ����
    print('Using {} dataloader workers every process'.format(nw))

    # ������
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

    net = model  # ��ʱ���һ��ȫ���Ӳ���1000���ڵ㡣
    # model_name = "vgg16"
    # net = vgg(model_name=model_name, num_classes=4, init_weights=False)

    # Ԥѵ��Ȩ��·��

    if os.path.exists(model_weight_path):
        weights_dict = torch.load(model_weight_path, map_location=device)
        model_dict = model.state_dict()

        # ɸ�������صĲ�ṹ

        load_weights_dict = {k: v for k, v in weights_dict.items()
                             if (k in model_dict) and (model_dict[k].numel() == v.numel())}
        model_dict.update(load_weights_dict)

        missing_keys, unexpected_keys = net.load_state_dict(model_dict, strict=False)
        print(f"missing_keys:{missing_keys}")


    par = []
    for name, para in net.named_parameters():
        # ������ȫ���Ӳ��⣬����Ȩ��ȫ������
        if "fc" not in name and "layer4" not in name:  #
            para.requires_grad_(False)
            par.append(name)
    print("�������", par)

    net.to(device)
    loss_function = nn.CrossEntropyLoss()

    params = [p for p in net.parameters() if p.requires_grad]
    optimizer = optim.Adam(params, lr,weight_decay=0.006)
    scheduler = lr_scheduler.ExponentialLR(optimizer, gamma=gamma)

    best_acc = 0.0
    train_steps = len(train_loader)  # һ��330��ͼƬ��ÿ��8�ţ���Ҫ42����train_steps������42
    vail_steps = len(validate_loader)  # һ��330��ͼƬ��ÿ��8�ţ���Ҫ42����train_steps������42

    train_loss_tmp = {}
    train_acc_tmp = {}
    val_acc_tmp = {}
    val_loss_tmp = {}

    for epoch in range(epochs):
        start_t  = datetime.datetime.now()
        # train
        net.train()  # ѵ��ģʽ
        running_loss = 0.0
        val_loss = 0.0
        train_bar = tqdm(train_loader)
        train_acc = torch.zeros(1).to(device)
        for step, data in enumerate(train_bar):  # Ҳ����˵������Ҫѭ��42��,Ҳ����˵һ��epoch��Ҫ����������ͼƬ��������ͼƬ����ѵ����
            images, labels = data
            # ------mixup---------------

            optimizer.zero_grad()
            logits = net(images.to(device))
            loss = loss_function(logits, labels.to(device))
            loss.backward()  # loss���򴫲�
            optimizer.step()  # ��������

            train_y = torch.max(logits, dim=1)[1]  # �ҵ����Ԥ������dim��ʾ�ڵ�һ��ά������Ѱ�����ֵ��[1]������������
            train_acc += torch.eq(train_y, labels.to(device)).sum().item()

                # print statistics
            running_loss += loss.item()  # �ۼ���ʧ

            train_bar.desc = "train epoch[{}/{}] loss:{:.3f}".format(epoch,
                                                                         epochs,
                                                                         loss)
        scheduler.step()
        # validate
        net.eval()
        val_acc = torch.zeros(1).to(device)  # accumulate accurate number / epoch
        with torch.no_grad():  # �ڽ����������У���Ҫ����ÿ���ڵ�������ʧ�ݶȣ��������������������ڲ��Թ�����Ҳ�������ʧ����ݶȣ����������������ڴ���Դ��
            val_bar = tqdm(validate_loader)
            for val_data in val_bar:
                val_images, val_labels = val_data
                outputs = net(val_images.to(device))  # ���򴫲�
                loss = loss_function(outputs, val_labels.to(device))
                predict_y = torch.max(outputs, dim=1)[1]  # �ҵ����Ԥ������dim��ʾ�ڵ�һ��ά������Ѱ�����ֵ��[1]������������
                # torch.eq(predict_y, val_label).sum().�����Ϳ��Լ����һ���ж��ٸ���������ȷԤ�⣬��������Ľ����һ��tensor��ʹ��item()����ֵȡ��
                val_acc += torch.eq(predict_y, val_labels.to(device)).sum().item()

                val_bar.desc = "valid epoch[{}/{}]".format(epoch,
                                                           epochs)
                val_loss += loss.item()  # �ۼ���ʧ

        # train_acc������ѵ��ͼƬ��׼ȷ��֮�ͣ�train_num�����е�ѵ��ͼƬ��������
        # running_loss����ÿһ��batch_size֮���loss֮�͡�����˵330��ͼƬ����8 batchι�룬����Ҫι42�Σ�������һ��epoch��ÿ��ι���һ��loss,��ô�ܹ���loss����running_loss
        # ������Ҫ����ƽ��loss��ʱ����ǳ���train_steps��ѵ������
        end_t = datetime.datetime.now()
        print("����һ��ʱ��", (end_t - start_t).seconds)

        val_accurate = val_acc.item() / val_num
        train_accurate = train_acc.item() / train_num
        mean_loss = running_loss / train_steps
        val_loss_final = val_loss / vail_steps
        print('[epoch %d] train_loss: %.4f  train_accuracy: %.4f val_accuracy: %.4f  val_loss: %.4f' %
              (epoch, mean_loss, train_accurate, val_accurate, val_loss_final))

        tags = ["train_loss","train_accuracy", "val_accuracy", "leaning_rate", "val_loss"]
        tb_writer.add_scalar(tags[0], mean_loss, epoch)
        tb_writer.add_scalar(tags[1], train_accurate, epoch)
        tb_writer.add_scalar(tags[2], val_accurate, epoch)
        tb_writer.add_scalar(tags[3], optimizer.param_groups[0]["lr"], epoch)
        tb_writer.add_scalar(tags[4], val_loss_final, epoch)

        train_loss_tmp[epoch] =round(mean_loss,4)
        train_acc_tmp[epoch] = round(train_accurate,4)
        val_acc_tmp[epoch] = round(val_accurate,4)
        val_loss_tmp[epoch] = round(val_loss_final,4)

        # ��Loss,train_prec1,train_prec5,val_prec1,val_prec5��.txt���ļ�������
        data_save(save_weitht_txt + '/train_loss.txt', train_loss_tmp)
        data_save(save_weitht_txt + '/train_acc.txt', train_acc_tmp)
        data_save(save_weitht_txt + '/val_acc.txt', val_acc_tmp)
        data_save(save_weitht_txt + '/val_loss.txt', val_loss_tmp)

        # data_save(directory + 'train_loss.txt', Loss_plot)

        print(f"best_acc:{round(best_acc, 4)}")
        accurate = val_accurate
        if accurate > best_acc:
            best_acc = accurate
            print(
                f"save model epoch:{epoch},train_accurate:{round(train_accurate, 4)}, val_accurate:{round(val_accurate, 4)}")
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
#��ѵ�����ͼ
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
    ax.plot(x, train_acc, color="red", label="train_acc")  # �����ִ�����ɫ, 'red'�����ɫ
    ax.plot(x, val_acc, color="b",  label="val_acc")  # ��ɫ���룬(rgbcmyk)

    ax.set_title("train old  result", fontdict={"fontsize": 15})
    ax.set_xlabel("epochs")  # ��Ӻ����ǩ
    ax.set_ylabel("acc")  # ��������ǩ
    ax.legend(loc="best")  # չʾͼ��


    plt.title('Training and validation accuracy')
    plt.savefig('Training and validation accuracy.jpg')
    plt.show()





if __name__ == '__main__':
    main()

