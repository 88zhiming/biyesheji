# encoding=gbk
import os
import datetime
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
from torch.autograd import Variable
from tqdm import tqdm
from models.My_resnet_model_mix import resnet18
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
import torch.optim.lr_scheduler as lr_scheduler
import mydata
import util


# ѡ��'ggplot'��ʽ
plt.style.use("ggplot")

# ѵ������
freeze_layers = True  # 2����Ȩ��
num_classes = 3  # �ֵ����
lr = 0.001
gamma = 0.95

lrf = 0.01  # ѧϰ��˥��
epochs = 100
batch_size = 32

Tran_Learn = True
# model_name = "vgg16"
use_cbam = True
activate_silu = False
current_time = datetime.datetime.now().strftime("%Y.%m.%d-%H.%M.%S")
logs = True  #
# �Ƿ�����־
log_path = "/tmp/pycharm_project_773/�����ں�_ʹ���Լ��������ݼ�/models/ѵ�����/tensorboard" + current_time  # tensorboard
save_path = "/tmp/pycharm_project_773/�����ں�_ʹ���Լ��������ݼ�/models/ѵ�����/Ȩ�ش洢" + current_time  # Ȩ�ش洢
model_weight_path  = "/home/amax/YSPACK/pre_weigh/resnet18-pre.pth"
# ����ͼ���ļ���
image_path = "/home/amax/YSPACK/NOTSPIT_DATA/"
save_weight_txt = "/tmp/pycharm_project_773/�����ں�_ʹ���Լ��������ݼ�/models/ѵ�����/�ı���¼" + current_time  # �ı���¼ѵ������
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
net = resnet18(num_classes=num_classes, use_cbam= use_cbam, activate_silu = activate_silu)



def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("using {} device.".format(device))
    if os.path.exists(save_path) is False:
        os.makedirs(save_path)

    if logs:
        if os.path.exists(log_path) is False:
            os.makedirs(log_path)

        if not os.path.exists(save_weight_txt):
            os.makedirs(save_weight_txt)
    tb_writer = SummaryWriter(log_dir=log_path)
    #�����ݼ�����Ϊ9:1
    train_images_path, train_images_label, val_images_path, val_images_label = util.read_split_data(image_path)
    #
    data_transform = [
        {
            "train": transforms.Compose([transforms.RandomResizedCrop(224),  # ����ü�
                                         transforms.RandomHorizontalFlip(),
                                         transforms.ToTensor(),
                                         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]),
            "val": transforms.Compose([transforms.Resize(256),
                                       transforms.CenterCrop(224),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])},
        # vgg

        {
            "train": transforms.Compose([transforms.RandomResizedCrop(224),
                                         transforms.RandomHorizontalFlip(),
                                         transforms.ToTensor(),
                                         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]),
            "val": transforms.Compose([transforms.Resize((224, 224)),
                                       transforms.ToTensor(),
                                       transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])},
        # Alexnet
        {
            "train": transforms.Compose([transforms.RandomResizedCrop(224),
                                         transforms.RandomHorizontalFlip(),
                                         transforms.ToTensor(),
                                         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]),
            "val": transforms.Compose([transforms.Resize((224, 224)),  # cannot 224, must (224, 224)
                                       transforms.ToTensor(),
                                       transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        }]
#���ǽ�ͼ�����RGB��ת����Ȼ����Ŷ����ݽ�����Ӧ��Ԥ��������
    train_data_set = mydata.MyDataSet(images_path=train_images_path,
                               images_class=train_images_label,
                               transform=data_transform[0]["train"])
    val_data_set = mydata.MyDataSet(images_path=val_images_path,
                             images_class=val_images_label,
                             transform=data_transform[0]["val"])

    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])  # number of workers
    print('Using {} dataloader workers'.format(nw))
    train_loader = torch.utils.data.DataLoader(train_data_set,
                                               batch_size=batch_size,
                                               shuffle=True,
                                               num_workers=0,

                                               )
    val_loader = torch.utils.data.DataLoader(val_data_set,
                                             batch_size=batch_size,
                                             shuffle=True,
                                             num_workers=0,
                                             )
    # �����ֶ����ɱ�ǩ�ֵ䣬ע��˳�򡣲�ͬ���ݼ��ǵø��ġ�
    _LABEL = {0: '1', 1: '2', 2: '3'}
    print(_LABEL)
    val_num = len(val_data_set)
    train_num = len(train_data_set)
    print("using {} images for training, {} images for validation.".format(train_num,
                                                                           val_num))
    if Tran_Learn:
        if os.path.exists(model_weight_path):
            weights_dict = torch.load(model_weight_path, map_location=device)
            model_dict = net.state_dict()

            # ɸ�������صĲ�ṹ

            load_weights_dict = {k: v for k, v in weights_dict.items()
                                 if (k in model_dict) and (model_dict[k].numel() == v.numel())}
            model_dict.update(load_weights_dict)

            missing_keys, unexpected_keys = net.load_state_dict(model_dict, strict=False)
            print(f"missing_keys:{missing_keys}")

    # freeze features weights
    # �������net.features�µ����в���
    # for param in net.features.parameters():
    #     param.requires_grad = False
    if freeze_layers:
        par = []
        for name, para in net.named_parameters():
            # ������ȫ���Ӳ�͵��Ĳ��⣬����Ȩ��ȫ������
            if ("fc" not in name) and ("layer4" not in name):
                para.requires_grad_(False)
                par.append(name)
        print("�������", par)

    net.to(device)
    # define loss function��ʹ�ý�������ʧ����
    loss_function = nn.CrossEntropyLoss()

    # construct an optimizer
    params = [p for p in net.parameters() if p.requires_grad]
    optimizer = optim.Adam(params, lr=lr)
    scheduler = lr_scheduler.ExponentialLR(optimizer, gamma=gamma)
    best_acc = 0.0
    train_steps = len(train_loader)  #�ܹ�10800�ţ�һ��32�ţ���Ҫ�Ĳ���
    vail_steps = len(val_loader)  # �ܹ�1200�ţ�һ��32�ţ���Ҫ�Ĳ���

    train_loss_tmp = {}
    train_acc_tmp = {}
    val_acc_tmp = {}
    val_loss_tmp = {}

    for epoch in range(epochs):
        # train
        start_t = datetime.datetime.now()
        net.train()
        val_loss = 0.0
        running_loss = 0.0
        train_acc = torch.zeros(1).to(device)
        train_bar = tqdm(train_loader)
        for step, data in enumerate(train_bar):
            images, image_tensor, labels = data
            # ����3����������һ������ΪͼƬ���ڶ�������Ϊ����������ɵ�ͼƬ������������Ϊ��ǩ
            optimizer.zero_grad()
#��Ϊ��ʱ���е��Ǵ�ͳ������ģ����ȡ�����������ںϣ����Դ�ʱ������������Ϊ����������һ����ͼƬ����һ���Ǵ�ͳ��ȡ��������ɵ�����ͼ
            logits = net(images.to(device), image_tensor.to(device))
            loss = loss_function(logits, labels.to(device))
            loss.backward()
            optimizer.step()  # ��������

            train_y = torch.max(logits, dim=1)[1]  # �ҵ����Ԥ������dim��ʾ�ڵ�һ��ά������Ѱ�����ֵ��[1]������������
            train_acc += torch.eq(train_y, labels.to(device)).sum().item()
                # print statistics
            running_loss += loss.item()
            train_bar.desc = "train epoch[{}/{}] loss:{:.3f}".format(epoch + 1,
                                                                         epochs,
                                                                         loss)
        scheduler.step()
        # validate
        net.eval()
        acc = torch.zeros(1).to(device)  # accumulate accurate number / epoch
        with torch.no_grad():  # �ڽ����������У���Ҫ����ÿ���ڵ�������ʧ�ݶȣ��������������������ڲ��Թ�����Ҳ�������ʧ����ݶȣ����������������ڴ���Դ��
            val_bar = tqdm(val_loader)  #tqdm��ʾ���ǽ�������
            for val_data in val_bar:
                val_images, image_tensor, val_labels = val_data
                val_images, image_tensor, val_labels = val_images.to(device), image_tensor.to(device), val_labels.to(device)
                outputs = net(val_images, image_tensor)
                loss = loss_function(outputs, val_labels.to(device))
                predict_y = torch.max(outputs, dim=1)[1]
# torch.eq(predict_y, val_label).sum().�����Ϳ��Լ����һ���ж��ٸ���������ȷԤ�⣬��������Ľ����һ��tensor��ʹ��item()����ֵȡ��
                acc += torch.eq(predict_y, val_labels.to(device)).sum().item()
                misclassified_images(pred_y=predict_y, writer=tb_writer, target=val_labels, images=val_images,
                                     output=outputs, epoch=epoch, label=_LABEL, count=500)
                val_bar.desc = "valid epoch[{}/{}]".format(epoch + 1,
                                                           epochs)
                val_loss += loss.item()  # �ۼ���ʧ
 # train_acc������ѵ��ͼƬ��׼ȷ��֮�ͣ�train_num�����е�ѵ��ͼƬ��������
 # running_loss����ÿһ��batch_size֮���loss֮�͡�����˵330��ͼƬ����8 batchι�룬����Ҫι42�Σ�������һ��epoch��ÿ��ι���
# һ��loss,��ô�ܹ���loss����running_loss
# ������Ҫ����ƽ��loss��ʱ����ǳ���train_steps��ѵ������
        end_t = datetime.datetime.now()
        print("����һ��ʱ��", (end_t - start_t).seconds)

        val_accurate = acc.item() / val_num
        train_accurate = train_acc.item() / train_num
        mean_loss = running_loss / train_steps
        val_loss_final = val_loss / vail_steps

        print('[epoch %d] train_loss: %.4f  train_accuracy: %.4f val_accuracy: %.4f  val_loss: %.4f' %
              (epoch, mean_loss, train_accurate, val_accurate, val_loss_final))
        if logs:
            tags = ["train_loss", "train_accuracy", "val_accuracy", "leaning_rate", "val_loss"]
            tb_writer.add_scalar(tags[2], val_accurate, epoch)
            tb_writer.add_scalar(tags[0], mean_loss, epoch)
            tb_writer.add_scalar(tags[1], train_accurate, epoch)
            tb_writer.add_scalar(tags[3], optimizer.param_groups[0]["lr"], epoch)
            tb_writer.add_scalar(tags[4], val_loss_final, epoch)
#���������λ
            train_loss_tmp[epoch] = round(mean_loss, 4)
            train_acc_tmp[epoch] = round(train_accurate, 4)
            val_acc_tmp[epoch] = round(val_accurate, 4)
            val_loss_tmp[epoch] = round(val_loss_final, 4)

            # ��Loss,train_prec1,train_prec5,val_prec1,val_prec5��.txt���ļ�������
            data_save(save_weight_txt + '/train_loss.txt', train_loss_tmp)
            data_save(save_weight_txt + '/train_acc.txt', train_acc_tmp)
            data_save(save_weight_txt + '/val_acc.txt', val_acc_tmp)
            data_save(save_weight_txt + '/val_loss.txt', val_loss_tmp)

        # data_save(directory + 'train_loss.txt', Loss_plot)

        print(f"best_acc:{round(best_acc, 3)}")
        accurate = val_accurate
        if accurate > best_acc:
            best_acc = accurate
            print(
                f"save model epoch:{epoch},train_accurate:{round(train_accurate, 3)}, val_accurate:{round(val_accurate,3)}")
            torch.save(net.state_dict(), save_path + "/model-{}.pth".format(epoch))
    if logs:
        draw_picture(save_weight_txt)

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
    # ��ѵ�����ͼ
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
    x = range(int(epoch) + 1)
#����������ʽ
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    # linestyle = "solid"  "dashed" "dashdot" "dotted"; linewidth=3.0
    ax.plot(x, train_acc, color="red", label="train_acc")  # �����ִ�����ɫ, 'red'�����ɫ
    ax.plot(x, val_acc, color="b", label="val_acc")  # ��ɫ���룬(rgbcmyk)

    ax.set_title("����Ա�", fontdict={"fontsize": 15})
    ax.set_xlabel("����")  # ��Ӻ����ǩ
    ax.set_ylabel("׼ȷ��")  # ��������ǩ
    ax.legend(loc="best")  # չʾͼ��

    plt.show()


def image_show(inp):
    """
    inp:ͼ���tensor��b,c,x,h��
    ��ʾtensorͼƬ
    """
    plt.figure(figsize=(14, 3))
    # ��Ϊnumpy
    inp = inp.numpy().transpose((1, 2, 0))
    # ���һ�� std ��׼��
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean  # unnormalize

    # inp = np.clip(inp, 0, 1)#ֵ������01֮��
    plt.pause(0.001)

    plt.imshow(inp.astype('uint8'))  # ͼƬ��Ҫת��Ϊint8����


# ��¼��������ͼƬ
def misclassified_images(pred_y, writer, target, images, output, epoch, label, count):
    misclassified = (pred_y != target.data)  # �ж��Ƿ�һ��
    for index, image_tensor in enumerate(images[misclassified][:count]):
        # print(image_tensor.shape)
        img = image_tensor.cpu().numpy().transpose(1, 2, 0)
        # resnet
        img = ((img * [0.229, 0.224, 0.225] + [0.485, 0.456, 0.406]) * 255).astype('uint8')
        # vgg
        # img = ((img * [0.5, 0.5, 0.5] + [0.5, 0.5, 0.5]) * 255).astype('uint8')
        img = img.transpose(2, 1, 0)
        # print(img.shape)
        img_name = 'Epoch:{}-->Predict:{}-->Actual:{}'.format(epoch, label[pred_y[misclassified].tolist()[index]],
                                                              label[target.data[misclassified].tolist()[index]])
        writer.add_image(img_name, img, epoch)
        # writer.add_images(img_name, image_tensor, epoch)


if __name__ == '__main__':
    main()


