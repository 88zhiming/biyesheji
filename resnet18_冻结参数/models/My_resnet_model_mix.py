import torch.nn as nn
import torch
from torch import Tensor
from torch.nn import functional as F
import torchvision.models.resnet

import numpy as np
import torch
from torch import nn
from torch.nn import init
#torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True)
#kernel_size  卷积核尺寸，dilation  扩张操作：控制kernel点（卷积核点）的间距，默认值:1
#groups  group参数的作用是控制分组卷积，默认不分组，为1组。
#bias 为真，则在输出中添加一个可学习的偏差。默认：True。
#
class ChannelAttention(nn.Module):   #通道注意力机制进行定义
    def __init__(self, channel, reduction=16):
        super().__init__()    #b,w,h,c->  b ,c
        self.maxpool = nn.AdaptiveMaxPool2d(1)  #进行最大池化，将宽和高压缩成1，此时只有通道
        self.avgpool = nn.AdaptiveAvgPool2d(1) #进行平均池化，将宽和高压缩成1，此时只有通道
        self.se = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, 1, bias=False),  #进行通道的压缩
            nn.ReLU(),
            nn.Conv2d(channel // reduction, channel, 1, bias=False)   #进行通道的扩张
        )
        self.sigmoid = nn.Sigmoid()   #SIGMOD到0和1

    def forward(self, x):   #进行前向
        max_result = self.maxpool(x)
        avg_result = self.avgpool(x)
        max_out = self.se(max_result)
        avg_out = self.se(avg_result)
        output = self.sigmoid(max_out + avg_out)   #将池化和平均两块进行叠加
        return output
#在平面上进行变化，相当于是对面上的像素点权重进行变化，此时的通道数为1
#dim=1表示的是在通道维度上进行操作
class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        #定义卷积模块，要进行通道的变化，空间注意力机制中，输入的是最大和平均的两块，所以输入的通道数为2，输出的通道数为1
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=kernel_size // 2)
        self.sigmoid = nn.Sigmoid()
#b,c,w,h  所有第一维为通道
    def forward(self, x):
        max_result, _ = torch.max(x, dim=1, keepdim=True)  #跟上面的通道不一样，此时取的是通道上面的最大值
        avg_result = torch.mean(x, dim=1, keepdim=True)   #跟上面的通道不一样，此时取的是通道上面的平均值
        result = torch.cat([max_result, avg_result], 1)   #将平均和最大的两块进行叠加，在通道上
        output = self.conv(result)
        output = self.sigmoid(output)
        return output
#18层网络输出的最后的通道数时512

class CBAM(nn.Module):

    def __init__(self, channel=512, reduction=16, kernel_size=49):
        super().__init__()
        self.ca = ChannelAttention(channel=channel, reduction=reduction)
        self.sa = SpatialAttention(kernel_size=kernel_size)

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, x):
        b, c, _, _ = x.size()
        residual = x
        out = x * self.ca(x) #通道注意力机制的计算，此时输入的是原特征x
        out = out * self.sa(out)  #空间注意力机制的计算，此时输入的是通道注意力机制后的特征out
        return out + residual    #原来的图像雨特征进行叠加


def _make_divisible(ch, divisor=8, min_ch=None):
    """
    This function is taken from the original tf repo.
    It ensures that all layers have a channel number that is divisible by 8
    It can be seen here:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    """
    if min_ch is None:
        min_ch = divisor
    new_ch = max(min_ch, int(ch + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_ch < 0.9 * ch:
        new_ch += divisor
    return new_ch


#--------------------------eca注意力机制-start------------------------------
class eca_layer(nn.Module):
    """Constructs a ECA module.
    Args:
        channel: Number of channels of the input feature map
        k_size: Adaptive selection of kernel size
    """
    def __init__(self, channel, k_size=3):
        super(eca_layer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x: input features with shape [b, c, h, w]
        b, c, h, w = x.size()

        # feature descriptor on the global spatial information
        y = self.avg_pool(x)

        # Two different branches of ECA module
        y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)

        # Multi-scale information fusion
        y = self.sigmoid(y)

        return x * y.expand_as(x)
#-------------eca-end---------------------------------------------

def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)

class ECABasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, k_size=3):
        super(ECABasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes, 1)
        self.bn2 = nn.BatchNorm2d(planes)
        self.eca = eca_layer(planes, k_size)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.eca(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

#-------------------------------eca注意力机制end--------------------

#18和34层的残差结构
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channel, out_channel, stride=1, downsample=None, use_cbam = False, activate_silu = False ,**kwargs):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channel, out_channels=out_channel,
                               kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channel)  #归一化
        if activate_silu:    #激活函数
            self.relu = nn.SiLU()
        else:
            self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(in_channels=out_channel, out_channels=out_channel,   #残差结构中的第二个
                               kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channel)

        self.cbam = CBAM(out_channel)

        self.use_cbam = use_cbam

        self.downsample = downsample

    def forward(self, x):
        identity = x
        if self.downsample is not None:        #这部分指的是残差结构中的捷径部分
            identity = self.downsample(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.use_cbam:    #是否使用注意力机制
            out = self.cbam(out)

        out += identity
        out = self.relu(out)

        return out                #表示一个残差结构的完成


#50层以上的残差结构。Bottleneck有两个结构，一个结构是有跳跃层上有卷积层，调整深度，一个是不需要调整深度。
class Bottleneck(nn.Module):
    """
    注意：原论文中，在虚线残差结构的主分支上，第一个1x1卷积层的步距是2，第二个3x3卷积层步距是1。
    但在pytorch官方实现过程中是第一个1x1卷积层的步距是1，第二个3x3卷积层步距是2，
    这么做的好处是能够在top1上提升大概0.5%的准确率。
    可参考Resnet v1.5 https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch
    """
    expansion = 4  #第三层的卷积核个数是第一二层的4倍

    def __init__(self, in_channel, out_channel, stride=1, downsample=None,
                 groups=1, width_per_group=64, use_cbam = False, activate_silu = False):

        super(Bottleneck, self).__init__()

        width = int(out_channel * (width_per_group / 64.)) * groups  #卷积核个数

        self.conv1 = nn.Conv2d(in_channels=in_channel, out_channels=width,
                               kernel_size=1, stride=1, bias=False)  # squeeze channels #实线和虚线第一层stride=1，是一样的
        self.bn1 = nn.BatchNorm2d(width)
        # -----------------------------------------
        self.conv2 = nn.Conv2d(in_channels=width, out_channels=width, groups=groups,#从第二层开始步距发生变化
                               kernel_size=3, stride=stride, bias=False, padding=1)
        self.bn2 = nn.BatchNorm2d(width)
        # -----------------------------------------
        self.conv3 = nn.Conv2d(in_channels=width, out_channels=out_channel*self.expansion, #最后一层是前两层卷积核个数的4倍，步距为1
                               kernel_size=1, stride=1, bias=False)  # unsqueeze channels
        self.bn3 = nn.BatchNorm2d(out_channel*self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample


    def forward(self, x):
        identity = x   #输入特征
        if self.downsample is not None:  #为none是实线残差结构
            identity = self.downsample(x)   #不为none说明残差跳跃线上有采样。

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        # if self.use_ca:
        #     out = self.ca(out)
        # if self.use_ca:
        #     out = self.sa(out)

        out += identity
        out = self.relu(out)

        return out         #这就是50层以上的一个残差块。



class ResNet(nn.Module):

    def __init__(self,
                 block,                 #传入的残差结构
                 blocks_num,            #表示该层有几个残差模块
                 num_classes=1000,
                 include_top=True,      #方便restnet上实线更复杂网络
                 groups=1,
                 width_per_group=64,
                 use_cbam = False,
                 activate_silu = False):
        super(ResNet, self).__init__()
        self.include_top = include_top
        self.in_channel = 64           #表示的是经过MaxPool2d之后的通道数

        self.groups = groups
        self.width_per_group = width_per_group

        #resNet第一层 7x7层， padding = 3,是为了让输出图像高宽为减半，3是通道RGB, in_channel是卷积核个数
        self.conv1 = nn.Conv2d(3, self.in_channel, kernel_size=7, stride=2,
                               padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(self.in_channel)
        self.relu = nn.ReLU(inplace=True)
        #最大池化层，
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        #每一层的网络
        self.layer1 = self._make_layer(block, 64, blocks_num[0])
        self.layer2 = self._make_layer(block, 128, blocks_num[1], stride=2)
        self.layer3 = self._make_layer(block, 256, blocks_num[2], stride=2)
        self.layer4 = self._make_layer(block, 512, blocks_num[3], stride=2,  use_cbam = use_cbam, activate_silu = activate_silu)


        if self.include_top:
            #自适应平均池化下采样，最后输出高宽为1
            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))   #此时输出为（1,1,512）


            self.fc = nn.Sequential(
                nn.Linear(512 * block.expansion, 1024),
                nn.ReLU(True),                  #加了一层全连接层（512,1024）
                nn.Dropout(p=0.5),               #防止过拟合
                nn.Linear(1024, num_classes)
            )

        #对卷积层进行初始化。
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    #生成一层layer结构，block表示的残差结构，channel是每层layer的残差块首个卷积核的个数，block_num表示的是该层中有几个残差结构
    def _make_layer(self, block, channel, block_num, stride=1, use_cbam=False, activate_silu = False):
        """
            channel:number of kernel
        """
        downsample = None
        #18和34层会跳过if，50以上，会进入，产生下采样，在跳跃线上有卷积，主要调整深度

        #这个是定义跳跃线上的，卷积过程。因为有两种残差块，一种跳跃线上面没有卷积，一种有。有的仅仅是每个convx上第一个残差块有，所以先进行判断，整个是否是第一个残差块。
        if stride != 1 or self.in_channel != channel * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channel, channel * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(channel * block.expansion))

        layers = []

        #这个是第一个残差块。
        layers.append(block(self.in_channel,            #输入深度
                            channel,                    #第一个卷积层的卷积核个数
                            downsample=downsample,      #下采样函数
                            stride=stride,
                            groups=self.groups,
                            width_per_group=self.width_per_group,
                            use_cbam=use_cbam,
                            activate_silu=activate_silu
                            ))

        #30一下，通过第一层之后，深度没有发生变化，expansion是1；
        #虚线残差结构运算结束，深度会变大4倍，实线残差结构运算结束，深度不变。
        #第一层是虚线，所以深度变为expansion倍后，进入下一层。作为输入。但是后面都是实线残差结构，所以深度不变。也无跳跃下采样，故循环。
        self.in_channel = channel * block.expansion

        #一个layer中，第一个残差块是虚线的残差结构，残差捷径上有下采样，来改变图像深度，从第二个开始都是实现残差结构，不需要下采样
        #下面是对第二个残差块的实现
        for _ in range(1, block_num):
            layers.append(block(self.in_channel,
                                channel,
                                groups=self.groups,
                                width_per_group=self.width_per_group,
                                use_cbam = use_cbam,
                                activate_silu=activate_silu
                                ))

        return nn.Sequential(*layers)  #将一系列层结构组合在一起，并返回。将列表转换维非关键参数。


    #ResNetx正向传播，整体的网络正向传播
    def forward(self, x):
        # print(x.shape)
        # print(x1.shape)

        #第一层7x7
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        #最大池化下采样
        x = self.maxpool(x)

        #conv2
        x = self.layer1(x)
        #conv3
        x = self.layer2(x)
        #conv4
        x = self.layer3(x)

        x = self.layer4(x)

        if self.include_top:
            x = self.avgpool(x)  #平均下采样
    #这是对其进行展平处理，以便后面能进行全连接层计算
            x = torch.flatten(x, 1)

            x = self.fc(x)

        return x



def resnet18(num_classes=1000, include_top=True, use_cbam = False, activate_silu = False):
    # https://download.pytorch.org/models/resnet34-333f7ec4.pth
    return ResNet(BasicBlock,  [2, 2, 2, 2], num_classes=num_classes, include_top=include_top, use_cbam = use_cbam, activate_silu = activate_silu)

def resnet34(num_classes=1000, include_top=True):
    # https://download.pytorch.org/models/resnet34-333f7ec4.pth
    return ResNet(BasicBlock, [3, 4, 6, 3], num_classes=num_classes, include_top=include_top)


def resnet50(num_classes=1000, include_top=True):
    # https://download.pytorch.org/models/resnet50-19c8e357.pth
    return ResNet(Bottleneck, [3, 4, 6, 3], num_classes=num_classes, include_top=include_top)

