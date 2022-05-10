import torch.nn as nn
import torch
import torchvision.models.resnet


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channel, out_channel, stride=1, k_size=3, downsample=None, **kwargs):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channel, out_channels=out_channel,
                               kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channel)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(in_channels=out_channel, out_channels=out_channel,
                               kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channel)
        self.downsample = downsample

    def forward(self, x):
        identity = x
        if self.downsample is not None:
            identity = self.downsample(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += identity
        out = self.relu(out)

        return out
# -------------------------------eca注意力机制end--------------------




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
                 groups=1, width_per_group=64):

        super(Bottleneck, self).__init__()

        width = int(out_channel * (width_per_group / 64.)) * groups  #卷积核个数

        self.conv1 = nn.Conv2d(in_channels=in_channel, out_channels=width,
                               kernel_size=1, stride=1, bias=False)  # squeeze channels #实线和虚线第一层stride=1，是一样的
        self.bn1 = nn.BatchNorm2d(width)
        # -----------------------------------------
        self.conv2 = nn.Conv2d(in_channels=width, out_channels=width, groups=groups,
                               kernel_size=3, stride=stride, bias=False, padding=1)
        self.bn2 = nn.BatchNorm2d(width)
        # -----------------------------------------
        self.conv3 = nn.Conv2d(in_channels=width, out_channels=out_channel*self.expansion, #最后一层是前两层卷积核个数的4倍。
                               kernel_size=1, stride=1, bias=False)  # unsqueeze channels
        self.bn3 = nn.BatchNorm2d(out_channel*self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x):
        identity = x   #输入特征
        if self.downsample is not None:  #为none是实线残差结构
            identity = self.downsample(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        out += identity
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self,
                 block,                 #传入的残差结构
                 blocks_num,            #残差结构的数据，列表34层[3,4,6,3]
                 num_classes=1000,
                 include_top=True,      #方便restnet上实线更复杂网络
                 groups=1,
                 width_per_group=64):
        super(ResNet, self).__init__()
        self.include_top = include_top
        self.in_channel = 64           #开始输入大小

        self.groups = groups
        self.width_per_group = width_per_group

        #resNet第一层 7x7层， padding = 3,是为了让输出图像高宽为减半，3是通道RGB, in_channel是卷积核个数
        self.conv1 = nn.Conv2d(3, self.in_channel, kernel_size=7, stride=2,
                               padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(self.in_channel)
        self.relu = nn.ReLU(inplace=True)

        #最大池化层，
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        #64，卷积核个数； blocks_num：该残差块个数，因为第二层的
        self.layer1 = self._make_layer(block, 64, blocks_num[0])
        self.layer2 = self._make_layer(block, 128, blocks_num[1], stride=2)
        self.layer3 = self._make_layer(block, 256, blocks_num[2], stride=2)
        self.layer4 = self._make_layer(block, 512, blocks_num[3], stride=2)

        if self.include_top:
            #自适应平均池化下采样，最后输出高宽为1
            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))  # output size = (1, 1)
            #全连接层，输入节点个数，就是平均池化下采样展平后节点个数。num_classes：输出节点个数，分类类别个数。
            # self.fc = nn.Linear(512 * block.expansion, num_classes)
            # self.fc=  nn.Dropout(p=0.5),  # 防止过拟合
            self.fc = nn.Sequential(
                nn.Linear(512 * block.expansion, num_classes),
                # nn.ReLU(True),                  #加了一层全连接层（512,1024）
                # nn.LeakyReLU(True),
                # nn.Dropout(p=0.04),               #防止过拟合
                # nn.Linear(1024, num_classes)
            )

        #对卷积层进行初始化。
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    def _make_layer(self, block, channel, block_num, stride=1):
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
                            width_per_group=self.width_per_group))

        #30一下，通过第一层之后，深度没有发生变化，expansion是一；
        #虚线残差结构运算结束，深度会变大4倍，实线残差结构运算结束，深度不变。
        #第一层是虚线，所以深度变为expansion倍后，进入下一层。作为输入。但是后面都是实线残差结构，所以深度不变。也无跳跃下采样，故循环。
        self.in_channel = channel * block.expansion

        #下面是改大份卷积后面几个残差块。
        for _ in range(1, block_num):
            layers.append(block(self.in_channel,
                                channel,
                                groups=self.groups,
                                width_per_group=self.width_per_group))

        return nn.Sequential(*layers)  #将一系列层结构组合在一起，并返回

    #ResNetx正向传播
    def forward(self, x):

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
        #conv5
        x = self.layer4(x)

        if self.include_top:
            x = self.avgpool(x)
            #展平
            x = torch.flatten(x, 1)
            #全连接
            x = self.fc(x)

        return x


def resnet18(num_classes=1000, include_top=True):
    # https://download.pytorch.org/models/resnet34-333f7ec4.pth
    return ResNet(BasicBlock, [2, 2, 2, 2], num_classes=num_classes, include_top=include_top)


def resnet34(num_classes=1000, include_top=True):
    # https://download.pytorch.org/models/resnet34-333f7ec4.pth
    return ResNet(BasicBlock, [3, 4, 6, 3], num_classes=num_classes, include_top=include_top)


def resnet50(num_classes=1000, include_top=True):
    # https://download.pytorch.org/models/resnet50-19c8e357.pth
    return ResNet(Bottleneck, [3, 4, 6, 3], num_classes=num_classes, include_top=include_top)


def resnet101(num_classes=1000, include_top=True):
    # https://download.pytorch.org/models/resnet101-5d3b4d8f.pth
    return ResNet(Bottleneck, [3, 4, 23, 3], num_classes=num_classes, include_top=include_top)


def resnext50_32x4d(num_classes=1000, include_top=True):
    # https://download.pytorch.org/models/resnext50_32x4d-7cdf4587.pth
    groups = 32
    width_per_group = 4
    return ResNet(Bottleneck, [3, 4, 6, 3],
                  num_classes=num_classes,
                  include_top=include_top,
                  groups=groups,
                  width_per_group=width_per_group)


def resnext101_32x8d(num_classes=1000, include_top=True):
    # https://download.pytorch.org/models/resnext101_32x8d-8ba56ff5.pth
    groups = 32
    width_per_group = 8
    return ResNet(Bottleneck, [3, 4, 23, 3],
                  num_classes=num_classes,
                  include_top=include_top,
                  groups=groups,
                  width_per_group=width_per_group)
