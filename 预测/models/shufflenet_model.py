from typing import List, Callable

import torch
from torch import Tensor
import torch.nn as nn


def channel_shuffle(x: Tensor, groups: int) -> Tensor:

    batch_size, num_channels, height, width = x.size()
    #将channel分组
    channels_per_group = num_channels // groups

    # reshape
    # [batch_size, num_channels, height, width] -> [batch_size, groups, channels_per_group, height, width]
    #view：分组
    x = x.view(batch_size, groups, channels_per_group, height, width)

    #维度1，2调换，将tensor数据转化成内存中连续的数据。
    x = torch.transpose(x, 1, 2).contiguous()

    # flatten，展开
    x = x.view(batch_size, -1, height, width)

    return x

#它将2个模块集中在一起了
class InvertedResidual(nn.Module):
    def __init__(self, input_c: int, output_c: int, stride: int):
        super(InvertedResidual, self).__init__()

        if stride not in [1, 2]:
            raise ValueError("illegal stride value.")
        self.stride = stride

        assert output_c % 2 == 0   #因为左右分支的通道数一样，判断是否是2的整数倍
        branch_features = output_c // 2  #左或右的通道数
        # 当stride为1时，input_channel应该是branch_features的两倍,因为在输入的时候分开了
        # python中 '<<' 是位运算，可理解为计算×2的快速方法
        assert (self.stride != 1) or (input_c == branch_features << 1)
        #对应c和d图的左边分支
        if self.stride == 2:
            #对应d图左边
            self.branch1 = nn.Sequential(
                #dw卷积输入channel = 输出
                #dw+BN+conv+relu  dw层的输入和输出的通道数一样
                self.depthwise_conv(input_c, input_c, kernel_s=3, stride=self.stride, padding=1),
                nn.BatchNorm2d(input_c),
                nn.Conv2d(input_c, branch_features, kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(branch_features),
                nn.ReLU(inplace=True)
            )
        else:
            #对应c图左边
            self.branch1 = nn.Sequential()  #另一个捷径分支不做任何处理

       #对应c和d的右边分支，c图和d图的右分支结构一样
        self.branch2 = nn.Sequential(
            #stride = 1对应c图右边，=2对应d图右边
            nn.Conv2d(input_c if self.stride > 1 else branch_features, branch_features, kernel_size=1,
                      stride=1, padding=0, bias=False),
            nn.BatchNorm2d(branch_features),
            nn.ReLU(inplace=True),
            self.depthwise_conv(branch_features, branch_features, kernel_s=3, stride=self.stride, padding=1),
            nn.BatchNorm2d(branch_features),
            nn.Conv2d(branch_features, branch_features, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(branch_features),
            nn.ReLU(inplace=True)
        )

    @staticmethod
    def depthwise_conv(input_c: int,
                       output_c: int,
                       kernel_s: int,
                       stride: int = 1,
                       padding: int = 0,
                       bias: bool = False) -> nn.Conv2d:
        return nn.Conv2d(in_channels=input_c, out_channels=output_c, kernel_size=kernel_s,
                         stride=stride, padding=padding, bias=bias, groups=input_c)

    def forward(self, x: Tensor) -> Tensor:
        if self.stride == 1: #c图
            x1, x2 = x.chunk(2, dim=1) #均分处理，在通道上进行
            #branch2处理x2再和x1拼接，在channel维度上进行拼接
            out = torch.cat((x1, self.branch2(x2)), dim=1)
        else:
            #d图，将输入矩阵分别进入两个分支
            out = torch.cat((self.branch1(x), self.branch2(x)), dim=1)

        #channel_shuffle
        out = channel_shuffle(out, 2)

        return out


class ShuffleNetV2(nn.Module):
    def __init__(self,
                 stages_repeats: List[int],
                 stages_out_channels: List[int],  #channel数
                 num_classes: int = 1000,
                 inverted_residual: Callable[..., nn.Module] = InvertedResidual):
        super(ShuffleNetV2, self).__init__()

        if len(stages_repeats) != 3:#stages_repeats对应stage2，3，4
            raise ValueError("expected stages_repeats as list of 3 positive ints")
        if len(stages_out_channels) != 5: #对应stage1,2,3,4,5。的输出矩阵调整，应该有5个
            raise ValueError("expected stages_out_channels as list of 5 positive ints")
        self._stage_out_channels = stages_out_channels

        # input RGB image
        input_channels = 3
        output_channels = self._stage_out_channels[0]  #对应的通道数

        self.conv1 = nn.Sequential(
            nn.Conv2d(input_channels, output_channels, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(output_channels),
            nn.ReLU(inplace=True)
        )
        input_channels = output_channels

        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

       #对应的后面几个部分
        self.stage2: nn.Sequential
        self.stage3: nn.Sequential
        self.stage4: nn.Sequential

        #搭建stage2,3,4
        stage_names = ["stage{}".format(i) for i in [2, 3, 4]]
        for name, repeats, output_channels in zip(stage_names, stages_repeats,
                                                  self._stage_out_channels[1:]):
            #构建一个stage
            #下面一行构建整个stage中第一个block,它的stride=2,剩下的都为1
            seq = [inverted_residual(input_channels, output_channels, 2)]  #第一个block步距都为2
            for i in range(repeats - 1): #遍历剩下的block
                #然后一次遍历
                seq.append(inverted_residual(output_channels, output_channels, 1))
            #setattr给self设置一个变量，变量名字为name,变量值就是刚构建的seq
            setattr(self, name, nn.Sequential(*seq))
            input_channels = output_channels   #将当前输出channel,赋值给下一层input

        output_channels = self._stage_out_channels[-1]
        self.conv5 = nn.Sequential(
            nn.Conv2d(input_channels, output_channels, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(output_channels),
            nn.ReLU(inplace=True)
        )

        self.fc = nn.Linear(output_channels, num_classes)

    def _forward_impl(self, x: Tensor) -> Tensor:
        # See note [TorchScript super()]
        x = self.conv1(x)
        x = self.maxpool(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.conv5(x)
        x = x.mean([2, 3])  # 进行全局池化，global pool，使用mean做全局池化，只剩下batch和channel维度
        x = self.fc(x)
        return x

    def forward(self, x: Tensor) -> Tensor:
        return self._forward_impl(x)


def shufflenet_v2_x1_5(num_classes=1000):
    """
    Constructs a ShuffleNetV2 with 1.0x output channels, as described in
    `"ShuffleNet V2: Practical Guidelines for Efficient CNN Architecture Design"
    <https://arxiv.org/abs/1807.11164>`.
    weight: https://download.pytorch.org/models/shufflenetv2_x1-5666bf0f80.pth

    :param num_classes:
    :return:
    """
    model = ShuffleNetV2(stages_repeats=[4, 8, 4],    #是根据repeat来，对应stage2,3,4
                         stages_out_channels=[24, 176, 352, 704, 1024],  #整个是输入输出channel的调整。根据表outchannel来
                         num_classes=num_classes)

    return model


def shufflenet_v2_x1_0(num_classes=1000):
    """
    Constructs a ShuffleNetV2 with 1.0x output channels, as described in
    `"ShuffleNet V2: Practical Guidelines for Efficient CNN Architecture Design"
    <https://arxiv.org/abs/1807.11164>`.
    weight: https://download.pytorch.org/models/shufflenetv2_x1-5666bf0f80.pth

    :param num_classes:
    :return:
    """
    model = ShuffleNetV2(stages_repeats=[4, 8, 4],
                         stages_out_channels=[24, 116, 232, 464, 1024],  #整个是输入输出的调整。
                         num_classes=num_classes)

    return model


def shufflenet_v2_x0_5(num_classes=1000):
    """
    Constructs a ShuffleNetV2 with 0.5x output channels, as described in
    `"ShuffleNet V2: Practical Guidelines for Efficient CNN Architecture Design"
    <https://arxiv.org/abs/1807.11164>`.
    weight: https://download.pytorch.org/models/shufflenetv2_x0.5-f707e7126e.pth

    :param num_classes:
    :return:
    """
    model = ShuffleNetV2(stages_repeats=[4, 8, 4],
                         stages_out_channels=[24, 48, 96, 192, 1024],
                         num_classes=num_classes)

    return model
