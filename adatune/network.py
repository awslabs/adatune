# MIT License
#
# Copyright (c) 2017 liukuang
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init


def network(network_name, dataset):
    if dataset == 'mnist':
        return get_network(network_name, 10, 1)
    elif dataset == 'cifar_10':
        if network_name == 'mlp':
            raise Exception('MLP is currently only designed for grayscale images')
        return get_network(network_name, 10, 3)
    elif dataset == 'cifar_100':
        if network_name == 'mlp':
            raise Exception('MLP is currently only designed for grayscale images')
        return get_network(network_name, 100, 3)
    else:
        raise Exception('dataset is not supported')


def init_weights(m):
    if type(m) == nn.Conv2d:
        init.xavier_uniform_(m.weight, gain=np.sqrt(2))
    elif type(m) == nn.BatchNorm2d:
        init.constant_(m.weight, 1)
        init.constant_(m.bias, 0)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class BottleNeck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(BottleNeck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion * planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion * planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):

    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


def resnet_18(num_classes):
    return ResNet(BasicBlock, [2, 2, 2, 2], num_classes)


def resnet_34(num_classes):
    return ResNet(BasicBlock, [3, 4, 6, 3], num_classes)


def resnet_50(num_classes):
    return ResNet(BottleNeck, [3, 4, 6, 3], num_classes)


def resnet_101(num_classes):
    return ResNet(BottleNeck, [3, 4, 23, 3], num_classes)


def resnet_152(num_classes):
    return ResNet(BottleNeck, [3, 8, 36, 3], num_classes)


"""
Network for Wide-ResNet.
Code: https://github.com/meliketoy/wide-resnet.pytorch
"""


def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


class WideBasicBlock(nn.Module):

    def __init__(self, in_planes, planes, dropout_rate, stride=1):
        super(WideBasicBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, padding=1, bias=False)
        self.dropout = nn.Dropout(p=dropout_rate)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride, bias=False),
            )

    def forward(self, x):
        out = self.dropout(self.conv1(F.relu(self.bn1(x))))
        out = self.conv2(F.relu(self.bn2(out)))
        out += self.shortcut(x)

        return out


class WideResNet(nn.Module):

    def __init__(self, depth, widen_factor, dropout_rate, num_classes):
        super(WideResNet, self).__init__()
        self.in_planes = 16

        assert (int((depth - 4) % 6) == 0), 'Wide-resnet depth should be 6n+4'
        n = int((depth - 4) / 6)
        k = widen_factor

        n_stages = [16, 16 * k, 32 * k, 64 * k]

        self.conv1 = conv3x3(3, n_stages[0])
        self.layer1 = self._wide_layer(WideBasicBlock, n_stages[1], n, dropout_rate, stride=1)
        self.layer2 = self._wide_layer(WideBasicBlock, n_stages[2], n, dropout_rate, stride=2)
        self.layer3 = self._wide_layer(WideBasicBlock, n_stages[3], n, dropout_rate, stride=2)
        self.bn1 = nn.BatchNorm2d(n_stages[3], momentum=0.9)
        self.linear = nn.Linear(n_stages[3], num_classes)

    def _wide_layer(self, block, planes, num_blocks, dropout_rate, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []

        for stride in strides:
            layers.append(block(self.in_planes, planes, dropout_rate, stride))
            self.in_planes = planes

        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.relu(self.bn1(out))
        out = F.avg_pool2d(out, 8)
        out = out.view(out.size(0), -1)
        out = self.linear(out)

        return out


def wide_resnet_28_10(num_classes, dropout=0.0):
    return WideResNet(28, 10, dropout, num_classes)


def wide_resnet_28_20(num_classes, dropout=0.0):
    return WideResNet(28, 20, dropout, num_classes)


class LeNet(nn.Module):

    def __init__(self, num_classes, num_channels=1):
        super(LeNet, self).__init__()
        if num_channels == 1:
            self.conv1 = nn.Conv2d(1, 20, 5)
            self.conv2 = nn.Conv2d(20, 50, 5)
            self.fc1 = nn.Linear(4 * 4 * 50, 120)
            self.fc2 = nn.Linear(120, 84)
            self.fc3 = nn.Linear(84, num_classes)
        else:
            self.conv1 = nn.Conv2d(3, 6, 5)
            self.conv2 = nn.Conv2d(6, 16, 5)
            self.fc1 = nn.Linear(16 * 5 * 5, 120)
            self.fc2 = nn.Linear(120, 84)
            self.fc3 = nn.Linear(84, num_classes)

    def forward(self, x):
        out = F.relu(self.conv1(x))
        out = F.max_pool2d(out, 2)
        out = F.relu(self.conv2(out))
        out = F.max_pool2d(out, 2)
        out = out.view(out.size(0), -1)
        out = F.relu(self.fc1(out))
        out = F.relu(self.fc2(out))
        out = self.fc3(out)
        return out


def lenet_5(num_classes, num_channels):
    return LeNet(num_classes, num_channels)


class MLP(nn.Module):

    def __init__(self, num_classes, n_1, n_2):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(784, n_1)
        self.fc2 = nn.Linear(n_1, n_2)
        self.fc3 = nn.Linear(n_2, num_classes)

    def forward(self, din):
        din = din.view(-1, 28 * 28)
        dout = F.relu(self.fc1(din))
        dout = F.relu(self.fc2(dout))
        return self.fc3(dout)


def mlp(num_classes=10, n_1=512, n_2=256):
    return MLP(num_classes, n_1, n_2)


class DenseNetBottleNeck(nn.Module):

    def __init__(self, n_channels, growth_rate):
        super(DenseNetBottleNeck, self).__init__()
        inter_channels = 4 * growth_rate
        self.bn1 = nn.BatchNorm2d(n_channels)
        self.conv1 = nn.Conv2d(n_channels, inter_channels, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(inter_channels)
        self.conv2 = nn.Conv2d(inter_channels, growth_rate, kernel_size=3, padding=1, bias=False)

    def forward(self, x):
        out = self.conv1(F.relu(self.bn1(x)))
        out = self.conv2(F.relu(self.bn2(out)))
        out = torch.cat((x, out), 1)
        return out


class SingleLayer(nn.Module):

    def __init__(self, n_channels, growth_rate):
        super(SingleLayer, self).__init__()
        self.bn1 = nn.BatchNorm2d(n_channels)
        self.conv1 = nn.Conv2d(n_channels, growth_rate, kernel_size=3, padding=1, bias=False)

    def forward(self, x):
        out = self.conv1(F.relu(self.bn1(x)))
        out = torch.cat((x, out), 1)
        return out


class Transition(nn.Module):

    def __init__(self, n_channels, n_out_channels):
        super(Transition, self).__init__()
        self.bn1 = nn.BatchNorm2d(n_channels)
        self.conv1 = nn.Conv2d(n_channels, n_out_channels, kernel_size=1, bias=False)

    def forward(self, x):
        out = self.conv1(F.relu(self.bn1(x)))
        out = F.avg_pool2d(out, 2)
        return out


class DenseNet(nn.Module):

    def __init__(self, growth_rate, depth, reduction, num_classes, bottleneck):
        super(DenseNet, self).__init__()

        n_dense_blocks = (depth - 4) // 3
        if bottleneck:
            n_dense_blocks //= 2

        n_channels = 2 * growth_rate
        self.conv1 = nn.Conv2d(3, n_channels, kernel_size=3, padding=1, bias=False)
        self.dense1 = self._make_dense(n_channels, growth_rate, n_dense_blocks, bottleneck)
        n_channels += n_dense_blocks * growth_rate
        n_out_channels = int(math.floor(n_channels * reduction))
        self.trans1 = Transition(n_channels, n_out_channels)

        n_channels = n_out_channels
        self.dense2 = self._make_dense(n_channels, growth_rate, n_dense_blocks, bottleneck)
        n_channels += n_dense_blocks * growth_rate
        n_out_channels = int(math.floor(n_channels * reduction))
        self.trans2 = Transition(n_channels, n_out_channels)

        n_channels = n_out_channels
        self.dense3 = self._make_dense(n_channels, growth_rate, n_dense_blocks, bottleneck)
        n_channels += n_dense_blocks * growth_rate

        self.bn1 = nn.BatchNorm2d(n_channels)
        self.fc = nn.Linear(n_channels, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()

    @staticmethod
    def _make_dense(n_channels, growth_rate, n_dense_blocks, bottleneck):
        layers = []
        for i in range(int(n_dense_blocks)):
            if bottleneck:
                layers.append(DenseNetBottleNeck(n_channels, growth_rate))
            else:
                layers.append(SingleLayer(n_channels, growth_rate))
            n_channels += growth_rate
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.trans1(self.dense1(out))
        out = self.trans2(self.dense2(out))
        out = self.dense3(out)
        out = F.avg_pool2d(F.relu(self.bn1(out)), 8)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out


def densenet_bc_100(num_classes):
    return DenseNet(growth_rate=12, depth=100, reduction=0.5, bottleneck=True, num_classes=num_classes)


cfg = {
    'VGG11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'VGG19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}


class VGG(nn.Module):

    def __init__(self, vgg_name, num_classes):
        super(VGG, self).__init__()
        self.features = self._make_layers(cfg[vgg_name])
        self.dropout = nn.Dropout(0.5)
        self.classifier = nn.Linear(512, num_classes)

    def forward(self, x):
        out = self.features(x)
        out = out.view(out.size(0), -1)
        #out = self.dropout(out)
        out = self.classifier(out)
        return out

    @staticmethod
    def _make_layers(cfg):
        layers = []
        in_channels = 3
        for x in cfg:
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1),
                           nn.BatchNorm2d(x),
                           nn.ReLU(inplace=True)]
                in_channels = x
        layers += [nn.AvgPool2d(kernel_size=1, stride=1)]
        return nn.Sequential(*layers)


def vgg_11(num_classes):
    return VGG('VGG11', num_classes)


def get_network(network_name, num_classes, num_channels=1):
    if network_name == 'wide_resnet':
        return wide_resnet_28_10(num_classes)
    elif network_name == 'resnet':
        return resnet_18(num_classes)
    elif network_name == 'lenet':
        return lenet_5(num_classes, num_channels)
    elif network_name == 'mlp':
        return mlp(num_classes)
    elif network_name == 'densenet':
        return densenet_bc_100(num_classes)
    elif network_name == 'vgg':
        return vgg_11(num_classes)
    else:
        raise Exception('network is not supported')
