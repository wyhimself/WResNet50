import torch
import torch.nn as nn
import torch.nn.functional as F

'''
我打算自己写一遍ResNet50
'''

def conv1x1(in_channels, out_channels, stride=1, padding=0):
    '''
    stride=1, padding=0的默认设计不会改变特征图的大小
    stride=2, padding=1 -> 特征图大小减半
    '''
    return nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, padding=padding)

def conv3x3(in_channels, out_channels, stride=1, padding=1):
    '''
    stride=1, padding=1 -> 特征图大小不变
    stride=2, padding=1 -> 特征图大小减半
    '''
    return nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=padding)

class BasicBlock(nn.Module):
    '''
    残差块A: conv3x3 + conv3x3
    全流程中特征图都是64-d
    '''
    def __init__(self, width=64, downsample=False):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(width, width)
        self.bn1 = nn.BatchNorm2d(width)

        self.activate = nn.ReLU()

        self.conv2 = conv3x3(width, width)
        self.bn2 = nn.BatchNorm2d(width)
    
    def forward(self, x):
        temp = x
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.activate(x)
        x = self.conv2(x)
        x = self.bn2(x)
        return self.activate(x + temp)

class Bottleneck(nn.Module):
    '''
    残差块B: conv1x1 + conv3x3 + conv1x1
    输入是256-d，但中间有变化为64-d的情况
    '''
    def __init__(self, width, process, shortcut=0):
        super(Bottleneck, self).__init__()
        self.shortcut = shortcut
        self.width = width
        self.process = process

        self.conv1 = conv1x1(width, process)
        self.bn1 = nn.BatchNorm2d(process)
        
        self.activate = nn.ReLU()

        self.conv2 = conv3x3(process, process)
        self.bn2 = nn.BatchNorm2d(process)

        self.conv3 = conv1x1(process, process * 4)
        self.bn3 = nn.BatchNorm2d(process * 4)

        # 将 shortcut 放到这里作为模块的一部分，才能顺利放到对应的device上
        if shortcut > 0:
            self.shortcut_conv = conv1x1(width, width * shortcut)
        else:
            self.shortcut_conv = None

    def forward(self, x):
        temp = x
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.activate(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.activate(x)
        x = self.conv3(x)
        x = self.bn3(x)
        if self.shortcut_conv is not None:
            temp = self.shortcut_conv(temp)
        return self.activate(x + temp)

class ResNet50(nn.Module):
    def __init__(self, in_channels=3, width=224, out_features=1000):
        super(ResNet50, self).__init__()
        self.in_channels = in_channels
        self.width = width
        self.out_features = out_features

        self.conv1 = nn.Conv2d(in_channels, out_channels=64, kernel_size=7, stride=2, padding=3) # (B, 64, width/2, width/2), 112
        self.bn1 = nn.BatchNorm2d(num_features=64)
        self.activate = nn.ReLU()
        self.maxpool = nn.MaxPool2d(3, 2, 1) # (B, 64, width/4, width/4), 56
        self.conv2_x = nn.Sequential(
            Bottleneck(64, 64, 4),
            *[Bottleneck(256, 64) for _ in range(2)],
        ) # 256
        self.conv3_x = nn.Sequential(
            Bottleneck(256, 128, 2),
            *[Bottleneck(512, 128) for _ in range(3)],
        )# 512
        self.conv4_x = nn.Sequential(
            Bottleneck(512, 256, 2),
            *[Bottleneck(1024, 256) for _ in range(5)],
        ) # 1024
        self.conv5_x = nn.Sequential(
            Bottleneck(1024, 512, 2),
            *[Bottleneck(2048, 512) for _ in range(2)]
        ) # 2048
        self.avgpool= nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(2048, out_features)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.activate(x)

        x = self.maxpool(x)
        x = self.conv2_x(x)
        x = self.conv3_x(x)
        x = self.conv4_x(x)
        x = self.conv5_x(x)
        
        x = self.avgpool(x)
        x = x.flatten(1)
        x = self.fc(x)

        return x