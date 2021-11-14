import torch.nn as nn
import torch.nn.functional as F


class Block(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(Block, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            conv3 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False)
            bn3 = nn.BatchNorm2d(out_channels)
            self.shortcut = nn.Sequential(conv3, bn3)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.block1a = Block(64, 64, 1)
        self.block1b = Block(64, 64, 1)
        self.block2a = Block(64, 128, 2)
        self.block2b = Block(128, 128, 1)
        self.block3a = Block(128, 256, 2)
        self.block3b = Block(256, 256, 1)
        self.block4a = Block(256, 512, 2)
        self.block4b = Block(512, 512, 1)
        self.linear = nn.Linear(512, 10)

    def forward(self, x, include_penultimate=False):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.block1a(out)
        out = self.block1b(out)
        out = self.block2a(out)
        out = self.block2b(out)
        out = self.block3a(out)
        out = self.block3b(out)
        out = self.block4a(out)
        out = self.block4b(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        penultimate = out
        out = self.linear(out)
        if include_penultimate:
            out = (out, penultimate)
        return out
