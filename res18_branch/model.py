# --coding:utf-8--
import torch
import torch.nn as nn
import torch.nn.functional as F


# define resnet structure
class BasicBlock(nn.Module):
    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv3d(in_channels=in_planes, out_channels=planes, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm3d(planes),
            nn.ReLU()
        )

        self.conv2 = nn.Sequential(
            nn.Conv3d(in_channels=planes, out_channels=planes, kernel_size=3, stride=1, padding=1, bias=False),
        )

        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv3d(in_planes, planes, kernel_size=1, stride=stride),
            )
        else:
            self.shortcut = nn.Sequential()

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class ResNet(nn.Module):
    def __init__(self, block, num_block, num_classes):
        super(ResNet, self).__init__()
        self.in_planes = 16
        self.conv1 = nn.Conv3d(1, 16, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool3d(kernel_size=3, stride=1, padding=1)               # the first layer

        self.layer1 = self._make_layer(block, 16, num_block[0], stride=1)             # four layers 2-5
        self.layer2 = self._make_layer(block, 32, num_block[1], stride=2)            # four layers 6-9
        self.layer3 = self._make_layer(block, 64, num_block[2], stride=2)            # four layers 10-13
        self.layer4 = self._make_layer(block, 96, num_block[3], stride=2)            # four layers 14-17

        self.fc = nn.Linear(768, num_classes)                                         # the last layer

    def _make_layer(self, block, planes, num_blocks, stride):
        layers = []
        for i in range(num_blocks):
            if i == 0:
                layers.append(block(self.in_planes, planes, stride))
            else:
                layers.append(block(planes, planes, 1))

        self.in_planes = planes
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.maxpool(self.relu(self.conv1(x)))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = F.avg_pool3d(x, 4)
        x = x.view(x.size(0), -1)
        out = self.fc(x)
        return out


def resnet18():
    model = ResNet(BasicBlock, [1, 1, 1, 1], 1)
    return model

if __name__ == '__main__':
    input = torch.rand((8, 1, 64, 64, 64)).cuda()
    model = resnet18().cuda()
    output = model(input)
    print(output.shape)

