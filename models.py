import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
from torchsummary import summary
import utils
import torch.utils.tensorboard as tb

cv = partial(nn.Conv2d, bias=False)
bn = nn.BatchNorm2d
relu = nn.ReLU


class S7MNIST(nn.Module):
    def __init__(self):
        super().__init__()
        self.dropout = 0.05
        # Input Block
        self.convblock1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=12, kernel_size=(3, 3), padding=1, bias=False),
            nn.BatchNorm2d(12),
            nn.ReLU(),
            nn.Dropout(self.dropout))  # input = 28, output = 28, RF = 3

        # CONVOLUTION BLOCK 1
        self.convblock2 = nn.Sequential(
            nn.Conv2d(12, 16, 3, padding=0, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Dropout(self.dropout))  # input = 28, output = 26, RF = 5

        # TRANSITION BLOCK 1, let's have a mix of channels without extracting features here
        self.convblock3 = nn.Sequential(
            nn.Conv2d(16, 10, 1, padding=0, bias=False),
            nn.ReLU())  # input = 28, output = 26, RF = 5

        self.pool1 = nn.MaxPool2d(2, 2)  # input = 26, output = 13, RF = 10

        self.convblock4 = nn.Sequential(
            nn.Conv2d(10, 16, 3, padding=0, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.Dropout(self.dropout))  # input = 13, output = 11, RF = 12

        self.convblock5 = nn.Sequential(
            nn.Conv2d(16, 16, 3, padding=0, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.Dropout(self.dropout))  # input = 11, output = 9, RF = 14

        # Transition via 1x1 to reduce params and allow selection of relevant channels for next 3x3 layer to extract features
        self.convblock6 = nn.Sequential(
            nn.Conv2d(16, 10, 1, padding=0, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(10),
            nn.Dropout(self.dropout))  # input = 9, output = 9, RF = 14

        self.convblock7 = nn.Sequential(
            nn.Conv2d(10, 16, 3, padding=0, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.Dropout(self.dropout))  # input = 9, output = 7, RF = 16

        self.avg = nn.AvgPool2d(
            7)  # Average Pool layer to reduce dimensions and have a larger view for incoming dimensions to make a decision

        # Final layer with 1x1 to have 10 output channels
        self.convblock8 = nn.Sequential(
            nn.Conv2d(16, 10, 1, padding=0, bias=False))

    def forward(self, x):
        x = self.convblock1(x)
        x = self.convblock2(x)
        x = self.convblock3(x)
        x = self.pool1(x)
        x = self.convblock4(x)
        x = self.convblock5(x)
        x = self.convblock6(x)
        x = self.convblock7(x)
        x = self.avg(x)
        x = self.convblock8(x)
        x = x.view(-1, 10)
        return F.log_softmax(x, dim=-1)


class S9CIFAR10(nn.Module):
    def __init__(self):
        super().__init__()

        self.block1 = nn.Sequential(
            cv(3, 16, 3, padding=1),
            relu(),
            bn(16),
            cv(16, 16, 1),  # 1x1
            relu(),
            bn(16),
            cv(16, 32, 3, dilation=1, groups=1, padding=1, stride=1),
            relu(),
            bn(32),
            cv(32, 32, 3, dilation=1, groups=1, padding=1, stride=2),
            relu(),
            bn(32)
            # nn.Dropout2d(0.05)
        )

        self.block2 = nn.Sequential(
            cv(32, 64, 3, padding=1, dilation=1, groups=32),
            relu(),
            bn(64),
            cv(64, 32, 1),  # 1x1
            relu(),
            bn(32),
            cv(32, 64, 3, dilation=1, padding=1, groups=16),
            relu(),
            bn(64),
            cv(64, 48, 1),  # 1x1
            relu(),
            bn(48),
            cv(48, 64, 3, padding=1, dilation=1, groups=16, stride=2),
            relu(),
            bn(64),
            cv(64, 32, 1),  # 1x1
            relu(),
            bn(32)
        )

        self.block3 = nn.Sequential(
            cv(32, 64, 3, dilation=1, padding=1, groups=32, stride=1),
            relu(),
            bn(64),
            cv(64, 128, 1, groups=16, dilation=1, padding=0),
            relu(),
            bn(128),
            cv(128, 64, 1),  # 1x1
            relu(),
            bn(64),
            cv(64, 96, 3, padding=1, groups=32),
            relu(),
            bn(96),
            cv(96, 64, 1),  # 1x1
            relu(),
            bn(64),
            cv(64, 64, 3, padding=2, dilation=1, stride=2),
            relu(),
            bn(64)
        )

        self.block4 = nn.Sequential(
            cv(64, 96, 3, padding=1, groups=32, stride=1, dilation=1),
            bn(96),
            relu(),
            cv(96, 64, 1),  # 1x1
            bn(64),
            relu(),
            # nn.Dropout2d(0.05),
            cv(64, 64, 3, groups=64, padding=1, dilation=1),  # depthwise (a)
            cv(64, 32, 1),  # pointwise for preceding depthwise (b)
            bn(32),
            relu(),
            cv(32, 48, 3, dilation=2, padding=1, groups=8),
            relu(),
            bn(48),
            cv(48, 10, 1, stride=1),  # 1x1
            relu(),
            bn(10),
            nn.AdaptiveAvgPool2d(1)
        )

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)

        x = x.view(-1, 10)
        return F.log_softmax(x, dim=-1)


conv3x3 = partial(nn.Conv2d, bias=False)
bn_momentum = 0.3


class S10CustomResNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.prep = nn.Sequential(
            conv3x3(3, 64, 3, padding=1),
            bn(64, momentum=bn_momentum, ),
            nn.ReLU())

        self.layer1 = nn.Sequential(
            conv3x3(64, 128, 3, padding=1),
            nn.MaxPool2d(2, 2),
            bn(128, momentum=bn_momentum),
            nn.ReLU())

        self.res1 = nn.Sequential(
            conv3x3(128, 128, 3, padding=1),
            bn(128, momentum=bn_momentum),
            nn.ReLU(),
            conv3x3(128, 128, 3, padding=1),
            bn(128, momentum=bn_momentum),
            nn.ReLU()
        )

        self.layer2 = nn.Sequential(
            conv3x3(128, 256, 3, padding=1),
            nn.MaxPool2d(2, 2),
            bn(256, momentum=bn_momentum),
            nn.ReLU()
        )

        self.layer3 = nn.Sequential(
            conv3x3(256, 512, 3, padding=1),
            nn.MaxPool2d(2, 2),
            bn(512, momentum=bn_momentum),
            nn.ReLU()
        )

        self.res3 = nn.Sequential(
            conv3x3(512, 512, 3, padding=1),
            bn(512, momentum=bn_momentum),
            nn.ReLU(),
            conv3x3(512, 512, 3, padding=1),
            bn(512, momentum=bn_momentum),
            nn.ReLU()
        )

        self.final_max = nn.MaxPool2d(4)
        self.fc = nn.Linear(in_features=512, out_features=10, bias=False)

    def forward(self, x):
        prep = self.prep(x)

        layer1 = self.layer1(prep)
        res1 = self.res1(layer1)
        layer1 = layer1 + res1

        layer2 = self.layer2(layer1)
        layer3 = self.layer3(layer2)
        res3 = self.res3(layer3)
        layer3 = layer3 + res3

        max = self.final_max(layer3)
        out = max.view(max.size(0), -1)

        fc = self.fc(out)

        out = fc.view(-1, 10)

        return out


def test_s9_cifar10():
    model = S9CIFAR10()
    x = torch.randn(1, 3, 32, 32)
    y = model(x)
    print(y.size())
    print(summary(model, (3, 32, 32)))


def test_s10_cifar10():
    model = S10CustomResNet().to(utils.get_device())

    print(summary(model, (3, 32, 32)))

    # Save the model's graph to TensorBoard
    writer = tb.SummaryWriter()

    dummy_input = torch.rand(1, 3, 32, 32)

    writer.add_graph(model, dummy_input)
    # Close the writer
    writer.close()


if __name__ == '__main__':
    # test_s9_cifar10()
    test_s10_cifar10()
