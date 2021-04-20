import torch
import torch.nn as nn # all the neural network modules, nn.Linear, nn.conv2D(spatial convolutions), BatchNorm, Loss functions
import os
import sys


class googleNet(nn.Module):
    def __init__(self, in_channels=3, num_classes=1000):
        super(googleNet, self).__init__()
        self.conv1 = convBlock(in_channels=in_channels, out_channels=64, kernel_size=7, stride=2, padding=3)
        self.maxPool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.conv2 = convBlock(64, 192, kernel_size=3, stride=1, padding=1)
        self.maxPool2 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # In this order: in_channels, out_1x1, red_3x3, out_3x3, red_5x5, out_5x5, out_1x1pool
        self.inception3a = inceptionBlock(192, 64, 96, 128, 16, 32, 32)
        self.inception3b = inceptionBlock(256, 128, 128, 192, 32, 96, 64)
        self.maxPool3 = nn.MaxPool2d(kernel_size=(3, 3), stride=2, padding=1)

        self.inception4a = inceptionBlock(480, 192, 96, 208, 16, 48, 64)
        self.inception4b = inceptionBlock(512, 160, 112, 224, 24, 64, 64)
        self.inception4c = inceptionBlock(512, 128, 128, 256, 24, 64, 64)
        self.inception4d = inceptionBlock(512, 112, 144, 288, 32, 64, 64)
        self.inception4e = inceptionBlock(528, 256, 160, 320, 32, 128, 128)
        self.maxPool4 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.inception5a = inceptionBlock(832, 256, 160, 320, 32, 128, 128)
        self.inception5b = inceptionBlock(832, 384, 192, 384, 48, 128, 128)
        self.avgPool = nn.AvgPool2d(kernel_size=7, stride=1)
        self.dropout = nn.Dropout(p=0.4)
        self.fc1 = nn.Linear(1024, 1000)


    def forward(self, x):
        x = self.conv1(x)
        x = self.maxPool1(x)
        x = self.conv2(x)
        x = self.maxPool2(x)

        x = self.inception3a(x)
        x = self.inception3b(x)
        x = self.maxPool3(x)

        x = self.inception4a(x)
        x = self.inception4b(x)
        x = self.inception4c(x)
        x = self.inception4d(x)
        x = self.inception4e(x)
        x = self.maxPool4(x)

        x = self.inception5a(x)
        x = self.inception5b(x)
        x = self.avgPool(x)
        x = x.reshape(x.shape[0], -1)
        x = self.dropout(x)
        x = self.fc1(x)
        return x












class inceptionBlock(nn.Module):
    def __init__(self, in_channels, out_1x1, red_3x3, out_3x3, red_5x5, out_5x5, out_1x1pool):
        super(inceptionBlock, self).__init__()
        self.branch1 = convBlock(in_channels, out_1x1, kernel_size=1)
        self.branch2 = nn.Sequential(
            convBlock(in_channels, red_3x3, kernel_size=1),
            convBlock(red_3x3, out_3x3, kernel_size=3, stride=1, padding=1)
        )
        self.branch3 = nn.Sequential(
            convBlock(in_channels, red_5x5, kernel_size=1),
            convBlock(red_5x5, out_5x5, kernel_size=5, stride=1, padding=2)
        )
        self.branch4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            convBlock(in_channels, out_1x1pool, kernel_size=1)
        )

    def forward(self, x):
        # number of images *  filters * 28 * 28
        return torch.cat([self.branch1(x), self.branch2(x), self.branch3(x), self.branch4(x)], 1)






class convBlock(nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        super(convBlock, self).__init__()
        self.relu = nn.ReLU()
        self.conv = nn.Conv2d(in_channels, out_channels, **kwargs)
        self.batchNorm = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        return self.relu(self.batchNorm(self.conv(x)))



if __name__ == "__main__":
    x = torch.randn(3, 3, 224, 224)
    model = googleNet()
    output_shape = model(x).shape
    print("Output shape is ", output_shape)


