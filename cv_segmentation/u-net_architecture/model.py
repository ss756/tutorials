'''
Image Segmentation implementation from scratch
<suyash@subtl.in>

Theory:


'''

import torch
import torch.nn as nn
import torchvision.transforms.functional as TF


class DoubleConv(nn.Module):

    '''

    The function applies two convolutional layers each followed by a
    ReLU activation function
    :param in_channels: number of input channels
    :param out_channels: number of output channels
    :return: a down-conv layer

    '''

    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)


class NET(nn.Module):
    def __init__(self, in_channels=3, out_channels=1, features=[64, 128, 256, 512]):
        super(NET, self).__init__()
        print("Feature list is ", features)
        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        # symmetric contracting path to learn about the features
        for feature in features:
            self.downs.append(DoubleConv(in_channels, feature))
            in_channels = feature

        # somewhat symmetric to the contractive path, helps us to understand the precise localization of the object for segmentations tasks

        for feature in reversed(features):
            self.ups.append(
                nn.ConvTranspose2d(
                    feature*2, feature, kernel_size=2, stride=2,
                )
            )
            self.ups.append(DoubleConv(feature*2, feature))
        print("Model Architecture for contractive path is ", self.downs)
        print("Model architecture for expanding path is ", self.ups)

        self.bottleneck = DoubleConv(features[-1], features[-1]*2)
        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)

    def forward(self, x):
        skip_connections = []

        for down in self.downs:
                x = down(x)
                skip_connections.append(x)
                x = self.pool(x)
        x = self.bottleneck(x)
        skip_connections = skip_connections[::-1]
        for idx in range(0, len(self.ups), 2):
            x = self.ups[idx](x)
            skip_connection = skip_connections[idx//2]
            if x.shape != skip_connection.shape:
                x = TF.resize(x, size=skip_connection.shape[2:])  # do not consider the batch size and the number of channels in the image
            concat_skip = torch.cat((skip_connection, x), dim=1)
            x = self.ups[idx+1](concat_skip)
        return self.final_conv(x)









def test():
    x = torch.randn((3, 1, 160, 160))
    model = NET(in_channels=1, out_channels=1)
    preds = model(x)
    assert preds.shape == x.shape
    print("The predicted shape is ", preds.shape)
    print("The input shape is ", x.shape)


if __name__ == "__main__":
    test()
















