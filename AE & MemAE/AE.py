import torch
from torch import random
import torch.nn as nn
from memory import MemoryModule

def ConvBnRelu(channel_in, channel_out):
    ''' Conv2D + BatchNorm + LeakyReLU '''
    conv_bn_relu = nn.Sequential(
        nn.Conv2d(channel_in, channel_out, 3, stride=2, padding=1),
        nn.BatchNorm2d(channel_out),
        nn.LeakyReLU(0.2, inplace=True)
    )
    return conv_bn_relu


def DConvBnRelu(channel_in, channel_out):
    ''' DConv2D + BatchNorm + LeakyReLU '''
    d_conv_bn_relu = nn.Sequential(
        nn.ConvTranspose2d(channel_in, channel_out, 3, stride=2, padding=1, output_padding=1),
        nn.BatchNorm2d(channel_out),
        nn.LeakyReLU(0.2, inplace=True)
    )
    return d_conv_bn_relu


class AutoEncoder(nn.Module):
    ''' Vanilla AE '''
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            ConvBnRelu(3, 96),
            ConvBnRelu(96,128),
            ConvBnRelu(128, 256),
            ConvBnRelu(256, 256),
        )
        self.decoder = nn.Sequential(
            DConvBnRelu(256, 256),
            DConvBnRelu(256, 128),
            DConvBnRelu(128,96),
            nn.ConvTranspose2d(96, 3, 3, stride=2, padding=1, output_padding=1)
        )

    def forward(self, x):

        feature = self.encoder(x)
        output = self.decoder(feature)

        return output


class MemoryAutoEncoder(nn.Module):
    ''' MemAE '''
    def __init__(self, mem_dim, shrink_thres=0.0025):
        super().__init__()
        self.encoder = nn.Sequential(
            ConvBnRelu(3, 96),
            ConvBnRelu(96,128),
            ConvBnRelu(128, 256),
            ConvBnRelu(256, 256),
        )
        # Memory Module
        self.mem = MemoryModule(mem_dim=mem_dim, fea_dim=256, shrink_thres=shrink_thres)
        self.decoder = nn.Sequential(
            DConvBnRelu(256, 256),
            DConvBnRelu(256, 128),
            DConvBnRelu(128,96),
            nn.ConvTranspose2d(96, 3, 3, stride=2, padding=1, output_padding=1)
        )

    def forward(self, x):

        feature = self.encoder(x)
        att, feature = self.mem(feature)
        output = self.decoder(feature)

        return att, output


if __name__ == '__main__':
    x = torch.randn(10, 3, 256, 256)
    # model = AutoEncoder()
    model = MemoryAutoEncoder(mem_dim=2000)
    # print(model)
    att, rec = model(x)
    print(att.shape, rec.shape)
    