import numpy as np
import torch
import torch.nn as nn
from functools import reduce

def find_the_max_depth(num):
    i = 0
    while (num % 2) == 0:
        i += 1
        num >>= 1
    return i


class DoubleConv(nn.Module):
    def __init__(self, in_channels,  out_channels, dr_p=0.0, res=False):
        super(DoubleConv, self).__init__()
        self.res = res
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(out_channels),
            nn.Dropout2d(dr_p),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(out_channels),
            nn.Dropout2d(dr_p)
        )
        self.skip = nn.Conv2d(in_channels, out_channels, 1)

    def forward(self, x):
        c = self.double_conv(x)
        if self.res:
            c += self.skip(x)

        return c


class Down2D(nn.Module):
    def __init__(self, in_channels,  out_channels, dr_p=0.0):
        super(Down2D, self).__init__()
        self.conv = nn.Sequential(nn.MaxPool2d((2,2)),
            DoubleConv(in_channels, out_channels, dr_p, res=True))

    def forward(self, c):
        c = self.conv(c)

        return c


class ConvLayer(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dr_p=0.0):
        super(ConvLayer, self).__init__()
        self.conv = nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size, padding=1),
                                  nn.ReLU(),
                                  nn.BatchNorm2d(out_channels),
                                  nn.Dropout2d(dr_p))

    def forward(self, x):
        out = self.conv(x)
        return out


class Up2D(nn.Module):
    def __init__(self, skip_channels, in_channels, out_channels, dr_p=0.0, attr=False):
        super(Up2D, self).__init__()
        self.attr = attr
        # ops with no parameters
        self.up_sampling = nn.Upsample(scale_factor=2, mode='nearest')
        # self.up_sampling = nn.ConvTranspose2d(in_channels, in_channels, 2, stride=2)

        # ops with parameters
        self.conv = ConvLayer(in_channels, out_channels, 3, 0.0)
        self.double_conv = DoubleConv(out_channels + skip_channels, out_channels, dr_p, res=True)
        if attr:
            self.att = Attention_block_2D(out_channels, skip_channels, skip_channels)

    def forward(self, c, e):
        # up-sampling and merge
        e = self.up_sampling(e)
        e = self.conv(e)
        if self.attr:
            att = self.att(g=e, x=c)
            x = torch.cat((att, e), dim=1)
        else:
            x = torch.cat((c, e), dim=1)

        x = self.double_conv(x)

        return x

class Attention_block_2D(nn.Module):
    """
    Attention Block
    """

    def __init__(self, F_g, F_l, F_int):
        super(Attention_block_2D, self).__init__()

        self.W_g = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )

        self.W_x = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )

        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        out = x * psi

        return out



class UNet_2D(nn.Module):
    def __init__(self, dr=0.25):
        super(UNet_2D, self).__init__()
        self.depth = 3
        self.Img_shape = [720, 1280]
        self.input_channel = 3
        self.output_channel = 1
        self.initial_num_filter = 32
        # self.output_channel = config['style1_channel']
        # self.style2_channel = config['mask_shape'][-1] - self.output_channel

        max_downsampling_num = np.min(
            [find_the_max_depth(self.Img_shape[0]), find_the_max_depth(self.Img_shape[1])])

        if self.depth > max_downsampling_num:
            self.depth = max_downsampling_num
            print('Specified number of layers exceeds the divisibility of image. Reducing the depth to {}'.format(
                max_downsampling_num))

        self.filters_num = [2 ** n * self.initial_num_filter for n in range(self.depth+1)]
        # self.filters_num.append(self.filters_num[-1])
        print('filters:', self.filters_num)
        self.inconv = DoubleConv(self.input_channel, int(self.filters_num[0]), dr_p=0.0)
        self.down = nn.ModuleList()
        self.up = nn.ModuleList()


        for layer_num in range(0, self.depth):
            dr_p = dr * (layer_num+1) / (2.0 * self.depth)
            self.down.append(Down2D(self.filters_num[layer_num], self.filters_num[layer_num+1], dr_p=dr_p))

            up_feat = self.filters_num[self.depth - 1 - layer_num]
            in_feat = self.filters_num[self.depth - layer_num]
            dr_up = dr * (self.depth - layer_num) / (2*self.depth)
            self.up.append(Up2D(up_feat, in_feat, up_feat, dr_up, attr=False))

        self.outconv = nn.Conv2d(self.filters_num[0], self.output_channel, (1, 1))
        self.final1 = nn.Sequential(nn.Conv2d(self.output_channel, self.output_channel, 3, padding=1), nn.Sigmoid())

    def forward(self, x):
        down = {}
        up = {}
        depth = self.depth
        down[0] = self.inconv(x)
        # down[0] = torch.cat((self.inconv(x[:, :3]), self.inconv(x[:, 3:])), 1)
        for i in range(depth):
            down[i + 1] = self.down[i](down[i])
        up[0] = down[depth]
        for i in range(depth):
            up[i + 1] = self.up[i](down[depth - (i + 1)], up[i])
        x = self.outconv(up[depth])
        # output = torch.cat([self.final1(x), self.final2(x)], 1)
        output = self.final1(x)

        return output
