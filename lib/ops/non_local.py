# Non-local block using embedded gaussian
# Code from
# https://github.com/AlexHex7/Non-local_pytorch/blob/master/Non-Local_pytorch_0.3.1/lib/non_local_embedded_gaussian.py
import torch
from torch import nn
from torch.nn import functional as F

import lib.model.resnet as lib_resnet

from collections import OrderedDict


# class STA(nn.Module):
#     """Self Temporal Attention"""
#
#     def __init__(self, in_channels, bn_layer=True):
#         super(STA, self).__init__()
#         self.in_channels = in_channels
#         assert self.in_channels % 16 == 0
#
#         self.inter_channels = self.in_channels // 16
#         self.reduce_dim = nn.Conv2d(self.in_channels, self.inter_channels,
#                                     kernel_size=1, stride=1, padding=0)
#         self.restore_dim = nn.Sequential(OrderedDict([
#             ("conv", nn.Conv2d(self.inter_channels, self.in_channels,
#                                kernel_size=1, stride=1, padding=0))
#         ]))
#         if bn_layer:
#             self.restore_dim.add_module("bn", nn.BatchNorm2d(self.in_channels))
#
#         nn.init.constant_(self.restore_dim[-1].weight, 0)
#         nn.init.constant_(self.restore_dim[-1].bias, 0)
#         self.relu = nn.ReLU(inplace=True)
#
#         self.pooling = nn.AdaptiveAvgPool2d(1)
#
#     def forward(self, x):
#         """
#         Args:
#             x: (b, t, c, h, w)
#             return: (b, t, c, h, w)
#         """
#         b, t, c, h, w = x.size()
#         reduce_x = self.reduce_dim(x.view(-1,c,h,w))
#         reduce_c = reduce_x.size(1)
#         pooled_x = self.pooling(x.view(-1, c, h, w)).view(-1, c).view(-1, t, c)  # (bt c) to (b t c)
#         att_map = pooled_x.matmul(pooled_x.permute(0, 2, 1))  # (b t t)
#         att_map = F.softmax(att_map, dim=-1)
#         y = att_map.matmul(reduce_x.view(b, t, -1))  # (b t c*h*w)
#         y = y.view(b, t, reduce_c, h, w)
#
#         y = y.view(-1, reduce_c, h, w).contiguous()  # (bt, c, h, w)
#         y = self.restore_dim(y).view(b, t, c, h, w)
#         y = self.relu(y + x)
#
#         return y

class STA(nn.Module):
    def __init__(self, in_channels, inter_channels=None, dimension=3, sub_sample=False, bn_layer=True):
        super(STA, self).__init__()

        assert dimension in [1, 2, 3]

        self.dimension = dimension
        self.sub_sample = sub_sample

        self.in_channels = in_channels
        self.inter_channels = inter_channels

        if self.inter_channels is None:
            self.inter_channels = in_channels // 2
            if self.inter_channels == 0:
                self.inter_channels = 1

        if dimension == 3:
            conv_nd = nn.Conv3d
            max_pool_layer = nn.MaxPool3d(kernel_size=(1, 2, 2))
            bn = nn.BatchNorm3d
            self.avg_pool_layer = nn.AdaptiveAvgPool3d((None, 1, 1))
        elif dimension == 2:
            conv_nd = nn.Conv2d
            max_pool_layer = nn.MaxPool2d(kernel_size=(2, 2))
            bn = nn.BatchNorm2d
        else:
            conv_nd = nn.Conv1d
            max_pool_layer = nn.MaxPool1d(kernel_size=(2))
            bn = nn.BatchNorm1d

        self.g = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels,
                         kernel_size=1, stride=1, padding=0)

        if bn_layer:
            self.W = nn.Sequential(
                conv_nd(in_channels=self.inter_channels, out_channels=self.in_channels,
                        kernel_size=1, stride=1, padding=0),
                bn(self.in_channels)
            )
            nn.init.constant_(self.W[1].weight, 0)
            nn.init.constant_(self.W[1].bias, 0)
        else:
            self.W = conv_nd(in_channels=self.inter_channels, out_channels=self.in_channels,
                             kernel_size=1, stride=1, padding=0)
            nn.init.constant_(self.W.weight, 0)
            nn.init.constant_(self.W.bias, 0)

        self.theta = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels,
                             kernel_size=1, stride=1, padding=0)
        self.phi = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels,
                           kernel_size=1, stride=1, padding=0)

        if sub_sample:
            self.g = nn.Sequential(self.g, max_pool_layer)
            self.phi = nn.Sequential(self.phi, max_pool_layer)

    def forward(self, x):
        '''
        :param x: (b, c, t, h, w)
        :return:
        '''

        batch_size = x.size(0)
        n_segment = x.size(2)
        # g
        g_x = self.g(x)
        g_x = g_x.permute(0, 2, 1, 3, 4).contiguous().view(batch_size, n_segment, -1)  # (b t chw)
        # theta
        theta_x = self.avg_pool_layer(self.theta(x)).view(batch_size, self.inter_channels, n_segment)  # (b c t)
        theta_x = theta_x.permute(0, 2, 1)  # (b t c)
        # phi
        phi_x = self.avg_pool_layer(self.phi(x)).view(batch_size, self.inter_channels, n_segment)  # (b c t)

        f = torch.matmul(theta_x, phi_x)  # (b t t)
        f_div_C = F.softmax(f, dim=-1)

        y = torch.matmul(f_div_C, g_x)  # (b t chw)
        y = y.view(batch_size, n_segment, self.inter_channels,
                   *x.size()[3:]).transpose(1, 2).contiguous()  # (b c t h w)

        W_y = self.W(y)
        z = W_y + x
        return z


class _NonLocalBlockND(nn.Module):
    def __init__(self, in_channels, inter_channels=None, dimension=3, sub_sample=False, bn_layer=True):
        super(_NonLocalBlockND, self).__init__()

        assert dimension in [1, 2, 3]

        self.dimension = dimension
        self.sub_sample = sub_sample

        self.in_channels = in_channels
        self.inter_channels = inter_channels

        if self.inter_channels is None:
            self.inter_channels = in_channels // 2
            if self.inter_channels == 0:
                self.inter_channels = 1

        if dimension == 3:
            conv_nd = nn.Conv3d
            max_pool_layer = nn.MaxPool3d(kernel_size=(1, 2, 2))
            bn = nn.BatchNorm3d
        elif dimension == 2:
            conv_nd = nn.Conv2d
            max_pool_layer = nn.MaxPool2d(kernel_size=(2, 2))
            bn = nn.BatchNorm2d
        else:
            conv_nd = nn.Conv1d
            max_pool_layer = nn.MaxPool1d(kernel_size=(2))
            bn = nn.BatchNorm1d

        self.g = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels,
                         kernel_size=1, stride=1, padding=0)

        if bn_layer:
            self.W = nn.Sequential(
                conv_nd(in_channels=self.inter_channels, out_channels=self.in_channels,
                        kernel_size=1, stride=1, padding=0),
                bn(self.in_channels)
            )
            nn.init.constant_(self.W[1].weight, 0)
            nn.init.constant_(self.W[1].bias, 0)
        else:
            self.W = conv_nd(in_channels=self.inter_channels, out_channels=self.in_channels,
                             kernel_size=1, stride=1, padding=0)
            nn.init.constant_(self.W.weight, 0)
            nn.init.constant_(self.W.bias, 0)

        self.theta = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels,
                             kernel_size=1, stride=1, padding=0)
        self.phi = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels,
                           kernel_size=1, stride=1, padding=0)

        if sub_sample:
            self.g = nn.Sequential(self.g, max_pool_layer)
            self.phi = nn.Sequential(self.phi, max_pool_layer)

    def forward(self, x):
        '''
        :param x: (b, c, t, h, w)
        :return:
        '''

        batch_size = x.size(0)
        # g
        g_x = self.g(x).view(batch_size, self.inter_channels, -1)  # (b c thw)
        g_x = g_x.permute(0, 2, 1)  # (b thw c)
        # theta
        theta_x = self.theta(x).view(batch_size, self.inter_channels, -1)  # (b c thw)
        theta_x = theta_x.permute(0, 2, 1)  # (b thw c)
        # phi
        phi_x = self.phi(x).view(batch_size, self.inter_channels, -1)  # (b c thw)

        f = torch.matmul(theta_x, phi_x)  # (b thw thw)
        f_div_C = F.softmax(f, dim=-1)

        y = torch.matmul(f_div_C, g_x)  # (b thw c)
        y = y.permute(0, 2, 1).contiguous()  # (b c thw)
        y = y.view(batch_size, self.inter_channels, *x.size()[2:])  # (b c t h w)

        W_y = self.W(y)
        z = W_y + x

        return z


class NONLocalBlock1D(_NonLocalBlockND):
    def __init__(self, in_channels, inter_channels=None, sub_sample=True, bn_layer=True):
        super(NONLocalBlock1D, self).__init__(in_channels,
                                              inter_channels=inter_channels,
                                              dimension=1, sub_sample=sub_sample,
                                              bn_layer=bn_layer)


class NONLocalBlock2D(_NonLocalBlockND):
    def __init__(self, in_channels, inter_channels=None, sub_sample=True, bn_layer=True):
        super(NONLocalBlock2D, self).__init__(in_channels,
                                              inter_channels=inter_channels,
                                              dimension=2, sub_sample=sub_sample,
                                              bn_layer=bn_layer)


class NONLocalBlock3D(_NonLocalBlockND):
    def __init__(self, in_channels, inter_channels=None, sub_sample=True, bn_layer=True):
        super(NONLocalBlock3D, self).__init__(in_channels,
                                              inter_channels=inter_channels,
                                              dimension=3, sub_sample=sub_sample,
                                              bn_layer=bn_layer)


class NL3DWrapper(nn.Module):
    def __init__(self, block, n_segment):
        super(NL3DWrapper, self).__init__()
        self.block = block
        self.nl = NONLocalBlock3D(block.bn3.num_features)
        self.n_segment = n_segment

    def forward(self, x):
        x = self.block(x)

        nt, c, h, w = x.size()
        x = x.view(nt // self.n_segment, self.n_segment, c, h, w).transpose(1, 2)  # n, c, t, h, w
        x = self.nl(x)  # n, c, t, h, w
        x = x.transpose(1, 2).contiguous().view(nt, c, h, w)
        return x


class STAWrapper(nn.Module):
    def __init__(self, block, n_segment):
        super(STAWrapper, self).__init__()
        self.block = block
        self.nl = STA(block.bn3.num_features)
        self.n_segment = n_segment

    def forward(self, x):
        x = self.block(x)

        nt, c, h, w = x.size()
        x = x.view(nt // self.n_segment, self.n_segment, c, h, w).transpose(1, 2)  # n, c, t, h, w
        x = self.nl(x)  # n, c, t, h, w
        x = x.transpose(1, 2).contiguous().view(nt, c, h, w)
        return x


def make_non_local(net, n_segment):
    if isinstance(net, lib_resnet.ResNet):

        # ---------------------------------------------------------
        # STA
        # net.layer2 = nn.Sequential(
        #     STAWrapper(net.layer2[0], n_segment),
        #     net.layer2[1],
        #     STAWrapper(net.layer2[2], n_segment),
        #     net.layer2[3],
        # )
        # net.layer3 = nn.Sequential(
        #     STAWrapper(net.layer3[0], n_segment),
        #     net.layer3[1],
        #     STAWrapper(net.layer3[2], n_segment),
        #     net.layer3[3],
        #     STAWrapper(net.layer3[4], n_segment),
        #     net.layer3[5],
        # )

        # ---------------------------------------------------------
        # Non-local
        net.layer2 = nn.Sequential(
            NL3DWrapper(net.layer2[0], n_segment),
            net.layer2[1],
            NL3DWrapper(net.layer2[2], n_segment),
            net.layer2[3],
        )
        net.layer3 = nn.Sequential(
            NL3DWrapper(net.layer3[0], n_segment),
            net.layer3[1],
            NL3DWrapper(net.layer3[2], n_segment),
            net.layer3[3],
            NL3DWrapper(net.layer3[4], n_segment),
            net.layer3[5],
        )
    else:
        raise NotImplementedError


if __name__ == '__main__':
    from torch.autograd import Variable
    import torch

    sub_sample = True
    bn_layer = True

    img = Variable(torch.zeros(2, 3, 20))
    net = NONLocalBlock1D(3, sub_sample=sub_sample, bn_layer=bn_layer)
    out = net(img)
    print(out.size())

    img = Variable(torch.zeros(2, 3, 20, 20))
    net = NONLocalBlock2D(3, sub_sample=sub_sample, bn_layer=bn_layer)
    out = net(img)
    print(out.size())

    img = Variable(torch.randn(2, 3, 10, 20, 20))
    net = NONLocalBlock3D(3, sub_sample=sub_sample, bn_layer=bn_layer)
    out = net(img)
    print(out.size())
