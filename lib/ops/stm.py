from collections import OrderedDict

import torch
from torch import nn
from torch.nn import functional as F

import lib.model.resnet as lib_resnet


class HierarchicalCSTM(nn.Module):
    """Channel-wise SpatioTemporal Module"""

    def __init__(self, in_channels):
        super(HierarchicalCSTM, self).__init__()
        self.temporal_combination_1 = nn.Conv1d(in_channels, in_channels, 3, stride=1, padding=1)
        self.temporal_combination_2 = nn.Conv1d(in_channels, in_channels, 3, stride=1, padding=1)
        self.temporal_combination_3 = nn.Conv1d(in_channels, in_channels, 3, stride=1, padding=1)

        self.local_spatial = nn.Conv2d(in_channels, in_channels, 3, stride=1, padding=1)

    def forward(self, x):
        """
        Args:
            x: (b, t, c, h, w)
            return: (b, t, c, h, w)
        """

        batch_size, t, c, h, w = x.size()
        transpose_x = x.permute(0, 3, 4, 2, 1).contiguous()  # (b h w c t)
        transpose_x = transpose_x.view(-1, c, t)  # (b*h*w c t)

        # Hierarchical=1
        x_1 = self.temporal_combination_1(transpose_x)  # output (b*h*w c t)
        # Hierarchical=2
        x_2 = self.temporal_combination_2(x_1 + transpose_x)
        # Hierarchical=3
        x_3 = self.temporal_combination_3(x_2 + transpose_x)

        combine = x_1 + x_2 + x_3

        combine = combine.view(batch_size, h, w, c, t).permute(0, 4, 3, 1, 2).contiguous()  # (b, t, c, h, w)
        combine = combine.view(-1, c, h, w)  # (bt, c, h, w)

        x = self.local_spatial(combine)  # (bt, c, h, w)
        x = x.view(batch_size, t, c, h, w)

        return x


class CSTM(nn.Module):
    """Channel-wise SpatioTemporal Module"""

    def __init__(self, in_channels):
        super(CSTM, self).__init__()
        self.temporal_combination = nn.Conv1d(in_channels, in_channels, 3, stride=1, padding=1)
        self.local_spatial = nn.Conv2d(in_channels, in_channels, 3, stride=1, padding=1)

    def forward(self, x):
        """
        Args:
            x: (b, t, c, h, w)
            return: (b, t, c, h, w)
        """

        batch_size, t, c, h, w = x.size()
        transpose_x = x.permute(0, 3, 4, 2, 1).contiguous()  # (b h w c t)
        transpose_x = transpose_x.view(-1, c, t)  # (b*h*w c t)
        x = self.temporal_combination(transpose_x)  # output (b*h*w c t)
        x = x.view(batch_size, h, w, c, t)
        x = x.permute(0, 4, 3, 1, 2).contiguous()  # (b, t, c, h, w)
        x = x.view(-1, c, h, w)
        x = self.local_spatial(x)  # (bt, c, h, w)
        x = x.view(batch_size, t, c, h, w)

        return x


class CMM(nn.Module):
    """Channel-wise Motion Module"""

    def __init__(self, in_channels):
        super(CMM, self).__init__()
        self.channel_wise_conv = nn.Conv2d(in_channels, in_channels, 3, stride=1, padding=1)

    def forward(self, x):
        """
        Args:
            x: (b, t, c, h, w)
            return: (b, t, c, h, w)
        """
        chunks = [item.squeeze(1) for item in x.split(1, dim=1)]
        new_chunks = []
        for idx in range(1, len(chunks)):
            chunk = chunks[idx]  # (b, c, h, w)
            new_chunks.append(self.channel_wise_conv(chunk))  # (b, c, h, w)
        new_chunks.append(chunks[-1])

        chunks = torch.stack(chunks, dim=1)  # (b, t, c, h, w)
        new_chunks = torch.stack(new_chunks, dim=1)  # (b, t, c, h, w)
        motion = new_chunks - chunks

        return motion


class _STM(nn.Module):
    def __init__(self, in_channels, n_segment, cstm=True, cmm=False,
                 reduce_factor=16, bn_layer=True):
        super(_STM, self).__init__()
        assert cstm or cmm

        self.in_channels = in_channels
        self.n_segment = n_segment
        assert self.in_channels % reduce_factor == 0
        self.inter_channels = self.in_channels // reduce_factor
        self.has_cstm = cstm  # Channel-wise SpatioTemporal Module
        self.has_cmm = cmm  # Channel-wise Motion Module
        self.reduce_dim = nn.Conv2d(self.in_channels, self.inter_channels,
                                    kernel_size=1, stride=1, padding=0)
        self.restore_dim = nn.Sequential(OrderedDict([
            ("conv", nn.Conv2d(self.inter_channels, self.in_channels,
                               kernel_size=1, stride=1, padding=0))
        ]))
        if bn_layer:
            self.restore_dim.add_module("bn", nn.BatchNorm2d(self.in_channels))

        nn.init.constant_(self.restore_dim[-1].weight, 0)
        nn.init.constant_(self.restore_dim[-1].bias, 0)
        self.relu = nn.ReLU(inplace=True)

        if self.has_cstm:
            self.cstm = HierarchicalCSTM(self.inter_channels)
        if self.has_cmm:
            self.cmm = CMM(self.inter_channels)

    def forward(self, x):
        """
        Args:
            x: (bt, c, h, w)
            return: (bt, c, h, w)
        """
        bt, c, h, w = x.size()
        reduce_x = self.reduce_dim(x)
        reduce_c = reduce_x.size(1)

        reduce_x = reduce_x.view(bt // self.n_segment, self.n_segment, reduce_c, h, w)  # (b, t, c, h, w)

        residual = None
        if self.has_cstm:
            residual = self.cstm(reduce_x).view(bt, reduce_c, h, w)  # (b, t, c, h, w) to (bt c h w)
        if self.has_cmm:
            cmm_x = self.cmm(reduce_x).view(bt, reduce_c, h, w)  # (b, t, c, h, w) to (bt c h w)
            if residual is not None:
                residual += cmm_x
            else:
                residual = cmm_x
        residual = self.restore_dim(residual)  # (bt c h w)

        x = x + residual
        x = self.relu(x)
        return x


class STMWrapper(nn.Module):
    def __init__(self, block, n_segment, cstm=True, cmm=False):
        super(STMWrapper, self).__init__()
        self.block = block
        self.stm = _STM(block.bn3.num_features, n_segment, cstm=cstm, cmm=cmm)

    def forward(self, x):
        x = self.block(x)  # (nt, c, h, w)
        x = self.stm(x)  # (nt, c, h, w)
        return x


def make_stm(net, n_segment, cstm=True, cmm=False):
    """ Spatiotemporal and Motion (STM) Block"""
    if isinstance(net, lib_resnet.ResNet):
        kwargs = dict(n_segment=n_segment,
                      cstm=cstm, cmm=cmm)
        # ---------------------------------------------------------
        # option-1
        # tmp = [STMWrapper(block, **kwargs) for block in net.layer1]
        # net.layer1 = nn.Sequential(*tmp)
        # tmp = [STMWrapper(block, **kwargs) for block in net.layer2]
        # net.layer2 = nn.Sequential(*tmp)
        # tmp = [STMWrapper(block, **kwargs) for block in net.layer3]
        # net.layer3 = nn.Sequential(*tmp)
        # tmp = [STMWrapper(block, **kwargs) for block in net.layer4]
        # net.layer4 = nn.Sequential(*tmp)

        # ---------------------------------------------------------
        # option-2
        net.layer2 = nn.Sequential(
            STMWrapper(net.layer2[0], **kwargs),
            net.layer2[1],
            STMWrapper(net.layer2[2], **kwargs),
            net.layer2[3],
        )
        net.layer3 = nn.Sequential(
            STMWrapper(net.layer3[0], **kwargs),
            net.layer3[1],
            STMWrapper(net.layer3[2], **kwargs),
            net.layer3[3],
            STMWrapper(net.layer3[4], **kwargs),
            net.layer3[5],
        )
    else:
        raise NotImplementedError
