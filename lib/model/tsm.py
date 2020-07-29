# Code for "TSM: Temporal Shift Module for Efficient Video Understanding"
# arXiv:1811.08383
# Ji Lin*, Chuang Gan, Song Han
# {jilin, songhan}@mit.edu, ganchuang@csail.mit.edu
import numpy as np
import torch
from torch import nn
import torchvision
import torch.nn.functional as F

from lib.ops.basic_ops import ConsensusModule
from torch.nn.init import normal_, constant_
import lib.model.resnet as lib_resnet
import lib.model.res2net as lib_res2net


class TSN(nn.Module):
    def __init__(self,
                 num_classes, num_segments, modality,
                 base_model='resnet101', conv5_stride=2, bn=False,
                 pool_type='avg',
                 consensus_type='avg', loss={'xent'},
                 non_local=False, stm=[],
                 is_shift=False, shift_div=8, shift_place='blockres',
                 fc_lr5=False,
                 temporal_pool=False, partial_bn=False, new_length=None, ):
        super(TSN, self).__init__()
        assert not temporal_pool
        assert not partial_bn

        self.modality = modality
        self.num_segments = num_segments
        self.consensus_type = consensus_type
        self.is_shift = is_shift
        self.shift_div = shift_div
        self.shift_place = shift_place
        self.base_model_name = base_model
        self.fc_lr5 = fc_lr5
        self.temporal_pool = temporal_pool
        self.non_local = non_local
        self.stm = stm
        self.loss = loss

        if new_length is None:
            self.new_length = 1 if modality == "RGB" else 5
        else:
            self.new_length = new_length
        print(("""Initializing TSN with base model: {}.
                  TSN Configurations:
                      spatio-temporal-motion:   {}
                      input_modality:           {}
                      num_segments:             {}
                      temporal_consensus:       {}
        """.format(base_model, self.stm, self.modality, self.num_segments, consensus_type)))

        model_kwargs = {"conv5_stride": conv5_stride,
                        "bn": bn,
                        "pool_type": pool_type}

        self._prepare_base_model(base_model, num_classes, **model_kwargs)

        if self.modality == 'Flow':
            print("Converting the ImageNet model to a flow init model")
            self.base_model = self._construct_flow_model(self.base_model)
            print("Done. Flow model ready...")
        elif self.modality == 'RGBDiff':
            print("Converting the ImageNet model to RGB+Diff init model")
            self.base_model = self._construct_diff_model(self.base_model)
            print("Done. RGBDiff model ready.")

        self.consensus = ConsensusModule(consensus_type)

        self._enable_pbn = partial_bn
        if partial_bn:
            self.partialBN(True)

    def _prepare_base_model(self, base_model, num_class, conv5_stride=1, bn=False, pool_type='avg'):
        assert pool_type in ['avg', 'max']

        if 'res' in base_model:
            self.num_features = 2048
            lib_model = lib_res2net if "res2" in base_model else lib_resnet
            self.base_model = getattr(lib_model, base_model)(
                pretrained=True,
                conv5_stride=conv5_stride)

            if self.is_shift:
                print('Adding temporal shift...')
                from lib.ops.temporal_shift import make_temporal_shift
                make_temporal_shift(self.base_model, self.num_segments,
                                    n_div=self.shift_div, place=self.shift_place, temporal_pool=self.temporal_pool)

            if self.non_local:
                print('Adding non-local module...')
                from lib.ops.non_local import make_non_local
                make_non_local(self.base_model, self.num_segments)

            if self.stm:
                print('Adding spatio-temporal module...')
                from lib.ops.stm import make_stm
                cstm = True if 'cstm' in self.stm else False
                cmm = True if 'cmm' in self.stm else False
                make_stm(self.base_model, self.num_segments, cstm=cstm, cmm=cmm)

            conv1 = nn.Sequential(self.base_model.conv1,
                                  self.base_model.bn1,
                                  self.base_model.relu,
                                  self.base_model.maxpool)
            self.base_model = nn.Sequential(conv1,
                                            self.base_model.layer1,
                                            self.base_model.layer2,
                                            self.base_model.layer3,
                                            self.base_model.layer4)
            self.bn = nn.BatchNorm1d(self.num_features) if bn else None
            self.fc = nn.Linear(self.num_features, num_class, bias=False)
            self.pool_layer = nn.AdaptiveAvgPool2d(1) if 'avg' == pool_type else nn.AdaptiveMaxPool2d(1)
            normal_(self.fc.weight, 0, 0.001)
        else:
            raise ValueError('Unknown base model: {}'.format(base_model))

    def train(self, mode=True):
        """
        Override the default train() to freeze the BN parameters
        :return:
        """
        super(TSN, self).train(mode)
        count = 0
        if self._enable_pbn and mode:
            print("Freezing BatchNorm2D except the first one.")
            for m in self.base_model.modules():
                if isinstance(m, nn.BatchNorm2d):
                    count += 1
                    if count >= (2 if self._enable_pbn else 1):
                        m.eval()
                        # shutdown update in frozen mode
                        m.weight.requires_grad = False
                        m.bias.requires_grad = False

    def partialBN(self, enable):
        self._enable_pbn = enable

    def get_optim_policies(self, lr, wd):
        first_conv_weight = []
        first_conv_bias = []
        normal_weight = []
        normal_bias = []
        lr5_weight = []
        lr10_bias = []
        bn = []
        custom_ops = []

        conv_cnt = 0
        bn_cnt = 0
        for m in self.modules():
            if isinstance(m, torch.nn.Conv2d) or isinstance(m, torch.nn.Conv1d) or isinstance(m, torch.nn.Conv3d):
                ps = list(m.parameters())
                conv_cnt += 1
                if conv_cnt == 1:
                    first_conv_weight.append(ps[0])
                    if len(ps) == 2:
                        first_conv_bias.append(ps[1])
                else:
                    normal_weight.append(ps[0])
                    if len(ps) == 2:
                        normal_bias.append(ps[1])
            elif isinstance(m, torch.nn.Linear):
                ps = list(m.parameters())
                if self.fc_lr5:
                    lr5_weight.append(ps[0])
                else:
                    normal_weight.append(ps[0])
                if len(ps) == 2:
                    if self.fc_lr5:
                        lr10_bias.append(ps[1])
                    else:
                        normal_bias.append(ps[1])

            elif isinstance(m, torch.nn.BatchNorm2d):
                bn_cnt += 1
                # later BN's are frozen
                if not self._enable_pbn or bn_cnt == 1:
                    bn.extend(list(m.parameters()))
            elif isinstance(m, torch.nn.BatchNorm3d):
                bn_cnt += 1
                # later BN's are frozen
                if not self._enable_pbn or bn_cnt == 1:
                    bn.extend(list(m.parameters()))
            elif len(m._modules) == 0:
                if len(list(m.parameters())) > 0:
                    raise ValueError("New atomic module type: {}. Need to give it a learning policy".format(type(m)))

        return [
            {'params': first_conv_weight, 'lr': 5 * lr if self.modality == 'Flow' else 1 * lr, 'weight_decay': 1 * wd,
             'name': "first_conv_weight"},
            {'params': first_conv_bias, 'lr': 10 * lr if self.modality == 'Flow' else 2 * lr, 'weight_decay': 0 * wd,
             'name': "first_conv_bias"},
            {'params': normal_weight, 'lr': 1 * lr, 'weight_decay': 1 * wd,
             'name': "normal_weight"},
            {'params': normal_bias, 'lr': 2 * lr, 'weight_decay': 0 * wd,
             'name': "normal_bias"},
            {'params': bn, 'lr': 1 * lr, 'weight_decay': 0 * wd,
             'name': "BN scale/shift"},
            {'params': custom_ops, 'lr': 1 * lr, 'weight_decay': 1 * wd,
             'name': "custom_ops"},
            # for fc
            {'params': lr5_weight, 'lr': 5 * lr, 'weight_decay': 1 * wd,
             'name': "lr5_weight"},
            {'params': lr10_bias, 'lr': 10 * lr, 'weight_decay': 0 * wd,
             'name': "lr10_bias"},
        ]

    def forward(self, input, no_reshape=False):
        b, t = input.size(0), input.size(1)
        input = input.view(b * t, input.size(2), input.size(3), input.size(4))

        if not no_reshape:
            sample_len = (3 if self.modality == "RGB" else 2) * self.new_length

            if self.modality == 'RGBDiff':
                sample_len = 3 * self.new_length
                input = self._get_diff(input)

            base_out = self.base_model(input.view((-1, sample_len) + input.size()[-2:]))
        else:
            base_out = self.base_model(input)

        base_out = self.pool_layer(base_out)
        base_out = base_out.view(base_out.size(0), -1)  # (b*t c)
        if self.bn is not None:
            base_out = self.bn(base_out)
        predict = self.fc(base_out)

        if self.is_shift and self.temporal_pool:
            predict = predict.view((-1, self.num_segments // 2) + predict.size()[1:])
            feature = base_out.view((-1, self.num_segments // 2) + base_out.size()[1:])
        else:
            predict = predict.view((-1, self.num_segments) + predict.size()[1:])  # (b t num_class)
            feature = base_out.view((-1, self.num_segments) + base_out.size()[1:])  # (b t c)

        predict = self.consensus(predict).squeeze(1)
        feature = self.consensus(feature).squeeze(1)

        if not self.training:
            return F.normalize(feature, dim=1)

        return predict, feature

    def _get_diff(self, input, keep_rgb=False):
        input_c = 3 if self.modality in ["RGB", "RGBDiff"] else 2
        input_view = input.view((-1, self.num_segments, self.new_length + 1, input_c,) + input.size()[2:])
        if keep_rgb:
            new_data = input_view.clone()
        else:
            new_data = input_view[:, :, 1:, :, :, :].clone()

        for x in reversed(list(range(1, self.new_length + 1))):
            if keep_rgb:
                new_data[:, :, x, :, :, :] = input_view[:, :, x, :, :, :] - input_view[:, :, x - 1, :, :, :]
            else:
                new_data[:, :, x - 1, :, :, :] = input_view[:, :, x, :, :, :] - input_view[:, :, x - 1, :, :, :]

        return new_data

    def _construct_flow_model(self, base_model):
        # modify the convolution layers
        # Torch models are usually defined in a hierarchical way.
        # nn.modules.children() return all sub modules in a DFS manner
        modules = list(self.base_model.modules())
        first_conv_idx = list(filter(lambda x: isinstance(modules[x], nn.Conv2d), list(range(len(modules)))))[0]
        conv_layer = modules[first_conv_idx]
        container = modules[first_conv_idx - 1]

        # modify parameters, assume the first blob contains the convolution kernels
        params = [x.clone() for x in conv_layer.parameters()]
        kernel_size = params[0].size()
        new_kernel_size = kernel_size[:1] + (2 * self.new_length,) + kernel_size[2:]
        new_kernels = params[0].data.mean(dim=1, keepdim=True).expand(new_kernel_size).contiguous()

        new_conv = nn.Conv2d(2 * self.new_length, conv_layer.out_channels,
                             conv_layer.kernel_size, conv_layer.stride, conv_layer.padding,
                             bias=True if len(params) == 2 else False)
        new_conv.weight.data = new_kernels
        if len(params) == 2:
            new_conv.bias.data = params[1].data  # add bias if neccessary
        layer_name = list(container.state_dict().keys())[0][:-7]  # remove .weight suffix to get the layer name

        # replace the first convlution layer
        setattr(container, layer_name, new_conv)

        if self.base_model_name == 'BNInception':
            import torch.utils.model_zoo as model_zoo
            sd = model_zoo.load_url('https://www.dropbox.com/s/35ftw2t4mxxgjae/BNInceptionFlow-ef652051.pth.tar?dl=1')
            base_model.load_state_dict(sd)
            print('=> Loading pretrained Flow weight done...')
        else:
            print('#' * 30, 'Warning! No Flow pretrained model is found')
        return base_model

    def _construct_diff_model(self, base_model, keep_rgb=False):
        # modify the convolution layers
        # Torch models are usually defined in a hierarchical way.
        # nn.modules.children() return all sub modules in a DFS manner
        modules = list(self.base_model.modules())
        first_conv_idx = filter(lambda x: isinstance(modules[x], nn.Conv2d), list(range(len(modules))))[0]
        conv_layer = modules[first_conv_idx]
        container = modules[first_conv_idx - 1]

        # modify parameters, assume the first blob contains the convolution kernels
        params = [x.clone() for x in conv_layer.parameters()]
        kernel_size = params[0].size()
        if not keep_rgb:
            new_kernel_size = kernel_size[:1] + (3 * self.new_length,) + kernel_size[2:]
            new_kernels = params[0].data.mean(dim=1, keepdim=True).expand(new_kernel_size).contiguous()
        else:
            new_kernel_size = kernel_size[:1] + (3 * self.new_length,) + kernel_size[2:]
            new_kernels = torch.cat(
                (params[0].data, params[0].data.mean(dim=1, keepdim=True).expand(new_kernel_size).contiguous()),
                1)
            new_kernel_size = kernel_size[:1] + (3 + 3 * self.new_length,) + kernel_size[2:]

        new_conv = nn.Conv2d(new_kernel_size[1], conv_layer.out_channels,
                             conv_layer.kernel_size, conv_layer.stride, conv_layer.padding,
                             bias=True if len(params) == 2 else False)
        new_conv.weight.data = new_kernels
        if len(params) == 2:
            new_conv.bias.data = params[1].data  # add bias if neccessary
        layer_name = list(container.state_dict().keys())[0][:-7]  # remove .weight suffix to get the layer name

        # replace the first convolution layer
        setattr(container, layer_name, new_conv)
        return base_model

    @property
    def crop_size(self):
        return self.input_size

    @property
    def scale_size(self):
        return self.input_size * 256 // 224


if __name__ == '__main__':
    x = torch.rand(2 * 8, 3, 224, 224)
    model = TSN(751, 8, "RGB",
                base_model="resnet50",
                consensus_type="avg",
                dropout=0,
                partial_bn=False,
                pretrain="imagenet",
                is_shift=False, shift_div=8, shift_place="blocker",
                fc_lr5=not (None and "kinetics" in None),
                temporal_pool=False,
                non_local=False)
    print(model.base_model)
    out = model(x)
    print(out['feature'].shape, out['predict'].shape)
