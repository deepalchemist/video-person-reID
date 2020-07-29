from __future__ import absolute_import
import os

import torch

from .resnet import *
from lib.model.tsm import TSN
from lib.model import resnet3d

__factory = {
    'tws': TempoWeightedSum,
    'rnn': TempoRNN,
    'tap': TempoAvgPooling,
    'tsn': TSN,
}


def get_names():
    return __factory.keys()


def init_model(name, *args, **kwargs):
    if name not in __factory.keys():
        raise KeyError("Unknown model: {}".format(name))
    return __factory[name](*args, **kwargs)


def get_model(cfg):
    if cfg.arch == '3d':
        model = resnet3d.resnet50(
            num_classes=cfg.num_train_pids, sample_width=cfg.width,
            sample_height=cfg.height, sample_duration=cfg.seq_len
        )
        if not cfg.pretrained_model or not os.path.exists(cfg.pretrained_model):
            raise IOError("Can't find pretrained model: {}".format(cfg.pretrained_model))
        print("Loading checkpoint from '{}'".format(cfg.pretrained_model))
        checkpoint = torch.load(cfg.pretrained_model)
        state_dict = {}
        for key in checkpoint['state_dict']:
            if 'fc' in key: continue
            state_dict[key.partition("module.")[2]] = checkpoint['state_dict'][key]
        model.load_state_dict(state_dict, strict=False)
    elif cfg.arch == 'tsn':
        model = init_model(
            name=cfg.arch,
            num_classes=cfg.num_train_pids, num_segments=cfg.seq_len, modality="RGB",
            base_model=cfg.base_model, conv5_stride=cfg.conv5_stride, bn=not cfg.no_batch_norm,
            pool_type=cfg.pool_type,
            consensus_type="avg", loss={'xent', 'htri'},
            non_local=cfg.non_local, stm=cfg.stm,
            is_shift=cfg.is_shift, shift_div=8, shift_place="blockres",
            fc_lr5=True,
        )
    else:
        model = init_model(
            name=cfg.arch,
            num_classes=cfg.num_train_pids
        )
    return model
