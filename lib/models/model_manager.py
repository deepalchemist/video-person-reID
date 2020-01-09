from __future__ import absolute_import

from .resnet import *
from lib.models.tsm import TSN

__factory = {
    'resnet50tp': ResNet50TP,
    'resnet50ta': ResNet50TP,
    'resnet50rnn': ResNet50RNN,
    'resnet50ts': TSN,
}


def get_names():
    return __factory.keys()


def init_model(name, *args, **kwargs):
    if name not in __factory.keys():
        raise KeyError("Unknown model: {}".format(name))
    return __factory[name](*args, **kwargs)
