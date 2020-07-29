from __future__ import absolute_import
import math
from collections import OrderedDict

import torch
import torchvision
from torch import nn
from torch.nn import functional as F
import torch.utils.model_zoo as model_zoo

from .res2net import res2net50_26w_4s

__all__ = ['ResNet', 'resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152',
           'TempoAvgPooling', 'TempoWeightedSum', 'TempoRNN']

model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}


def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class TempoAvgPooling(nn.Module):
    """ Temporal Average Pooling """

    def __init__(self, num_classes):
        super(TempoAvgPooling, self).__init__()
        # resnet50 = torchvision.models.resnet50(pretrained=True)
        resnet50 = res2net50_26w_4s(pretrained=True)
        self.backbone = nn.Sequential(*list(resnet50.children())[:-2])
        self.last_layer_ch = 2048
        self.classifier = nn.Linear(self.last_layer_ch, num_classes, bias=False)
        nn.init.normal_(self.classifier.weight, std=0.01)

    def forward(self, x):
        """
        Args:
            x: (b t 3 H W)
        """
        b, t = x.size(0), x.size(1)
        x = x.view(b * t, x.size(2), x.size(3), x.size(4))
        x = self.backbone(x)  # (b*t c h w)
        x = F.avg_pool2d(x, x.size()[2:])
        x = x.view(b, t, -1).permute(0, 2, 1)  # (b t c) to (b c t)
        feature = F.avg_pool1d(x, t)  # (b c 1)
        feature = feature.view(b, self.last_layer_ch)

        if not self.training:
            return feature

        logits = self.classifier(feature)

        return logits, feature


class TempoWeightedSum(nn.Module):
    def __init__(self, num_classes):
        super(TempoWeightedSum, self).__init__()
        resnet50 = torchvision.models.resnet50(pretrained=True)
        self.backbone = nn.Sequential(*list(resnet50.children())[:-2])
        self.att_gen = 'softmax'  # method for attention generation: softMax or sigmoid
        self.last_layer_ch = 2048  # feature dimension
        self.middle_dim = 256  # middle layer dimension
        self.classifier = nn.Linear(self.last_layer_ch, num_classes, bias=False)
        nn.init.normal_(self.classifier.weight, std=0.01)

        # (7,4) corresponds to (224, 112) input image size
        self.spatial_attn = nn.Conv2d(self.last_layer_ch, self.middle_dim, kernel_size=[7, 4])
        self.temporal_attn = nn.Conv1d(self.middle_dim, 1, kernel_size=3, padding=1)

    def forward(self, x):
        b, t = x.size(0), x.size(1)
        x = x.view(b * t, x.size(2), x.size(3), x.size(4))
        featmaps = self.backbone(x)  # (b*t c h w)
        attn = F.relu(self.spatial_attn(featmaps)).view(b, t, -1).permute(0, 2, 1)  # (b*t c 1 1) to (b t c) to (b c t)
        attn = F.relu(self.temporal_attn(attn)).view(b, t)  # (b 1 t) to (b t)

        if self.att_gen == 'softmax':
            attn = F.softmax(attn, dim=1)
        elif self.att_gen == 'sigmoid':
            attn = F.sigmoid(attn)
            attn = F.normalize(attn, p=1, dim=1)
        else:
            raise KeyError("Unsupported attention generation function: {}".format(self.att_gen))

        feature = F.avg_pool2d(featmaps, featmaps.size()[2:]).view(b, t, -1)  # (b*t c 1 1) to (b t c)
        att_x = feature * attn.unsqueeze(attn, dim=-1)  # (b t c)
        att_x = torch.sum(att_x, dim=1)

        feature = att_x.view(b, -1)  # (b c)

        if not self.training:
            return feature

        logits = self.classifier(feature)

        return logits, feature


class TempoRNN(nn.Module):
    def __init__(self, num_classes):
        super(TempoRNN, self).__init__()
        resnet50 = torchvision.models.resnet50(pretrained=True)
        self.base = nn.Sequential(*list(resnet50.children())[:-2])
        self.hidden_dim = 512
        self.feat_dim = 2048
        self.classifier = nn.Linear(self.hidden_dim, num_classes, bias=False)
        nn.init.normal_(self.classifier.weight, std=0.01)

        self.lstm = nn.LSTM(input_size=self.feat_dim, hidden_size=self.hidden_dim, num_layers=1, batch_first=True)

    def forward(self, x):
        b = x.size(0)
        t = x.size(1)
        x = x.view(b * t, x.size(2), x.size(3), x.size(4))
        x = self.base(x)
        x = F.avg_pool2d(x, x.size()[2:])
        x = x.view(b, t, -1)
        output, (h_n, c_n) = self.lstm(x)
        output = output.permute(0, 2, 1)
        f = F.avg_pool1d(output, t)
        f = f.view(b, self.hidden_dim)
        if not self.training:
            return f
        y = self.classifier(f)

        return y, f


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, layers, conv1_ch=3, conv5_stride=1, num_classes=1000):
        self.inplanes = 64
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(conv1_ch, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=conv5_stride)
        self.avgpool = nn.AvgPool2d(7, stride=1)
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion))

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


def resnet18(pretrained=False, **kwargs):
    """Constructs a ResNet-18 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet18']))
    return model


def resnet34(pretrained=False, **kwargs):
    """Constructs a ResNet-34 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [3, 4, 6, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet34']))
    return model


def resnet50(pretrained=False, **kwargs):
    """Constructs a ResNet-50 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet50']))
    return model


def resnet101(pretrained=False, **kwargs):
    """Constructs a ResNet-101 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 23, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet101']))
    return model


def resnet152(pretrained=False, **kwargs):
    """Constructs a ResNet-152 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 8, 36, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet152']))
    return model


if __name__ == '__main__':
    model = resnet50()
    print(model)
    for block in model.layer2:
        print(block)
