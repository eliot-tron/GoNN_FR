from copy import copy
from copy import deepcopy

import torch
import torch.nn as nn

""" utils """
def oneD_probability_to_twoD_class(input_proba):
    return torch.stack((1 - input_proba, input_proba), dim=-1)


""" XOR network """
def xor_net(hid_size = 8, non_linearity: nn.Module=nn.ReLU()) -> nn.Module:
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = nn.Sequential(
        nn.Linear(2, hid_size),
        copy(non_linearity),
        # nn.Linear(hid_size, hid_size),
        # copy(non_linearity),
        # nn.Linear(hid_size, hid_size),
        # copy(non_linearity),
        # nn.Linear(hid_size, hid_size),
        # copy(non_linearity),
        nn.Linear(hid_size, 2),
    )
    return net

def xor_net_old(hid_size = 8, non_linearity=nn.Sigmoid()) -> nn.Module:
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = nn.Sequential(
        nn.Linear(2, hid_size),
        non_linearity,
        nn.Linear(hid_size, 1),
        # nn.Sigmoid() if not score else nn.Sequential(),
    )
    return net


""" XOR 3D network """
def xor3d_net(hid_size = 2, non_linearity: nn.Module=nn.ReLU()) -> nn.Module:
    net = nn.Sequential(
        nn.Linear(3, hid_size),
        non_linearity,
        nn.Linear(hid_size, 2),
    )
    return net

""" Circle network """
def circle_net(hid_size = [2, 8, 8, 2], non_linearity: nn.Module=nn.ReLU(), nclasses=2) -> nn.Module:
    if not isinstance(hid_size, list):
        hid_size = [hid_size]
    net = nn.Sequential(
        nn.Linear(2, hid_size[0]),
        copy(non_linearity),
        *[nn.Sequential(nn.Linear(hid_size[i], hid_size[i + 1]), copy(non_linearity)) for i in range(len(hid_size[1:]))],
        nn.Linear(hid_size[-1], nclasses),
    )
    return net


def shallow_circle_net(hid_size = 16, non_linearity: nn.Module=nn.ReLU(), nclasses=2) -> nn.Module:
    net = nn.Sequential(
        nn.Linear(2, hid_size),
        non_linearity,
        nn.Linear(hid_size, nclasses),
    )
    return net


""" MNIST network """
def mnist_medium_cnn(num_classes: int=10, non_linearity:nn.Module=nn.ReLU(), maxpool=False) -> nn.Module:
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")    
    net = nn.Sequential(
        nn.Conv2d(1, 32, 3, 1),
        deepcopy(non_linearity),
        nn.Conv2d(32, 64, 3, 1),
        deepcopy(non_linearity),
        nn.AvgPool2d(2) if not maxpool else nn.MaxPool2d(2),
        nn.Flatten(),
        nn.Linear(64 * (28 - 2 * 2) * (28 - 2 * 2) // (2**2), 128),
        deepcopy(non_linearity),
        nn.Linear(128, num_classes),
    )
    return net


""" CIFAR10 network """
cfg = {
    'VGG11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'VGG19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}


class VGG(nn.Module):
    def __init__(self, vgg_name, non_linearity=nn.ReLU(), num_classes:int=10):
        super(VGG, self).__init__()
        self.non_linearity = non_linearity
        self.features = self._make_layers(cfg[vgg_name])
        self.classifier = nn.Linear(512, num_classes)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, x):
        out = self.features(x)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return out

    def _make_layers(self, cfg):
        layers = []
        in_channels = 3
        for x in cfg:
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1),
                           nn.BatchNorm2d(x),
                           deepcopy(self.non_linearity)]
                in_channels = x
        layers += [nn.AvgPool2d(kernel_size=1, stride=1)]
        return nn.Sequential(*layers)

def cifar_medium_cnn(num_classes: int=10, non_linearity=nn.ReLU(), maxpool=False) -> nn.Module:
    net = VGG(vgg_name='VGG11', non_linearity=non_linearity, num_classes=num_classes)
    return net

def cifar_medium_cnn_inter(num_classes: int=10, non_linearity=nn.ReLU(), maxpool=False) -> nn.Module:
    net = nn.Sequential(
        nn.Conv2d(3, 300, 3, 1),
        deepcopy(non_linearity),
        nn.AvgPool2d(2) if not maxpool else nn.MaxPool2d(2),
        nn.Conv2d(300, 300, 3, 1),
        deepcopy(non_linearity),
        nn.AvgPool2d(2) if not maxpool else nn.MaxPool2d(2),
        nn.Conv2d(300, 300, 3, 1),
        deepcopy(non_linearity),
        nn.AvgPool2d(2) if not maxpool else nn.MaxPool2d(2),
        nn.Flatten(),
        nn.Linear(1200, 300),
        deepcopy(non_linearity),
        nn.Linear(300, 100),
        deepcopy(non_linearity),
        nn.Linear(100, num_classes),
    )
    return net
