from typing import Any, Callable, List, Optional, Type, Union

import numpy as np
import torch.nn as nn
from torch import Tensor
from torch.nn.utils.parametrizations import spectral_norm

from helpers.utils import conv_output_size
from architectures.spectral_normalisation.spectral_norm_conv_inplace import spectral_norm_conv
from architectures.spectral_normalisation.spectral_norm_fc import spectral_norm_fc
# from architectures.spectral_normalisation.spectral_batchnorm import _SpectralBatchNorm, SpectralBatchNorm2d
from architectures.spectral_normalisation.due_sn.spectral_norm_fc import spectral_norm_fc
from architectures.spectral_normalisation.due_sn.spectral_norm_conv import spectral_norm_conv
from architectures.spectral_normalisation.due_sn.spectral_batchnorm import SpectralBatchNorm2d

O = 0


def conv3x3(in_planes: int, out_planes: int, stride: int = 1, groups: int = 1, dilation: int = 1) -> nn.Conv2d:
    """3x3 convolution with padding"""
    return nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=dilation,
        groups=groups,
        bias=False,
        dilation=dilation,
    )


def conv1x1(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class ModernBasicBlock(nn.Module):
    expansion: int = 1

    def __init__(
            self,
            inplanes: int,
            planes: int,
            stride: int = 1,
            downsample: Optional[nn.Module] = None,
            groups: int = 1,
            base_width: int = 64,
            dilation: int = 1,
            #norm_layer: Optional[Callable[..., nn.Module]] = nn.BatchNorm2d,
            spectral_normalization: bool = False,
            input_size=None,
            n_power_iter: int = 1,
            coeff: float = 1.0,
            activation=nn.ReLU(inplace=False),
    ) -> None:
        super().__init__()
        """if norm_layer is None:
            if spectral_normalization:
                norm_layer = SpectralBatchNorm2d  # _SpectralBatchNorm
            else:
                norm_layer = nn.BatchNorm2d"""
        def wrapped_bn(num_features):
            if spectral_normalization:
                bn = SpectralBatchNorm2d(num_features, coeff)
            else:
                bn = nn.BatchNorm2d(num_features)

            #bn = nn.BatchNorm2d(num_features)

            return bn

        self.wrapped_bn = wrapped_bn

        if groups != 1 or base_width != 64:
            raise ValueError("BasicBlock only supports groups=1 and base_width=64")
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")

        self.spectral_normalization = spectral_normalization
        self.n_power_iter = n_power_iter
        self.coeff = coeff

        output_size = input_size
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        # self.bn1 = norm_layer(inplanes)
        self.bn1 = self.wrapped_bn(inplanes)  # self._wrapper_bn_spectral_norm(norm_layer, inplanes, coeff)
        if spectral_normalization:
            # print(f"{planes} block conv1 input: ", inplanes, output_size[0], output_size[1])
            """self.conv1 = self._wrapper_spectral_norm(nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride,
                                                               padding=1, groups=1, bias=False, dilation=1),
                                                     (inplanes, output_size[0], output_size[1]), 3)"""
            self.conv1 = self._wrapper_spectral_norm(conv3x3(inplanes, planes, stride),
                                                     (inplanes, output_size[0] - O, output_size[1] - O), 3)
            output_size = conv_output_size(output_size, kernel=3, padding=1, stride=stride)
            # print(f"{planes} block conv2 input: ", planes, output_size[0], output_size[1])
            """self.conv2 = self._wrapper_spectral_norm(nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                                                               padding=1, groups=1, bias=False, dilation=1),
                                                     (planes, output_size[0], output_size[1]), 3)"""
            self.conv2 = self._wrapper_spectral_norm(conv3x3(planes, planes),
                                                     (planes, output_size[0] - O, output_size[1] - O), 3)
        else:
            # print(f"{planes} block conv1 input: ", inplanes, output_size[0], output_size[1])
            self.conv1 = conv3x3(inplanes, planes, stride)
            output_size = conv_output_size(output_size, kernel=3, padding=1, stride=stride)
            # print(f"{planes} block conv2 input: ", planes, output_size[0], output_size[1])
            self.conv2 = conv3x3(planes, planes)
        #self.relu = nn.ReLU(inplace=False)
        self.relu = activation

        # self.bn2 = norm_layer(planes)
        self.bn2 = self.wrapped_bn(planes)  # self._wrapper_bn_spectral_norm(norm_layer, planes, coeff)
        self.downsample = downsample
        self.stride = stride
        self.dropout = nn.Dropout(0.3)

    def _forward_func(self, x: Tensor) -> Tensor:
        """
                As suggested by wide resnet paper, changed order of operations from
                conv -> bn -> relu
                to
                bn -> relu -> conv
                (even though it's weird and scary)
                :param x:
                :return:
                """
        identity = x

        out = self.bn1(x)
        out = self.relu(out)
        out = self.conv1(out)

        out = self.bn2(out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.conv2(out)

        if self.downsample is not None:
            identity = self.downsample(x)  # note: the ResNet class will automatically pass in a 1x1 downsampler if in
            # and out dims don't match
        #print(out.shape, identity.shape)
        out += identity
        # out = self.relu(out)   # is this supposed to be here?

        return out

    def _wrapper_spectral_norm(self, layer, shapes, kernel_size):
        if kernel_size == 1:
            # use spectral norm fc, because bound are tight for 1x1 convolutions
            # return spectral_norm_fc(layer, self.coeff,
            #                        n_power_iterations=self.n_power_iter)
            return spectral_norm_fc(layer, self.coeff, self.n_power_iter)
        else:
            # use spectral norm based on conv, because bound not tight
            # return spectral_norm_conv(layer, self.coeff, shapes,
            #                          n_power_iterations=self.n_power_iter)
            return spectral_norm_conv(layer, self.coeff, shapes, self.n_power_iter)

    def _wrapper_bn_spectral_norm(self, layer, num_features, eps=1e-5, momentum=0.01, affine=True):
        # if self.spectral_normalization:
        #    return layer(num_features, coeff=self.coeff, eps=eps, momentum=momentum, affine=affine)
        return layer(num_features)

    def forward(self, x: Tensor) -> Tensor:
        return self._forward_func(x)
