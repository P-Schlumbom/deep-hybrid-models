from typing import Any, Callable, List, Optional, Type, Union, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch import Tensor
from torch.nn.utils.parametrizations import spectral_norm
from torchvision.models.resnet import ResNet, BasicBlock, Bottleneck, conv1x1, conv3x3

from helpers.utils import conv_output_size
from architectures.resnets.blocks import ModernBasicBlock
#from architectures.spectral_normalisation.spectral_norm_conv_inplace import spectral_norm_conv
#from architectures.spectral_normalisation.spectral_norm_fc import spectral_norm_fc
#from architectures.spectral_normalisation.spectral_batchnorm import _SpectralBatchNorm, SpectralBatchNorm2d

from architectures.spectral_normalisation.due_sn.spectral_norm_fc import spectral_norm_fc
from architectures.spectral_normalisation.due_sn.spectral_norm_conv import spectral_norm_conv
from architectures.spectral_normalisation.due_sn.spectral_batchnorm import SpectralBatchNorm2d

O = 0


class WideResNet(nn.Module):
    def __init__(
        self,
        block: Type[Union[BasicBlock, Bottleneck, ModernBasicBlock]],
        layers: List[int],
        input_size: Tuple[int, int, int] = (32, 32, 3),
        num_classes: int = 10,
        zero_init_residual: bool = False,
        groups: int = 1,
        width_per_group: int = 64,
        k=1,
        replace_stride_with_dilation: Optional[List[bool]] = None,
        #norm_layer: Optional[Callable[..., nn.Module]] = nn.BatchNorm2d,
        spectral_normalization: bool = False,
        n_power_iter: int = 1,
        coeff: float = 1.0,
    ) -> None:
        super().__init__()

        def wrapped_bn(num_features):
            if spectral_normalization:
                bn = SpectralBatchNorm2d(num_features, coeff)
            else:
                bn = nn.BatchNorm2d(num_features, eps=1e-5, momentum=0.01, affine=True)
            return bn

        self.wrapped_bn = wrapped_bn

        self.spectral_normalization = spectral_normalization
        self.n_power_iter = n_power_iter
        self.coeff = coeff

        self.inplanes = 16
        self.dilation = 1
        self.k = k
        self.in_width, self.in_height, self.in_depth = input_size
        self.out_size = np.asarray([self.in_width, self.in_height])
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError(
                "replace_stride_with_dilation should be None "
                f"or a 3-element tuple, got {replace_stride_with_dilation}"
            )
        self.groups = groups
        self.base_width = width_per_group
        if spectral_normalization:
            self.conv1 = self._wrapper_spectral_norm(nn.Conv2d(self.in_depth, self.inplanes, kernel_size=3, stride=1, padding=1,
                                                               bias=False), (self.in_depth, self.in_width-O, self.in_height-O), 3)
            self.out_size = conv_output_size(self.out_size, kernel=3, padding=1, stride=1)
        else:
            self.conv1 = nn.Conv2d(self.in_depth, self.inplanes, kernel_size=3, stride=1, padding=1, bias=False)
            self.out_size = conv_output_size(self.out_size, kernel=3, padding=1, stride=1)

        self.relu = nn.ReLU(inplace=True)
        self.layer1, self.out_size = self._make_layer(block, 16*self.k, layers[0], in_size=self.out_size)
        self.layer2, self.out_size = self._make_layer(block, 32*self.k, layers[1], stride=2, dilate=replace_stride_with_dilation[0], in_size=self.out_size)
        self.layer3, self.out_size = self._make_layer(block, 64*self.k, layers[2], stride=2, dilate=replace_stride_with_dilation[1], in_size=self.out_size)

        self.bnf = self.wrapped_bn(64*self.k)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(64 * k * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):  # and m != self.bnf:
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck) and m.bn3.weight is not None:
                    nn.init.constant_(m.bn3.weight, 0)  # type: ignore[arg-type]
                elif isinstance(m, BasicBlock) and m.bn2.weight is not None:
                    nn.init.constant_(m.bn2.weight, 0)  # type: ignore[arg-type]
                elif isinstance(m, ModernBasicBlock) and m.bn2.weight is not None:
                    nn.init.constant_(m.bn2.weight, 0)  # type: ignore[arg-type]

    def _make_layer(
        self,
        block: Type[Union[BasicBlock, Bottleneck, ModernBasicBlock]],
        planes: int,
        blocks: int,
        in_size,
        stride: int = 1,
        dilate: bool = False,
    ) -> nn.Sequential:
        #norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        output_size = in_size
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            if self.spectral_normalization:
                downsample = nn.Sequential(
                    self._wrapper_spectral_norm(nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1,
                                                          stride=stride, bias=False), (self.inplanes, in_size[0]-O, in_size[1]-O), 1),
                    #norm_layer(planes * block.expansion),
                )
            else:
                downsample = nn.Sequential(
                    conv1x1(self.inplanes, planes * block.expansion, stride),
                    #norm_layer(planes * block.expansion),
                )

        layers = []
        layers.append(
            block(
                self.inplanes, planes, stride, downsample, self.groups, self.base_width, previous_dilation, #norm_layer,
                self.spectral_normalization, in_size, self.n_power_iter, self.coeff
            )
        )
        output_size = conv_output_size(conv_output_size(output_size, kernel=3, padding=1, stride=stride), kernel=3,
                                       padding=1, stride=1)
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(
                block(
                    self.inplanes,
                    planes,
                    groups=self.groups,
                    base_width=self.base_width,
                    dilation=self.dilation,
                    #norm_layer=norm_layer,
                    spectral_normalization=self.spectral_normalization,
                    input_size=output_size,
                    n_power_iter=self.n_power_iter,
                    coeff=self.coeff
                )
            )
            output_size = conv_output_size(conv_output_size(output_size, kernel=3, padding=1, stride=1), kernel=3,
                                           padding=1, stride=1)

        return nn.Sequential(*layers), output_size

    def _forward_impl(self, x: Tensor, return_features=False) -> Tensor:
        x = self.conv1(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        x = self.relu(self.bnf(x))

        features = self.avgpool(x)
        features = features.view(features.size(0), -1)
        x = self.fc(features)

        if return_features:
            return x, features
        return x

    def _wrapper_spectral_norm(self, layer, shapes, kernel_size):
        if kernel_size == 1:
            # use spectral norm fc, because bound are tight for 1x1 convolutions
            #return spectral_norm_fc(layer, self.coeff,
            #                        n_power_iterations=self.n_power_iter)
            return spectral_norm_fc(layer, self.coeff, self.n_power_iter)
        else:
            # use spectral norm based on conv, because bound not tight
            #return spectral_norm_conv(layer, self.coeff, shapes,
            #                          n_power_iterations=self.n_power_iter)
            return spectral_norm_conv(layer, self.coeff, shapes, self.n_power_iter)

    def _wrapper_bn_spectral_norm(self, layer, num_features, eps=1e-5, momentum=0.01, affine=True):
        #if self.spectral_normalization:
        #    return layer(num_features, self.coeff, eps=eps, momentum=momentum, affine=affine)
        return layer(num_features)

    def forward(self, x: Tensor, return_features=False) -> Tensor:
        return self._forward_impl(x, return_features)


class SensitiveWideResNet(nn.Module):
    def __init__(
        self,
        block: Type[Union[BasicBlock, Bottleneck, ModernBasicBlock]],
        layers: List[int],
        input_size: Tuple[int, int, int] = (32, 32, 3),
        num_classes: int = 10,
        zero_init_residual: bool = False,
        groups: int = 1,
        width_per_group: int = 64,
        k=1,
        replace_stride_with_dilation: Optional[List[bool]] = None,
        #norm_layer: Optional[Callable[..., nn.Module]] = nn.BatchNorm2d,
        spectral_normalization: bool = False,
        n_power_iter: int = 1,
        coeff: int = 1.0,
    ) -> None:
        super().__init__()
        #_log_api_usage_once(self)
        """if norm_layer is None:
            if spectral_normalization:
                norm_layer = SpectralBatchNorm2d  # _SpectralBatchNorm
            else:
                norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer"""
        def wrapped_bn(num_features):
            if spectral_normalization:
                bn = SpectralBatchNorm2d(num_features, coeff)
            else:
                bn = nn.BatchNorm2d(num_features, eps=1e-5, momentum=0.01, affine=True)

            return bn

        self.wrapped_bn = wrapped_bn

        self.spectral_normalization = spectral_normalization
        self.n_power_iter = n_power_iter
        self.coeff = coeff

        self.inplanes = 16
        self.dilation = 1
        self.k = k
        self.in_width, self.in_height, self.in_depth = input_size
        self.out_size = np.asarray([self.in_width, self.in_height])
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError(
                "replace_stride_with_dilation should be None "
                f"or a 3-element tuple, got {replace_stride_with_dilation}"
            )
        self.groups = groups
        self.base_width = width_per_group
        if spectral_normalization:
            self.conv1 = self._wrapper_spectral_norm(nn.Conv2d(self.in_depth, self.inplanes, kernel_size=3, stride=1, padding=1,
                                                               bias=False), (self.in_depth, self.in_width-O, self.in_height-O), 3)
            self.out_size = conv_output_size(self.out_size, kernel=3, padding=1, stride=1)
        else:
            self.conv1 = nn.Conv2d(self.in_depth, self.inplanes, kernel_size=3, stride=1, padding=1, bias=False)
            self.out_size = conv_output_size(self.out_size, kernel=3, padding=1, stride=1)

        self.relu = nn.LeakyReLU(inplace=True)
        self.layer1, self.out_size = self._make_layer(block, 16*self.k, layers[0], in_size=self.out_size)
        self.layer2, self.out_size = self._make_layer(block, 32*self.k, layers[1], stride=2, dilate=replace_stride_with_dilation[0], in_size=self.out_size)
        self.layer3, self.out_size = self._make_layer(block, 64*self.k, layers[2], stride=2, dilate=replace_stride_with_dilation[1], in_size=self.out_size)

        self.bnf = self.wrapped_bn(64 * self.k)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(64 * k * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):  # and m != self.bnf:
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck) and m.bn3.weight is not None:
                    nn.init.constant_(m.bn3.weight, 0)  # type: ignore[arg-type]
                elif isinstance(m, BasicBlock) and m.bn2.weight is not None:
                    nn.init.constant_(m.bn2.weight, 0)  # type: ignore[arg-type]
                elif isinstance(m, ModernBasicBlock) and m.bn2.weight is not None:
                    nn.init.constant_(m.bn2.weight, 0)  # type: ignore[arg-type]

    def _make_layer(
        self,
        block: Type[Union[BasicBlock, Bottleneck, ModernBasicBlock]],
        planes: int,
        blocks: int,
        in_size,
        stride: int = 1,
        dilate: bool = False,
    ) -> nn.Sequential:
        downsample = None
        previous_dilation = self.dilation
        output_size = in_size
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            pooling_kernel_size = (np.asarray(output_size +2 - ((output_size + 1) / stride)).astype(int))[0]
            #pooling_kernel_size = np.asarray(output_size)
            expected_output_size = conv_output_size(output_size, pooling_kernel_size, stride=1)
            #print(f"output size: {output_size}, original stride: {stride}, new pooling kernel size: {pooling_kernel_size}, expected output size: {expected_output_size}")
            if self.spectral_normalization:
                downsample = nn.Sequential(
                    nn.AvgPool2d(pooling_kernel_size, stride=1),
                    self._wrapper_spectral_norm(nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1,
                                                          stride=1, bias=False), (self.inplanes, in_size[0]-O, in_size[1]-O), 1),

                )
            else:
                downsample = nn.Sequential(
                    nn.AvgPool2d(pooling_kernel_size, stride=1),
                    conv1x1(self.inplanes, planes * block.expansion, stride=1),
                )

        layers = []
        layers.append(
            block(
                self.inplanes, planes, stride, downsample, self.groups, self.base_width, previous_dilation, #norm_layer,
                self.spectral_normalization, in_size, self.n_power_iter, self.coeff, activation=nn.LeakyReLU(inplace=False)
            )
        )
        output_size = conv_output_size(conv_output_size(output_size, kernel=3, padding=1, stride=stride), kernel=3,
                                       padding=1, stride=1)
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(
                block(
                    self.inplanes,
                    planes,
                    groups=self.groups,
                    base_width=self.base_width,
                    dilation=self.dilation,
                    spectral_normalization=self.spectral_normalization,
                    input_size=output_size,
                    n_power_iter=self.n_power_iter,
                    coeff=self.coeff,
                    activation=nn.LeakyReLU(inplace=False)
                )
            )
            output_size = conv_output_size(conv_output_size(output_size, kernel=3, padding=1, stride=1), kernel=3,
                                           padding=1, stride=1)

        return nn.Sequential(*layers), output_size

    def _forward_impl(self, x: Tensor, return_features=False) -> Tensor:
        x = self.conv1(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        x = self.relu(self.bnf(x))
        #x = self.relu(x)

        features = self.avgpool(x)
        features = features.view(features.size(0), -1)
        x = self.fc(features)

        if return_features:
            return x, features
        return x

    def _wrapper_spectral_norm(self, layer, shapes, kernel_size):
        if kernel_size == 1:
            # use spectral norm fc, because bound are tight for 1x1 convolutions
            return spectral_norm_fc(layer, self.coeff, self.n_power_iter)
        else:
            # use spectral norm based on conv, because bound not tight
            return spectral_norm_conv(layer, self.coeff, shapes, self.n_power_iter)

    def _wrapper_bn_spectral_norm(self, layer, num_features, eps=1e-5, momentum=0.01, affine=True):
        return layer(num_features)

    def forward(self, x: Tensor, return_features=False) -> Tensor:
        return self._forward_impl(x, return_features)
