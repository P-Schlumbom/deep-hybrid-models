from typing import Any, Callable, List, Optional, Type, Union
from os import makedirs
from os.path import join, exists

import numpy as np
import torch
from torch import nn
# from torchvision.transforms import Normalize
# from torch.nn.utils.parametrizations import spectral_norm
from torch.nn import BatchNorm2d
from math import log, pi

import normflows as nfl

from architectures.resnets.blocks import ModernBasicBlock
from architectures.resnets.wide_resnet import WideResNet, spectral_norm_fc, SensitiveWideResNet
from architectures.resnets.due_wide_resnet import DUE_WideResNet, DUE_SensitiveWideResNet
from architectures.normalising_flows.glow import Glow
from architectures.normalising_flows.residual_flows.residual_flow import ResidualFlow, ACT_FNS, create_resflow
from architectures.normalising_flows.residual_flows.layers.elemwise import LogitTransform, Normalize, IdentityTransform, \
    ZeroMeanTransform
from architectures.normalising_flows.residual_flows.layers.squeeze import SqueezeLayer
from architectures.normalising_flows.residual_flows.residual_flow_models import create_flow_model
from architectures.distribution_models.multivariate_normal import NormalModel, GaussianMixtureModel
from helpers.utils import running_average, print_model_params, get_model_params, print_model_architecture, set_seed


def standard_normal_logprob(z):
    logZ = -0.5 * log(2 * pi)
    return logZ - z.pow(2) / 2


def define_flow_model(batch, in_channels, nf_type, n_blocks, n_flows, idim, factor_out, actnorm, act, fc_end,
                      n_exact_terms, affine, no_lu, squeeze_first=False, in_shape=None, fc=True, input_layer="ident"):
    if in_shape is None:
        in_shape = (batch, in_channels, 2, 2)
    else:
        in_shape = (batch, in_channels, in_shape[0], in_shape[1])
    if nf_type == "resflow":
        # init_layer = LogitTransform(0.05)  # TODO: need to check how this value is actually found...
        if input_layer == "logit":
            init_layer = LogitTransform(0.05)
        elif input_layer == "norm":
            init_layer = Normalize(0, 1)
        else:
            init_layer = IdentityTransform()
        # init_layer = IdentityTransform()
        # init_layer = Normalize([0 for i in range(64 * k)], [1 for i in range(64 * k)])
        # init_layer = ZeroMeanTransform()
        if squeeze_first:
            input_size = (in_channels, 2, 2)
            squeeze_layer = SqueezeLayer(2)
        return create_resflow(
            in_shape,
            n_blocks=[n_flows for i in range(n_blocks)],
            intermediate_dim=idim,
            factor_out=factor_out,
            init_layer=init_layer,
            actnorm=actnorm,
            activation_fn=act,
            fc_end=fc_end,
            n_exact_terms=n_exact_terms,
            fc=fc
        )
    elif nf_type == "glow":
        return Glow(
            in_channel=(in_channels),
            n_flow=n_flows,
            n_block=n_blocks,
            affine=affine,
            conv_lu=not no_lu,
            filter_size=idim,
        )
    else:
        print("Error! normalising flow type '{}' is not a valid option.".format(nf_type))
        raise Exception


# --------------------------IRESFLOW MODEL---------------------------------------------------------------------------- #

# TODO


# --------------------------NORMFLOWS MODEL--------------------------------------------------------------------------- #


def define_normflows_flow_model(n_flows=16, idim=128, hidden_layers=3, in_size=None, in_channels=64):
    K = n_flows
    latent_size = in_size if in_size else 2
    hidden_units = idim
    hidden_layers = hidden_layers

    flows = []
    for i in range(K):
        net = nfl.nets.LipschitzMLP([latent_size] + [hidden_units] * (hidden_layers - 1) + [latent_size],
                                    init_zeros=True, lipschitz_const=0.98)
        # net = nfl.nets.LipschitzCNN([in_channels] + [hidden_units]*(hidden_layers-1) + [in_channels], kernel_size=3,
        #                            init_zeros=True, lipschitz_const=0.9)
        flows += [nfl.flows.Residual(net, reduce_memory=True)]
        flows += [nfl.flows.ActNorm(latent_size)]

    # Set prior and q0
    # q0 = nfl.distributions.DiagGaussian(in_size, trainable=False)
    q0 = nfl.distributions.base.GaussianMixture(10, latent_size, trainable=True)

    # Construct flow model
    nfm = nfl.NormalizingFlow(q0=q0, flows=flows)

    return nfm


def logit_transform(x, alpha=0.05):
    s = alpha + (1 - 2 * alpha) * x
    y = torch.log(s) - torch.log(1 - s)
    return y


def normalise_features(features, is_training=False, ord=None, target_dims=(1, 2, 3)):
    max_vals = torch.amax(features, dim=target_dims, keepdim=True)  # per-sample max
    min_vals = torch.amin(features, dim=target_dims, keepdim=True)  # per-sample min
    # print(f"max_vals shape: {max_vals.shape}")

    if not ord:
        nf_features = (features - min_vals + 1e-5) / (max_vals - min_vals + 2e-5)
    else:
        nf_features = nn.functional.normalize(features, dim=target_dims, p=ord)
        max_vals = torch.norm(features, dim=target_dims, p=ord)
    # print(f"resulting maxvals: {torch.amax(nf_features, dim=(1, 2, 3))}")

    # max_vals = torch.max(features)  # cross-batch max (warning: is cheating!)
    # nf_features = (features + 1e-5) / (max_vals + 2e-5)

    """if is_training:
        max_val = torch.max(features)
        nf_features = features
    else:
        #max_val = 200
        max_val = torch.amax(features, dim=(1, 2, 3), keepdim=True)
        nf_features = torch.clamp(features, max=max_val - 1e-5)
    nf_features = (nf_features + 1e-5) / (max_val + 2e-5)"""

    """if is_training:
        max_vals = torch.max(features)
        nf_features = (features + 1e-5) / (max_vals + 2e-5)
    else:
        max_vals = 104.07  # empirical max over C10 dataset...
        nf_features = (features + 1e-5) / (max_vals + 2e-5)"""

    # print(max_vals.shape)
    #if target_dims == (1, 2, 3):
    #    max_vals = torch.squeeze(max_vals, 3)
    #    max_vals = torch.squeeze(max_vals, 2)
    # print(max_vals.shape)
    return nf_features, max_vals


class DHM(nn.Module):
    def __init__(
            self,
            dnn,
            flow,
            flow_in_shape=None,
            normalise_features=False,
            flatten_features=False,
            activate_features=True,
            **kwargs
    ) -> None:
        """
        Instantiate a Deep Hybrid Model (DHM) as described by Cao & Zhang (2022), which consists of a WideResNet (WRN)
        and a normalising flow, in this case optionally defined as either a glow or a residual flow model.

        :param resnet: the resnet classification model (preferably of type WideResNet)
        :param flow: the normalising flow model
        :param kwargs: other keyword arguments which will be passed on to the normalising flow (keywords differ by type)
        """
        super().__init__()
        self.normalise_features = normalise_features
        self.flatten_features = flatten_features
        self.activate_features = activate_features

        self.dnn = dnn
        self.dnn_out_size = [2, 2]
        if hasattr(self.dnn, 'out_size'):  # if the dnn stores its output tensor size, save it
            self.dnn_out_size = self.dnn.out_size
        if flow_in_shape is not None:
            self.flow_in_shape = flow_in_shape
        else:
            self.flow_in_shape = [8, 8, 640]
        # if flatten_features:
        #    self.dnn_out_size = [1, 1]
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU()
        # self.normalise = Normalize(0, 1)
        # self.normalize = BatchNorm2d(self.flow_in_shape[2])  # for batch normalisation, need the number of output channels from the DNN
        # self.dnn = nn.Sequential(*list(self.dnn.children())[:-1])  # remove the final layer from the input dnn model

        # self.bn = BatchNorm2d(64 * self.dnn.k)  # final batch normalisation layer

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        # self.avgpool = nn.AdaptiveMaxPool2d((1, 1))  # since feature maxvals seem to have strong influence on OOD
        # performance, might make sense to rely on maxpool instead of avgpool

        self.fc = self.dnn.fc
        self.dnn = nn.Sequential(*list(self.dnn.children())[:-2])  # remove the final fc layer and averagepool layer
        # self.dnn = nn.Sequential(*list(self.dnn.children())[:-3])  # remove the final fc layer and averagepool layer
        # from the input dnn model
        # delattr(self.dnn, 'fc')
        self.flow = flow

    def forward(self, x, logpz=0, inverse=False, save_feature_name=None, return_features=False):
        if inverse:
            return self.inverse(x, logpz)

        if self.activate_features:
            features = self.relu(self.dnn(x))
        else:
            features = self.dnn(x)

        # print(features.shape)
        if save_feature_name is not None:
            # torch.save(features[0], save_feature_name)
            # torch.save(torch.mean(features, dim=0), save_feature_name)
            print(torch.max(features))

        if self.flatten_features:
            # logpz, logdet, z = self.flow(nf_features)
            # features = self.bn(features)
            features = self.avgpool(features)
            if self.normalise_features:
                nf_features = normalise_features(features)
                # nf_features = self.normalize(features)
            else:
                nf_features = features
        else:
            if self.normalise_features:
                # print(f"before normalisation: {features.min()}, {features.mean()}, {features.max()}")
                nf_features = normalise_features(features)
                # print(f"after normalisation: {nf_features.min()}, {nf_features.mean()}, {nf_features.max()}")
                # nf_features = self.normalize(features)
            else:
                nf_features = features
            # features = self.bn(features)
            features = self.avgpool(features)

        if return_features:
            ret_features = nf_features
        # features = self.avgpool(features)
        # nf_features = normalise_features(features)
        # nf_features = features
        # features = torch.nn.functional.normalize(features)
        # print(nf_features.shape)
        # print(nf_features.min(), nf_features.mean(), nf_features.max())
        logpz, logdet, z = self.flow(nf_features)
        # print("feature and output shapes: ", nf_features.shape, logpz.shape, logdet.shape, z.shape)
        features = torch.flatten(features, 1)
        y = self.fc(features)
        if return_features:
            return y, logpz, logdet, z, ret_features
        return y, logpz, logdet, z, None

    def inverse(self, z, logpz=None):
        features_recon, logpx = self.flow(z, logpz, inverse=True)
        return features_recon, logpx

    def log_prob(self, x, return_features=False):
        _, logpz, logdet, _, features = self.forward(x, return_features=return_features)
        return logpz - logdet, features

    def feature_logprob(self, features):
        logpz, logdet, _ = self.flow(features)
        return logpz - logdet


class DHM_normflows(nn.Module):
    def __init__(
            self,
            dnn,
            flow: nfl.NormalizingFlow,
            flow_in_shape=None,
            normalise_features=False,
            flatten_features=False,
            activate_features=True,
            logit=False,
            common_features=True,
            **kwargs
    ) -> None:
        """
        Instantiate a Deep Hybrid Model (DHM) as described by Cao & Zhang (2022), which consists of a WideResNet (WRN)
        and a normalising flow, in this case optionally defined as either a glow or a residual flow model.

        :param resnet: the resnet classification model (preferably of type WideResNet)
        :param flow: the normalising flow model
        :param kwargs: other keyword arguments which will be passed on to the normalising flow (keywords differ by type)
        """
        super().__init__()
        self.normalise_features = normalise_features
        self.flatten_features = flatten_features
        self.activate_features = activate_features
        self.logit = logit
        self.common_features = common_features

        self.dnn = dnn
        self.dnn_out_size = [2, 2]
        if hasattr(self.dnn, 'out_size'):  # if the dnn stores its output tensor size, save it
            self.dnn_out_size = self.dnn.out_size
        if flow_in_shape is not None:
            self.flow_in_shape = flow_in_shape
        else:
            self.flow_in_shape = [8, 8, 640]
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU()
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        self.fc = self.dnn.fc
        self.dnn = nn.Sequential(*list(self.dnn.children())[:-2])
        self.flow = flow

    def get_features(self, features):
        if self.flatten_features:
            features = self.avgpool(features)
            if self.normalise_features:
                nf_features, max_vals = normalise_features(features)
            else:
                nf_features = features
        else:
            if self.normalise_features:
                nf_features, max_vals = normalise_features(features)
            else:
                nf_features = features
            features = self.avgpool(features)
        if self.logit:
            nf_features = logit_transform(nf_features)

        return features, nf_features

    def forward(self, x, logpz=0, inverse=False, return_features=False):
        if inverse:
            return self.inverse(x, logpz)

        if self.activate_features:
            features = self.relu(self.dnn(x))
        else:
            features = self.dnn(x)

        """if self.flatten_features:
            features = self.avgpool(features)
            if self.normalise_features:
                nf_features = normalise_features(features)
            else:
                nf_features = features
        else:
            if self.normalise_features:
                nf_features = normalise_features(features)
            else:
                nf_features = features
            features = self.avgpool(features)"""
        features, nf_features = self.get_features(features)

        if return_features:
            ret_features = nf_features

        if self.common_features:
            if self.flatten_features:
                features = nf_features
            else:
                features = self.avgpool(nf_features)
            # print("class function: ", ret_features.min(), ret_features.mean(), ret_features.max())
        # logpz, logdet, z = self.flow(nf_features)
        nf_features = torch.flatten(nf_features, 1)
        features = torch.flatten(features, 1)
        flow_loss = self.flow.forward_kld(nf_features)
        # y = self.fc(features)
        # y = self.fc(nf_features)
        y = self.fc(features)
        if return_features:
            return y, flow_loss, ret_features
        return y, flow_loss, None

    def inverse(self, z):
        # features_recon, logpx = self.flow(z, logpz, inverse=True)
        features_recon, logdet = self.flow.inverse_and_log_det(z)
        return features_recon, logdet

    def log_prob(self, x, return_features=False):
        if self.activate_features:
            features = self.relu(self.dnn(x))
        else:
            features = self.dnn(x)

        features, nf_features = self.get_features(features)

        nf_features = torch.flatten(nf_features, 1)
        if return_features:
            # print(nf_features.shape)
            return self.flow.log_prob(nf_features), nf_features
        return self.flow.log_prob(nf_features), None

    def feature_logprob(self, features):
        return self.flow.log_prob(features)


class DHM_iresflows(nn.Module):
    def __init__(
            self,
            dnn,
            flow,
            flow_in_shape=None,
            normalise_features=False,
            flatten_features=False,
            logit=False,
            common_features=True,
            norm_ord=torch.inf,
            **kwargs
    ) -> None:
        """
        Instantiate a Deep Hybrid Model (DHM) as described by Cao & Zhang (2022), which consists of a WideResNet (WRN)
        and a normalising flow, in this case optionally defined as either a glow or a residual flow model.

        :param resnet: the resnet classification model (preferably of type WideResNet)
        :param flow: the normalising flow model
        :param kwargs: other keyword arguments which will be passed on to the normalising flow (keywords differ by type)
        """
        super().__init__()
        self.normalise_features = normalise_features
        self.norm_ord = norm_ord
        self.flatten_features = flatten_features
        self.logit = logit
        self.common_features = common_features

        self.dnn = dnn
        self.dnn_out_size = [1, 1]
        if hasattr(self.dnn, 'out_size'):  # if the dnn stores its output tensor size, save it
            self.dnn_out_size = self.dnn.out_size
        if flow_in_shape is not None:
            self.flow_in_shape = flow_in_shape
        else:
            self.flow_in_shape = [8, 8, 640]
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU()
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        self.fc = self.dnn.fc
        self.dnn = nn.Sequential(*list(self.dnn.children())[:-2])  # original
        #self.dnn = nn.Sequential(*list(self.dnn.children())[:-1])  # DUE
        self.flow = flow

    def get_features(self, features):
        max_vals = None
        if self.flatten_features:
            features = self.avgpool(features)
            if self.normalise_features:
                #  print(features.shape)  # [batch, idim, 1, 1]
                nf_features, max_vals = normalise_features(features, ord=self.norm_ord)
            else:
                nf_features = features
        else:
            if self.normalise_features:
                nf_features, max_vals = normalise_features(features, ord=self.norm_ord)
            else:
                nf_features = features
            features = self.avgpool(features)
        if self.logit:
            nf_features = nf_features.squeeze()
            nf_features = (nf_features + 1.0) / 2.0  # rescale to 0-1. Also, max_vals has to be reshaped for broadcasting op
            max_vals = max_vals * 2
            nf_features = logit_transform(nf_features)  # note that this is probably a silly thing to add

        init_probs = torch.zeros(nf_features.shape[0], 1).to(nf_features)
        if self.normalise_features:
            init_probs = -torch.log(max_vals.unsqueeze(1))  # needs to be negative, since we calculate determinant in x -> z direction
        return features, nf_features, init_probs

    def forward(self, x, logpz=0, inverse=False, return_features=False):
        if inverse:
            return self.inverse(x, logpz)

        features = self.dnn(x)

        features, nf_features, init_probs = self.get_features(features)

        if return_features:
            ret_features = nf_features
            ret_features = ret_features.view(ret_features.size(0), -1)

        if self.common_features:
            if self.flatten_features:
                features = nf_features
            else:
                features = self.avgpool(nf_features)

        # x = x.view(x.size(0), -1)
        # nf_features = torch.flatten(nf_features, 1)
        nf_features = nf_features.view(nf_features.size(0), -1)
        # zero = torch.zeros(nf_features.shape[0], 1).to(nf_features)
        # print(zero.shape)
        # features = torch.flatten(features, 1)
        features = features.view(features.size(0), -1)
        #print(init_probs.shape, nf_features.shape)
        #print(min(init_probs), max(init_probs))
        z, logdet = self.flow(nf_features, init_probs)

        # compute logp(z)
        logpz = standard_normal_logprob(z).sum(1, keepdim=True)

        y = self.fc(features)  # this is what the server has been using...
        if return_features:
            return y, logpz, logdet, z, nf_features
        return y, logpz, logdet, z, None

    def inverse(self, z):
        return None

    def log_prob(self, x, return_features=False):
        _, logpz, logdet, _, features = self.forward(x, return_features=return_features)
        return logpz - logdet, features
        #return logpz + logdet, features

    def feature_logprob(self, features):
        zero = torch.zeros(features.shape[0], 1).to(features)
        features = features.view(features.size(0), -1)
        z, logdet = self.flow(features, zero)
        logpz = standard_normal_logprob(z).sum(1, keepdim=True)
        return logpz - logdet


class DHM_iresflows_bottleneck(DHM_iresflows):
    def __init__(
            self,
            dnn,
            flow,
            bottleneck=10,
            flow_in_shape=None,
            normalise_features=False,
            flatten_features=False,
            logit=False,
            common_features=True,
            **kwargs
    ) -> None:
        super().__init__(dnn, flow, flow_in_shape, normalise_features, flatten_features, logit, common_features,
                         **kwargs)
        in_shape = self.fc.in_features
        self.bottleneck = spectral_norm_fc(nn.Linear(in_shape, bottleneck), kwargs['dnn_coeff'],
                                           n_power_iterations=kwargs['n_power_iter'])
        self.flow_in_shape = bottleneck
        output_size = self.fc.out_features
        self.fc = nn.Linear(bottleneck, output_size)

    def get_features(self, features):
        if self.flatten_features:
            features = self.avgpool(features)
            nf_features = features
        else:
            nf_features = features
            features = self.avgpool(features)
        if self.logit:
            nf_features = logit_transform(nf_features)  # note that this is probably a silly thing to add

        init_probs = torch.zeros(nf_features.shape[0], 1).to(nf_features)

        # print(nf_features.shape)
        B = nf_features.size(0)
        nf_features = nf_features.view(B, -1)
        nf_features = self.bottleneck(nf_features)
        if self.normalise_features:
            nf_features, _ = normalise_features(nf_features, target_dims=1, ord=self.norm_ord)
        nf_features = nf_features.view(B, self.flow_in_shape, 1, 1)
        return features, nf_features, init_probs


class DHM_MVN(DHM_iresflows):
    def __init__(
            self,
            dnn,
            mvn,
            bottleneck=10,
            flow_in_shape=None,
            normalise_features=False,
            flatten_features=False,
            logit=False,
            common_features=True,
            norm_ord=torch.inf,
            **kwargs
    ) -> None:
        super().__init__(dnn, mvn, flow_in_shape, normalise_features, flatten_features, logit, common_features,
                         norm_ord=norm_ord, **kwargs)
        self.mvn = mvn

    def forward(self, x, logpz=0, inverse=False, return_features=False):
        if inverse:
            return self.inverse(x, logpz)

        features = self.dnn(x)

        features, nf_features, init_probs = self.get_features(features)

        if return_features:
            ret_features = nf_features
            ret_features = ret_features.view(ret_features.size(0), -1)

        if self.common_features:
            if self.flatten_features:
                features = nf_features
            else:
                features = self.avgpool(nf_features)

        nf_features = nf_features.view(nf_features.size(0), -1)
        features = features.view(features.size(0), -1)

        logpz = self.mvn(nf_features)

        y = self.fc(features)
        zero = torch.zeros(features.shape[0], 1).to(features)
        if return_features:
            return y, logpz, zero, None, ret_features
        return y, logpz, zero, None, None

    def log_prob(self, x, return_features=False):
        _, logpz, _, _, features = self.forward(x, return_features=return_features)
        return logpz, features

    def feature_logprob(self, features):
        features = features.view(features.size(0), -1)
        logpz = self.mvn(features)
        return logpz


class BottleneckDHM(DHM_normflows):
    def __init__(
            self,
            dnn,
            flow: nfl.NormalizingFlow,
            bottleneck_size=2,
            flow_in_shape=None,
            normalise_features=False,
            flatten_features=False,
            activate_features=True,
            logit=False,
            common_features=True,
            **kwargs

    ):
        super().__init__(dnn, flow, flow_in_shape, normalise_features, flatten_features, activate_features, logit,
                         common_features, **kwargs)
        in_shape = self.flow_in_shape
        if flatten_features:
            in_shape = in_shape[-1]
        else:
            in_shape = np.prod(in_shape)
        # self.bottleneck = nn.Linear(in_shape, bottleneck_size)
        self.bottleneck = spectral_norm_fc(nn.Linear(in_shape, bottleneck_size), kwargs['coeff'],
                                           n_power_iterations=kwargs['n_power_iter'])
        self.flow_in_shape = bottleneck_size
        output_size = self.fc.out_features
        self.fc = nn.Linear(bottleneck_size, output_size)

    def get_features(self, features):
        if self.flatten_features:
            features = self.avgpool(features)
            # if self.normalise_features:
            #    nf_features = normalise_features(features)
            # else:
            #    nf_features = features
            nf_features = features
        else:
            # if self.normalise_features:
            #    nf_features = normalise_features(features)
            # else:
            #    nf_features = features
            nf_features = features
            features = self.avgpool(features)
        if self.logit:
            nf_features = logit_transform(nf_features)

        # nf_features = nf_features.squeeze(dim=2).squeeze(dim=2)
        # nf_features = self.bottleneck(nf_features)
        # nf_features = nf_features.unsqueeze(dim=2).unsqueeze(dim=2)

        orig_shape = nf_features.shape
        B = orig_shape[0]
        nf_features = nf_features.view(B, -1)
        nf_features = self.bottleneck(nf_features)
        if self.normalise_features:
            nf_features = torch.nn.functional.normalize(nf_features)
        nf_features = nf_features.view(B, self.flow_in_shape, 1, 1)

        return features, nf_features


# ----------------------------------------------------------------------------------------------------------------------#
#
# helper functions
#
# ----------------------------------------------------------------------------------------------------------------------#

def normflows_logpx_loop(model: DHM_normflows, inputs):
    return model.log_prob(inputs)


# ----------------------------------------------------------------------------------------------------------------------#
#
# implementation functions
#
# ----------------------------------------------------------------------------------------------------------------------#


def create_dhm(
        batch,
        input_size=(32, 32, 3),
        n_classes=10,
        N=4,
        k=10,
        n_flows=6,
        idim=64,
        flatten=True,
        init_layer="logit",
        normalise_features=True,
        use_normflows=True,
        activate_features=False,
        common_features=True,
        **kwargs
):
    default_args = {
        "sn": True,
        "n_power_iter": 1,
        "coeff": 1.0,
        "nf_type": "resflow",
        "n_blocks": 1,
        "fc": True,
        "factor_out": False,
        "actnorm": True,
        "act": 'swish',
        "fc_end": True,
        "n_exact_terms": 2,
        "affine": False,
        "no_lu": False,
        "squeeze_first": False,
        "dirpath": "checkpoints/testing",
        "src_name": None,
    }
    default_args.update(kwargs)
    # --- DEFINE DNN --- #
    # forming Wide ResNet 28-10, WRN 28-10:
    n = N * 6 + 4
    print("Creating model WRN-{}-{} with N={}".format(n, k, N))
    # model = resnet.ResNet(ModernBasicBlock, [4, 4, 4], num_classes=10, width_per_group=64*k)
    print("sn: ", default_args['sn'])
    dnn = WideResNet(ModernBasicBlock, [N, N, N], input_size=input_size, num_classes=n_classes, k=k,
                     spectral_normalization=default_args['sn'], n_power_iter=default_args['n_power_iter'],
                     coeff=default_args['coeff'])
    dnn_outshape = dnn.out_size
    if flatten: dnn_outshape = [1, 1]
    print(f"outshape: {dnn_outshape}")
    # model.to(device)
    print("number of parameters: {} ({:,})".format(get_model_params(dnn), get_model_params(dnn)))

    # --- DEFINE NF --- #
    print("Creating {} type normalising flow with {} blocks and {} flows each, hidden dimension {}".format(
        default_args['nf_type'],
        default_args['n_blocks'],
        n_flows,
        idim))
    # nf = define_flow_model(dnn_outshape)
    M = np.prod(dnn_outshape) * k * 64
    # M = 2
    fc = True if flatten else default_args['fc']
    if use_normflows:
        nf = define_normflows_flow_model(n_flows=n_flows, idim=idim, in_size=int(M))
    else:
        nf = define_flow_model(batch, 64 * k, default_args['nf_type'], default_args['n_blocks'], n_flows, idim,
                               default_args['factor_out'],
                               default_args['actnorm'], default_args['act'], default_args['fc_end'],
                               default_args['n_exact_terms'], default_args['affine'], default_args['no_lu'],
                               default_args['squeeze_first'], dnn_outshape, fc=fc, input_layer=init_layer)

    print("number of parameters: {} ({:,})".format(get_model_params(nf), get_model_params(nf)))

    # --- DEFINE DHM --- #
    print("creating deep hybrid model with WRN-{}-{} dnn and {}-{}x{} nf".format(n, k, default_args['nf_type'],
                                                                                 default_args['n_blocks'],
                                                                                 n_flows))
    print(f"model code: DHM-{n}-{k}-{default_args['n_blocks']}-{n_flows}-{idim}")
    print(f"flatten: {flatten}, init layer: {init_layer}, normalise features: {normalise_features}, "
          f"activate features: {activate_features}, use normflows: {use_normflows}, "
          f"use common features: {common_features}")
    if use_normflows:
        use_logit = init_layer == "logit"
        # bottleneck_dim = 3
        dhm = DHM_normflows(dnn, nf, normalise_features=normalise_features, flatten_features=flatten,
                            flow_in_shape=[dnn_outshape[0], dnn_outshape[1], k * 64],
                            activate_features=activate_features, logit=use_logit, common_features=common_features)
        # dhm = BottleneckDHM(dnn, nf, bottleneck_size=M, normalise_features=normalise_features, flatten_features=flatten,
        #                    flow_in_shape=[dnn_outshape[0], dnn_outshape[1], k * 64],
        #                    activate_features=activate_features, logit=use_logit, common_features=common_features, n_power_iter=default_args['n_power_iter'], coeff=default_args['coeff'])
    else:
        dhm = DHM(dnn, nf, normalise_features=normalise_features, flatten_features=flatten,
                  flow_in_shape=[dnn_outshape[0], dnn_outshape[1], k * 64], activate_features=activate_features)

    if not exists(default_args['dirpath']):
        print(f"creating dir {default_args['dirpath']}...")
        makedirs(default_args['dirpath'])
    if default_args['src_name'] is not None:
        print(f"loading parameters from {join(default_args['dirpath'], default_args['src_name'])}...")
        model_dict = torch.load(join(default_args['dirpath'], default_args['src_name']))
        dhm.load_state_dict(model_dict['state_dict'])
    # dhm.load_state_dict(torch.load("checkpoints/testing/20230404_dry-silence-52.pth"))
    print("number of parameters: {} ({:,})".format(get_model_params(dhm), get_model_params(dhm)))

    return dhm


def create_dhm_with_custom_nf(
        nf_model,
        input_size=(32, 32, 3),
        n_classes=10,
        N=4,
        k=10,
        n_flows=6,
        idim=64,
        flatten=True,
        init_layer="logit",
        normalise_features=True,
        activate_features=False,
        common_features=True,
        **kwargs):
    default_args = {
        "sn": True,
        "n_power_iter": 1,
        "coeff": 1.0,
        "nf_type": "resflow",
        "n_blocks": 1,
        "fc": True,
        "factor_out": False,
        "actnorm": True,
        "act": 'swish',
        "fc_end": True,
        "n_exact_terms": 2,
        "affine": False,
        "no_lu": False,
        "squeeze_first": False,
        "dirpath": "checkpoints/testing",
        "src_name": None,
    }
    default_args.update(kwargs)

    # --- DEFINE DNN --- #
    # forming Wide ResNet 28-10, WRN 28-10:
    n = N * 6 + 4
    print("Creating model WRN-{}-{} with N={}".format(n, k, N))
    # model = resnet.ResNet(ModernBasicBlock, [4, 4, 4], num_classes=10, width_per_group=64*k)
    print("sn: ", default_args['sn'])
    dnn = WideResNet(ModernBasicBlock, [N, N, N], input_size=input_size, num_classes=n_classes, k=k,
                     spectral_normalization=default_args['sn'], n_power_iter=default_args['n_power_iter'],
                     coeff=default_args['coeff'])
    dnn_outshape = dnn.out_size
    if flatten: dnn_outshape = [1, 1]
    print(f"outshape: {dnn_outshape}")
    # model.to(device)
    print("number of parameters: {} ({:,})".format(get_model_params(dnn), get_model_params(dnn)))

    M = np.prod(dnn_outshape) * k * 64
    fc = True if flatten else default_args['fc']
    print("number of parameters: {} ({:,})".format(get_model_params(nf_model), get_model_params(nf_model)))

    # --- DEFINE DHM --- #
    print("creating deep hybrid model with WRN-{}-{} dnn and {}-{}x{} nf".format(n, k, default_args['nf_type'],
                                                                                 default_args['n_blocks'],
                                                                                 n_flows))
    print(f"model code: DHM-{n}-{k}-{default_args['n_blocks']}-{n_flows}-{idim}")
    print(f"flatten: {flatten}, init layer: {init_layer}, normalise features: {normalise_features}, "
          f"activate features: {activate_features}, "
          f"use common features: {common_features}")

    dhm = DHM(dnn, nf_model, normalise_features=normalise_features, flatten_features=flatten,
              flow_in_shape=[dnn_outshape[0], dnn_outshape[1], k * 64], activate_features=activate_features)

    # load parameters if requested...
    if not exists(default_args['dirpath']):
        print(f"creating dir {default_args['dirpath']}...")
        makedirs(default_args['dirpath'])
    if default_args['src_name'] is not None:
        print(f"loading parameters from {join(default_args['dirpath'], default_args['src_name'])}...")
        model_dict = torch.load(join(default_args['dirpath'], default_args['src_name']))
        dhm.load_state_dict(model_dict['state_dict'])
    # dhm.load_state_dict(torch.load("checkpoints/testing/20230404_dry-silence-52.pth"))
    print("number of parameters: {} ({:,})".format(get_model_params(dhm), get_model_params(dhm)))

    return dhm


def create_ires_dhm(
        input_size=(32, 32, 3),
        n_classes=10,
        bottleneck=None,
        N=4,
        k=10,
        common_features=True,
        normalise_features=False,
        flatten=True,
        sn=True,
        n_power_iter=1,
        dnn_coeff=0.98,
        n_blocks=10,
        dims='128-128-128-128',
        actnorm=False,
        act='swish',
        n_dist='geometric',
        n_power_series=None,
        exact_trace=False,
        brute_force=False,
        n_samples=1,
        batchnorm=False,
        vnorms='222222',
        learn_p=False,
        mixed=True,
        nf_coeff=0.9,
        n_lipschitz_iters=5,
        atol=None,
        rtol=None,
        dirpath="checkpoints/testing",
        src_name=None,
        init_layer=None,
        norm_ord=torch.inf,
):
    # load existing model if available
    if not exists(dirpath):
        print(f"creating dir {dirpath}...")
        makedirs(dirpath)
    if src_name is not None:
        print(f"loading parameters from {join(dirpath, src_name)}...")
        model_dict = torch.load(join(dirpath, src_name))
        args = model_dict["args"]
        N = args["N"]
        k = args["k"]
        common_features = args["common_features"]
        normalise_features = args["normalise_features"]
        flatten=args["flatten"]
        sn=args["sn"]
        n_power_iter=args["n_power_iter"]
        dnn_coeff=args["dnn_coeff"]
        n_blocks=args["n_blocks"]
        dims=args["dims"]
        actnorm=args["actnorm"]
        act=args["act"]
        n_dist=args["n_dist"]
        n_power_series=args["n_power_series"]
        exact_trace=args["exact_trace"]
        brute_force=args["brute_force"]
        n_samples=args["n_samples"]
        batchnorm=args["batchnorm"]
        vnorms=args["vnorms"]
        learn_p=args["learn_p"]
        mixed=args["mixed"]
        nf_coeff=args["nf_coeff"]
        n_lipschitz_iters=args["n_lipschitz_iters"]
        atol=args["atol"]
        rtol=args["rtol"]
        init_layer=args["init_layer"]

    # --- DEFINE DNN --- #
    # forming Wide ResNet 28-10, WRN 28-10:
    n = N * 6 + 4
    print("Creating model WRN-{}-{} with N={}".format(n, k, N))

    print("sn: ", sn)

    dnn = WideResNet(ModernBasicBlock, [N, N, N], input_size=input_size, num_classes=n_classes, k=k,
                     spectral_normalization=sn, n_power_iter=n_power_iter,
                     coeff=dnn_coeff)
    #dnn = SensitiveWideResNet(ModernBasicBlock, [N, N, N], input_size=input_size, num_classes=n_classes, k=k,
    #                          spectral_normalization=sn, n_power_iter=n_power_iter,
    #                          coeff=dnn_coeff)
    # dnn = DUE_WideResNet(input_size=input_size[0], spectral_conv=True, spectral_bn=True, num_classes=10)
    #dnn = DUE_SensitiveWideResNet(input_size=input_size[0], spectral_conv=True, spectral_bn=True, num_classes=10)

    dnn_outshape = [1, 1]  # dnn.out_size
    if flatten: dnn_outshape = [1, 1]
    print(f"outshape: {dnn_outshape}")
    # print_model_architecture(dnn)
    # model.to(device)
    print("number of parameters: {} ({:,})".format(get_model_params(dnn), get_model_params(dnn)))

    # --- DEFINE NF --- #
    print("Creating ires normalising flow with {} blocks and {} flows each, hidden dimension {}. Init layer: {}".format(
        n_blocks,
        len(dims.split('-')),
        dims.split('-')[0],
        init_layer)
    )
    """if init_layer:
        if init_layer == "logit":
            init_layer = LogitTransform(0.05)
            normalise_features = True  # has to be true for logittransform"""
    # nf = define_flow_model(dnn_outshape)
    M = np.prod(dnn_outshape) * k * 64
    if bottleneck:
        M = bottleneck

    nf = create_flow_model(
        input_size=M,
        dims=dims,
        actnorm=actnorm,
        n_blocks=n_blocks,
        act=act,
        n_dist=n_dist,
        n_power_series=n_power_series,
        exact_trace=exact_trace,
        brute_force=brute_force,
        n_samples=n_samples,
        batchnorm=batchnorm,
        vnorms=vnorms,
        learn_p=learn_p,
        mixed=mixed,
        coeff=nf_coeff,
        n_lipschitz_iters=n_lipschitz_iters,
        atol=atol,
        rtol=rtol,
        init_layer=None,
    )

    print("number of parameters: {} ({:,})".format(get_model_params(nf), get_model_params(nf)))

    # --- DEFINE DHM --- #
    print("creating deep hybrid model with WRN-{}-{} dnn and {}x{} nf".format(n, k,
                                                                              n_blocks,
                                                                              len(dims.split('-'))))
    print(f"model code: DHM-{n}-{k}-{n_blocks}-{len(dims.split('-'))}-{dims.split('-')[0]}")
    print(f"flatten: {flatten}, normalise features: {normalise_features}, "
          f"use common features: {common_features}")

    if not bottleneck:
        dhm = DHM_iresflows(dnn, nf, flow_in_shape=[dnn_outshape[0], dnn_outshape[1], k * 64],
                            normalise_features=normalise_features, flatten_features=flatten,
                            common_features=common_features, norm_ord=norm_ord, logit=(init_layer=="logit"))
    else:
        dhm = DHM_iresflows_bottleneck(dnn, nf, bottleneck=bottleneck, flow_in_shape=M,
                                       normalise_features=normalise_features, flatten_features=flatten,
                                       common_features=common_features, dnn_coeff=dnn_coeff, n_power_iter=n_power_iter)

    if not exists(dirpath):
        print(f"creating dir {dirpath}...")
        makedirs(dirpath)
    if src_name is not None:
        print(f"loading parameters from {join(dirpath, src_name)}...")
        model_dict = torch.load(join(dirpath, src_name))
        dhm.load_state_dict(model_dict['state_dict'])

    print("number of parameters: {} ({:,})".format(get_model_params(dhm), get_model_params(dhm)))
    print_model_architecture(dhm.dnn)

    return dhm


def create_mvn_dhm(
        input_size=(32, 32, 3),
        n_classes=10,
        dist_model="mvn",
        bottleneck=None,
        N=4,
        k=10,
        common_features=True,
        normalise_features=False,
        flatten=True,
        sn=True,
        n_power_iter=1,
        dnn_coeff=0.98,
        norm_ord=torch.inf,
        dirpath="checkpoints/testing",
        src_name=None,
):
    # --- DEFINE DNN --- #
    # forming Wide ResNet 28-10, WRN 28-10:
    n = N * 6 + 4
    print("Creating model WRN-{}-{} with N={}".format(n, k, N))

    print("sn: ", sn)
    dnn = WideResNet(ModernBasicBlock, [N, N, N], input_size=input_size, num_classes=n_classes, k=k,
                     spectral_normalization=sn, n_power_iter=n_power_iter,
                     coeff=dnn_coeff)
    dnn_outshape = dnn.out_size
    if flatten: dnn_outshape = [1, 1]
    print(f"outshape: {dnn_outshape}")
    # print_model_architecture(dnn)
    # model.to(device)
    print("number of parameters: {} ({:,})".format(get_model_params(dnn), get_model_params(dnn)))

    # --- DEFINE MVN --- #
    M = np.prod(dnn_outshape) * k * 64
    if bottleneck:
        M = bottleneck
    print("Creating MVN model with input size".format(M))

    if dist_model == "mvn":
        mvn = NormalModel(M)
    elif dist_model == "gmm":
        mvn = GaussianMixtureModel(n_classes, M)

    print("number of parameters: {} ({:,})".format(get_model_params(mvn), get_model_params(mvn)))

    # --- DEFINE DHM --- #
    print("creating deep hybrid model with MVN-{} dnn".format(M))
    print(f"model code: DHM-{n}-{k}-mvn-{M}")
    print(f"flatten: {flatten}, normalise features: {normalise_features}, "
          f"use common features: {common_features}")

    if not bottleneck:
        dhm = DHM_MVN(dnn, mvn, flow_in_shape=[dnn_outshape[0], dnn_outshape[1], k * 64],
                      normalise_features=normalise_features, flatten_features=flatten,
                      common_features=common_features, norm_ord=norm_ord)
    else:
        dhm = DHM_iresflows_bottleneck(dnn, mvn, bottleneck=bottleneck, flow_in_shape=M,
                                       normalise_features=normalise_features, flatten_features=flatten,
                                       common_features=common_features, dnn_coeff=dnn_coeff, n_power_iter=n_power_iter)

    if not exists(dirpath):
        print(f"creating dir {dirpath}...")
        makedirs(dirpath)
    if src_name is not None:
        print(f"loading parameters from {join(dirpath, src_name)}...")
        model_dict = torch.load(join(dirpath, src_name))
        dhm.load_state_dict(model_dict['state_dict'])

    print("number of parameters: {} ({:,})".format(get_model_params(dhm), get_model_params(dhm)))
    print_model_architecture(dhm.dnn)

    return dhm
