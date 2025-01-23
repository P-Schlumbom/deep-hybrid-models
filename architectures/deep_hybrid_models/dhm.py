from os import makedirs
from os.path import join, exists

import numpy as np
import torch
from torch import nn
from math import log, pi

from architectures.resnets.blocks import ModernBasicBlock
from architectures.resnets.wide_resnet import WideResNet, spectral_norm_fc, SensitiveWideResNet
from architectures.normalising_flows.residual_flows.residual_flow_models import create_flow_model
from helpers.utils import running_average, print_model_params, get_model_params, print_model_architecture, set_seed


def standard_normal_logprob(z):
    logZ = -0.5 * log(2 * pi)
    return logZ - z.pow(2) / 2

# --------------------------NORMFLOWS MODEL--------------------------------------------------------------------------- #



def logit_transform(x, alpha=0.05):
    s = alpha + (1 - 2 * alpha) * x
    y = torch.log(s) - torch.log(1 - s)
    return y


def normalise_features(features, is_training=False, ord=None, target_dims=(1, 2, 3)):
    max_vals = torch.amax(features, dim=target_dims, keepdim=True)  # per-sample max
    min_vals = torch.amin(features, dim=target_dims, keepdim=True)  # per-sample min

    if not ord:
        nf_features = (features - min_vals + 1e-5) / (max_vals - min_vals + 2e-5)
    else:
        nf_features = nn.functional.normalize(features, dim=target_dims, p=ord)
        max_vals = torch.norm(features, dim=target_dims, p=ord)

    return nf_features, max_vals


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

        nf_features = nf_features.view(nf_features.size(0), -1)

        features = features.view(features.size(0), -1)

        z, logdet = self.flow(nf_features, init_probs)

        # compute logp(z)
        logpz = standard_normal_logprob(z).sum(1, keepdim=True)

        y = self.fc(features)
        if return_features:
            return y, logpz, logdet, z, nf_features
        return y, logpz, logdet, z, None

    def inverse(self, z):
        return None

    def log_prob(self, x, return_features=False):
        _, logpz, logdet, _, features = self.forward(x, return_features=return_features)
        return logpz - logdet, features

    def feature_logprob(self, features):
        zero = torch.zeros(features.shape[0], 1).to(features)
        features = features.view(features.size(0), -1)
        z, logdet = self.flow(features, zero)
        logpz = standard_normal_logprob(z).sum(1, keepdim=True)
        return logpz - logdet


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

    dnn_outshape = [1, 1]  # dnn.out_size
    if flatten: dnn_outshape = [1, 1]
    print(f"outshape: {dnn_outshape}")

    print("number of parameters: {} ({:,})".format(get_model_params(dnn), get_model_params(dnn)))

    # --- DEFINE NF --- #
    print("Creating ires normalising flow with {} blocks and {} flows each, hidden dimension {}. Init layer: {}".format(
        n_blocks,
        len(dims.split('-')),
        dims.split('-')[0],
        init_layer)
    )

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

    dhm = DHM_iresflows(dnn, nf, flow_in_shape=[dnn_outshape[0], dnn_outshape[1], k * 64],
                        normalise_features=normalise_features, flatten_features=flatten,
                        common_features=common_features, norm_ord=norm_ord, logit=(init_layer=="logit"))

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

