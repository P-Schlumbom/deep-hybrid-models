import numpy as np
import torch
import torch.nn as nn
from math import log, pi

from architectures.spectral_normalisation.due_sn.spectral_norm_fc import spectral_norm_fc
from architectures.spectral_normalisation.due_sn.spectral_norm_conv import spectral_norm_conv
from architectures.normalising_flows.residual_flows.residual_flow_models import create_flow_model


def standard_normal_logprob(z):
    logZ = -0.5 * log(2 * pi)
    return logZ - z.pow(2) / 2


class StripesDetector(nn.Module):
    def __init__(self, in_channels=1, spectral_normalisation=True, n_power_iter=1, coeff: float = 1.0):
        super().__init__()
        self.sn = spectral_normalisation
        self.n_power_iter = n_power_iter
        self.coeff = coeff

        #self.conv = nn.Conv2d(in_channels=in_channels, out_channels=2, kernel_size=3, bias=False)
        self.conv = spectral_norm_conv(nn.Conv2d(in_channels=in_channels, out_channels=2, kernel_size=3, bias=False), self.coeff, (1, 9, 9), self.n_power_iter)
        self.maxpool = nn.AdaptiveMaxPool2d((1, 1))

    def forward(self, x):
        x = self.conv(x)
        x = torch.abs(x)
        x = self.maxpool(x)
        y = x.view(x.size(0), -1)
        return y


class StripesDHM(nn.Module):
    def __init__(self, dims, n_blocks, dnn_coeff=0.98, n_features=2):
        super(StripesDHM, self).__init__()

        self.dnn = StripesDetector(coeff=dnn_coeff)
        self.flow = create_flow_model(
            n_features,
            dims=dims,
            actnorm=False,
            n_blocks=n_blocks,
            coeff=0.98,
        )
        #self.fc = self.dnn.fcf
        #self.fc = nn.Linear(2, n_classes)

    def forward(self, x):
        x = self.dnn(x)
        init_probs = torch.zeros(x.shape[0], 1).to(x)
        z, logdet = self.flow(x, init_probs)
        logpz = standard_normal_logprob(z).sum(1, keepdim=True)

        return x, logpz, logdet, z

    def feature_logprob(self, features):
        init_probs = torch.zeros(features.shape[0], 1).to(features)
        z, logdet = self.flow(features, init_probs)
        logpz = standard_normal_logprob(z).sum(1, keepdim=True)
        logpx = logpz - logdet
        return logpx

