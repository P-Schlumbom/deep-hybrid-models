import numpy as np
import torch
from torch import nn
from torch.distributions import MultivariateNormal, Normal, multivariate_normal, normal


def log_sum_exp(x):
    """
    compute the log of the sum of the exponents of x, using the log-sum-exp trick
    :param x:
    :return:
    """
    maxvalue = torch.max(x, dim=1).values.view(-1, 1)
    x -= maxvalue
    #print("x:", x.shape)
    #print("sum(x):", torch.sum(x, dim=1).shape)
    #print("maxval + sum(x):", (maxvalue + torch.sum(x, dim=1).view(-1, 1)).shape)
    return maxvalue + torch.log(torch.sum(torch.exp(x), dim=1).view(-1, 1))


class NormalModel(nn.Module):
    def __init__(self, d, loc=None, scale=None, diagonal=True):
        super().__init__()
        self.d = d
        if not loc:
            loc = np.zeros(d)
            scale = np.ones(d)
        self.mean = nn.parameter.Parameter(torch.tensor(loc, requires_grad=True))
        self.cov = nn.parameter.Parameter(torch.tensor(scale, requires_grad=True))
        self.normal = MultivariateNormal(loc=self.mean.view(1, d), scale_tril=torch.diag(self.cov).view(-1, d, d))

    def forward(self, x):
        self.cov.data = torch.clamp(self.cov.data, min=1e-6)
        mv_normal = MultivariateNormal(loc=self.mean.view(1, self.d), scale_tril=torch.diag(self.cov).view(-1, self.d, self.d))
        return mv_normal.log_prob(x)


class GaussianMixtureModel(nn.Module):
    def __init__(self, n_components, dimensions):
        super().__init__()
        self.n_components = n_components
        self.log_n_components = np.log(n_components)
        self.dimensions = dimensions
        self.means = nn.parameter.Parameter(torch.randn((n_components, dimensions)))  # for each dimension, n components
        self.covs = nn.parameter.Parameter(torch.randn((n_components, dimensions)))  # for each component, keep a dimensions size vector of diagonal covariances
        self.c_mvns = [MultivariateNormal(loc=torch.zeros(dimensions), scale_tril=torch.diag(torch.ones(dimensions))) for i in range(n_components)]
        #self.c_log_probs = torch.zeros(n_components)

    def forward(self, x):
        self.covs.data = torch.clamp(self.covs.data, min=1e-6)
        c_log_probs = []
        for i in range(self.n_components):
            #self.covs[i].data = torch.clamp(self.covs[i].data, min=1e-6)
            self.c_mvns[i] = MultivariateNormal(loc=self.means[i].view(1, self.dimensions), scale_tril=torch.diag(self.covs[i]).view(-1, self.dimensions, self.dimensions))
            c_log_probs.append(self.c_mvns[i].log_prob(x).view(-1, 1))
        c_log_probs = torch.cat(c_log_probs, dim=1)  # should be [batch, n_components]

        return log_sum_exp(c_log_probs) - self.log_n_components
        #return torch.logsumexp(c_log_probs, dim=1, keepdim=True) - self.log_n_components


