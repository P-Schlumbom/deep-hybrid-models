import torch
import torch.nn as nn
from torch.nn import Parameter

__all__ = ['ActNorm1d', 'ActNorm2d']


class ActNormNd(nn.Module):

    def __init__(self, num_features, eps=1e-12):
        super(ActNormNd, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.weight = Parameter(torch.Tensor(num_features))
        self.bias = Parameter(torch.Tensor(num_features))
        self.register_buffer('initialized', torch.tensor(0))

    @property
    def shape(self):
        raise NotImplementedError

    def forward(self, x, logpx=None):
        #print(f"ActNormNd: x={torch.mean(x).item()}, logpx={logpx}")
        c = x.size(1)

        if not self.initialized:
            with torch.no_grad():
                # compute batch statistics
                x_t = x.transpose(0, 1).contiguous().view(c, -1)
                #print(f"x_t: {torch.mean(x_t).item()}, x_t shape: {x_t.shape}")
                batch_mean = torch.mean(x_t, dim=1)
                batch_var = torch.var(x_t, dim=1)
                #print(f"batch_var: {torch.mean(batch_var).item()}")

                # for numerical issues
                batch_var = torch.max(batch_var, torch.tensor(0.2).to(batch_var))

                self.bias.data.copy_(-batch_mean)
                self.weight.data.copy_(-0.5 * torch.log(batch_var))
                self.initialized.fill_(1)

        bias = self.bias.view(*self.shape).expand_as(x)
        weight = self.weight.view(*self.shape).expand_as(x)

        #print(f"bias: {torch.mean(bias).item()}, weight: {torch.mean(weight).item()}")
        y = (x + bias) * torch.exp(weight)

        if logpx is None:
            return y
        else:
            #print(f"Actnorm returns: y={torch.mean(y).item()}, logdetgrad={torch.mean(self._logdetgrad(x)).item()}, new logpx={torch.mean(logpx - self._logdetgrad(x)).item()}")
            return y, logpx - self._logdetgrad(x)

    def inverse(self, y, logpy=None):
        assert self.initialized
        bias = self.bias.view(*self.shape).expand_as(y)
        weight = self.weight.view(*self.shape).expand_as(y)

        x = y * torch.exp(-weight) - bias

        if logpy is None:
            return x
        else:
            return x, logpy + self._logdetgrad(x)

    def _logdetgrad(self, x):
        return self.weight.view(*self.shape).expand(*x.size()).contiguous().view(x.size(0), -1).sum(1, keepdim=True)

    def __repr__(self):
        return ('{name}({num_features})'.format(name=self.__class__.__name__, **self.__dict__))


class ActNorm1d(ActNormNd):

    @property
    def shape(self):
        return [1, -1]


class ActNorm2d(ActNormNd):

    @property
    def shape(self):
        return [1, -1, 1, 1]
