import torch

from architectures.normalising_flows.residual_flows import layers as layers
from architectures.normalising_flows.residual_flows.layers import base as base_layers

ACTIVATION_FNS = {
    'relu': torch.nn.ReLU,
    'tanh': torch.nn.Tanh,
    'elu': torch.nn.ELU,
    'selu': torch.nn.SELU,
    'fullsort': base_layers.FullSort,
    'maxmin': base_layers.MaxMin,
    'swish': base_layers.Swish,
    'lcube': base_layers.LipschitzCube,
}


# ---------------------------MODEL DEFINITION------------------------------------------------------------------------- #


def parse_vnorms(vnorms):
    ps = []
    for p in vnorms:
        if p == 'f':
            ps.append(float('inf'))
        else:
            ps.append(float(p))
    return ps[:-1], ps[1:]


def build_nnet(
        dims,
        activation_fn=torch.nn.ReLU,
        vnorms='222222',
        learn_p=False,
        mixed=True,
        coeff=0.9,
        n_lipschitz_iters=5,
        atol=None,
        rtol=None,
):
    nnet = []
    domains, codomains = parse_vnorms(vnorms)
    if learn_p:
        if mixed:
            domains = [torch.nn.Parameter(torch.tensor(0.)) for _ in domains]
        else:
            domains = [torch.nn.Parameter(torch.tensor(0.))] * len(domains)
        codomains = domains[1:] + [domains[0]]
    for i, (in_dim, out_dim, domain, codomain) in enumerate(zip(dims[:-1], dims[1:], domains, codomains)):
        nnet.append(activation_fn())
        nnet.append(
            base_layers.get_linear(
                in_dim,
                out_dim,
                coeff=coeff,
                n_iterations=n_lipschitz_iters,
                atol=atol,
                rtol=rtol,
                domain=domain,
                codomain=codomain,
                zero_init=(out_dim == 2),
            )
        )
    return torch.nn.Sequential(*nnet)


def create_flow_model(
        input_size,
        dims='128-128-128-128',
        actnorm=False,
        n_blocks=100,
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
        coeff=0.9,
        n_lipschitz_iters=5,
        atol=None,
        rtol=None,
        init_layer=None,
):
    dims = [input_size] + list(map(int, dims.split('-'))) + [input_size]
    print(dims)
    blocks = []
    if init_layer:
        blocks.append(init_layer)
    if actnorm: blocks.append(layers.ActNorm1d(input_size))
    for _ in range(n_blocks):
        blocks.append(
            layers.iResBlock(
                build_nnet(
                    dims,
                    ACTIVATION_FNS[act],
                    vnorms=vnorms,
                    learn_p=learn_p,
                    mixed=mixed,
                    coeff=coeff,
                    n_lipschitz_iters=n_lipschitz_iters,
                    atol=atol,
                    rtol=rtol
                ),
                n_dist=n_dist,
                n_power_series=n_power_series,
                exact_trace=exact_trace,
                brute_force=brute_force,
                n_samples=n_samples,
                neumann_grad=False,
                grad_in_forward=False,
            )
        )
        if actnorm: blocks.append(layers.ActNorm1d(input_size))
        if batchnorm: blocks.append(layers.MovingBatchNorm1d(input_size))
    model = layers.SequentialFlow(blocks)  #.to(device)

    return model

