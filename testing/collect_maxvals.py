from tqdm import tqdm
from math import log
import numpy as np
from numpy import trapz
from os.path import join
import matplotlib as mpl
from matplotlib import pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import MultiStepLR
import torchvision
import torchvision.transforms as transforms

import argparse

import sys
import os

# Get the absolute path of the directory containing your script
script_directory = os.path.dirname(os.path.abspath(__file__))

# Append the directory containing the modules to the sys.path list
module_directory = os.path.join(script_directory, "..")
sys.path.append(module_directory)

from architectures.resnets.blocks import ModernBasicBlock
from architectures.resnets.wide_resnet import WideResNet
from architectures.normalising_flows.glow import Glow
from architectures.normalising_flows.residual_flows.residual_flow import ResidualFlow, ACT_FNS, create_resflow
from architectures.normalising_flows.residual_flows.layers.elemwise import LogitTransform, Normalize, IdentityTransform
from architectures.normalising_flows.residual_flows.layers.squeeze import SqueezeLayer
from architectures.deep_hybrid_models.dhm import DHM, define_flow_model
from helpers.utils import running_average, print_model_params, get_model_params


mpl.rcParams['figure.figsize'] = (10, 6)

# set seeds
torch.manual_seed(0)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

parser = argparse.ArgumentParser(description="DHM testing")

parser.add_argument("--name", type=str, help="name of the file to load")
parser.add_argument("--dirpath", type=str, default="checkpoints/testing",
                    help="path of the dir to load the model from (checkpoints/testing by default)")
parser.add_argument("--batch", default=64, type=int, help="batch size")


# -------------------------------------------------------------------------------------------------------------------- #
# ----------------------------------HELPERS--------------------------------------------------------------------------- #
# -------------------------------------------------------------------------------------------------------------------- #


def local_define_flow_model(args, in_shape=None):
    if in_shape is None:
        in_shape = (args['batch'], 64 * args['k'], 2, 2)
    else:
        in_shape = (args['batch'], 64 * args['k'], in_shape[0], in_shape[1])
    if args['nf_type'] == "resflow":
        #init_layer = LogitTransform(0.05)  # TODO: need to check how this value is actually found...
        init_layer = IdentityTransform()
        if args['squeeze_first']:
            input_size = (64 * args['k'], 2, 2)
            squeeze_layer = SqueezeLayer(2)
        return create_resflow(
            in_shape,
            n_blocks=[args['n_flows'] for i in range(args['n_blocks'])],
            intermediate_dim=args['idim'],
            factor_out=args['factor_out'],
            init_layer=init_layer,
            actnorm=args['actnorm'],
            activation_fn=args['act'],
            fc_end=args['fc_end'],
            n_exact_terms=args['n_exact_terms'],
            fc=True
        )
    elif args['nf_type'] == "glow":
        return Glow(
            in_channel=(64 * args['k']),
            n_flow=args['n_flows'],
            n_block=args['n_blocks'],
            affine=args['affine'],
            conv_lu=not args['no_lu'],
            filter_size=args['idim'],
        )
    else:
        print("Error! normalising flow type '{}' is not a valid option.".format(args['nf_type']))
        raise Exception


def compute_uncertainty(y, logp, logdet):
    #y_logits = nn.functional.softmax(y)
    #y_logits = y_logits.gather(1, labels.unsqueeze(1)).squeeze()  # select the predicted logit values for the correct y
    #log_y = torch.log(y_logits)

    logpy = torch.nn.functional.log_softmax(y, dim=1)  # apply softmax to each y prediction (log_softmax supposed to be
    # more efficient than doing log and softmax separately)
    logpy = torch.max(logpy, dim=1, keepdim=True)[0]  # keep only the max y for each prediction
    #print(logpy.mean(), logp.mean(), logdet.mean())
    return torch.mean(logpy + logp + logdet, 1, True)


def compute_logpx(logp, logdet):
    # question: should I be dividing this by the number of elements? To get the average probabiilty per input element?
    #return torch.mean(logp + logdet, 1, True)
    return logp - logdet  # p(x) = 1/|det| * p(z), logp(x) = -log(|det|) + logp(z), and logdet has already been reversed
    #return logp + logdet


def aggregate_data(x, scale=10):
    #x_min, x_max = min(x), max(x)
    #step = (x_max - x_min) / n_bins
    aggregated = []
    x.sort()
    for i in range(0, len(x)-1, scale):
        aggregated.append(np.mean(x[i*scale:i*scale+scale]))
    return aggregated


def create_online_hist(data, n_bins=50, batch=512):
    """
    Construct histogram in online fashion so that whole dataset doesn't have to be handled at once
    :param data:
    :param n_bins:
    :param batch:
    :return:
    """
    datamin, datamax = np.min(data), np.max(data)
    bins = np.linspace(datamin, datamax, n_bins)
    hist = np.zeros(n_bins-1, dtype='int32')
    for i in range(0, len(data), batch):
        d = data[i*batch:i*batch + batch]
        htemp, edges = np.histogram(d, bins)
        hist += htemp
    return hist, edges


def compute_precision(baseline_data, ood_data, thresh=None):
    """
    precision = ntp / (ntp + nfp)
    :param baseline_data:
    :param A:
    :param B:
    :param thresh:
    :return:
    """
    if thresh is None:
        thresh = np.min(baseline_data)
    ntp = (baseline_data >= thresh).sum()
    nfp = (ood_data >= thresh).sum()
    return ntp / (ntp + nfp)


def compute_recall(baseline_data, ood_data, thresh=None):
    """
    recall = ntp / (ntp + nfn)
    :param baseline_data:
    :param A:
    :param B:
    :param thresh:
    :return:
    """
    if thresh is None:
        thresh = np.min(baseline_data)
    ntp = (baseline_data >= thresh).sum()
    nfn = (baseline_data < thresh).sum()
    return ntp / (ntp + nfn)


def compute_fscore(baseline_data, ood_data, thresh=None):
    """
    f-score = 2 * (precision * recall) / (precision + recall)
    :param baseline_data:
    :param A:
    :param B:
    :param thresh:
    :return:
    """
    precision = compute_precision(baseline_data, ood_data, thresh)
    recall = compute_recall(baseline_data, ood_data, thresh)
    return 2 * (precision * recall) / (precision + recall)


def compute_AUROC(baseline_data, ood_data, n_samples=500):
    #bmin, bmax = np.min(baseline_data), np.max(baseline_data)
    bmin, bmax = min(np.min(baseline_data), np.min(ood_data)), max(np.max(baseline_data), np.max(ood_data))
    brange = bmax - bmin
    bstep = brange / n_samples

    TPR, FPR = [], []
    for i in range(n_samples + 1):
        threshold = bmin + (i*bstep)
        tpr = (baseline_data >= threshold).sum() / len(baseline_data)
        fpr = ((ood_data >= threshold).sum()) / (len(ood_data))
        TPR.append(tpr)
        FPR.append(fpr)
    return TPR, FPR


def compute_total_AUROC(baseline_data, A, B, n_samples=20):
    bmin, bmax = np.min(baseline_data), np.max(baseline_data)
    brange = bmax - bmin
    bstep = brange / n_samples

    TPR, FPR = [], []
    for i in range(n_samples + 1):
        threshold = bmin + (i*bstep)
        tpr = (baseline_data >= threshold).sum() / len(baseline_data)
        fpr = ((A >= threshold).sum() + (B >= threshold).sum()) / (len(A) + len(B))
        TPR.append(tpr)
        FPR.append(fpr)

    return TPR, FPR

# ---------------------------------------------------------------------------------------------------------------------#
# ---------------------------------TESTING FUNCTIONS-------------------------------------------------------------------#
# ---------------------------------------------------------------------------------------------------------------------#


def generate_histograms(dhm, batch_size=None, tgt_name=None):
    if batch_size is None:
        batch_size = args.batch
    if tgt_name is None:
        tgt_name = args.name.split('.')[0]
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.25, 0.25, 0.25)),
         ])

    C10_fmax= []
    C100_fmax = []
    SVHN_fmax = []

    CIFAR10_dataset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                            download=True, transform=transform)
    CIFAR100_dataset = torchvision.datasets.CIFAR100(root='./data', train=False,
                                                     download=True, transform=transform)
    SVHN_dataset = torchvision.datasets.SVHN(root='./data', split='test',
                                             download=True, transform=transform)

    CIFAR10_loader = torch.utils.data.DataLoader(CIFAR10_dataset, batch_size=batch_size,
                                              shuffle=True, num_workers=2)
    CIFAR100_loader = torch.utils.data.DataLoader(CIFAR100_dataset, batch_size=batch_size,
                                                 shuffle=True, num_workers=2)
    SVHN_loader = torch.utils.data.DataLoader(SVHN_dataset, batch_size=batch_size,
                                                  shuffle=True, num_workers=2)

    print("computing CIFAR10 feature max vals...")
    for i, data in enumerate(tqdm(CIFAR10_loader)):
        inputs, _ = data
        inputs = inputs.to(device)

        # forward + backward + optimize
        y, logpz, logdet, z, features = dhm(inputs, return_features=True)
        fmax = torch.amax(features, dim=(1, 2, 3), keepdim=True)  # per-sample max
        #fmax = torch.max(features)  # total batch max
        #fmax = features.view(batch_size, -1).mean(1, keepdim=True)  # per-sample mean
        C10_fmax.extend(fmax.detach().cpu().numpy().tolist())
    print("computing CIFAR100 feature max vals...")
    for i, data in enumerate(tqdm(CIFAR100_loader)):
        inputs, _ = data
        inputs = inputs.to(device)

        # forward + backward + optimize
        y, logpz, logdet, z, features = dhm(inputs, return_features=True)
        fmax = torch.amax(features, dim=(1, 2, 3), keepdim=True)
        #fmax = torch.max(features)
        #fmax = features.view(batch_size, -1).mean(1, keepdim=True)  # per-sample mean
        C100_fmax.extend(fmax.detach().cpu().numpy().tolist())
    print("computing SVHN feature max vals...")
    for i, data in enumerate(tqdm(SVHN_loader)):
        inputs, _ = data
        inputs = inputs.to(device)

        # forward + backward + optimize
        y, logpz, logdet, z, features = dhm(inputs, return_features=True)
        fmax = torch.amax(features, dim=(1, 2, 3), keepdim=True)
        #fmax = torch.max(features)
        #fmax = features.view(batch_size, -1).mean(1, keepdim=True)  # per-sample mean
        SVHN_fmax.extend(fmax.detach().cpu().numpy().tolist())
    C10_meanfmax = np.mean(C10_fmax)
    C100_meanfmax = np.mean(C100_fmax)
    SVHN_meanfmax = np.mean(SVHN_fmax)
    print(f"CIFAR10 mean feature max: {C10_meanfmax:.5g}, std: {np.std(C10_fmax):.5g}")
    print(f"CIFAR10 min, mean, and max feature max: {np.min(C10_fmax)}, {C10_meanfmax:.5g}, {np.max(C10_fmax):.5g}")
    print(f"CIFAR100 mean feature max: {C100_meanfmax:.5g}, std: {np.std(C100_fmax):.5g}")
    print(f"SVHN mean feature max: {SVHN_meanfmax:.5g}, std: {np.std(SVHN_fmax):.5g}")

    # create AUROC curve

    # tpr, fpr = compute_AUROC(CIFAR10_probs, CIFAR100_probs, SVHN_probs)
    cifar100_tpr, cifar100_fpr = compute_AUROC(C10_fmax, C100_fmax)
    cifar100_area = trapz(cifar100_tpr, x=cifar100_fpr) * -1
    print(f"CIFAR100 AUROC: {cifar100_area}")
    plt.plot(cifar100_fpr, cifar100_tpr, label=f"CIFAR100:{cifar100_area:.4f}")
    svhn_tpr, svhn_fpr = compute_AUROC(C10_fmax, SVHN_fmax)
    svhn_area = trapz(svhn_tpr, x=svhn_fpr) * -1
    print(f"SVHN AUROC: {svhn_area}")
    plt.plot(svhn_fpr, svhn_tpr, label=f"SVHN:{svhn_area:.4f}")
    plt.title(f"AUROC curves for CIFAR100 and SVHN")
    plt.legend()
    plt.savefig(f"results/feature_analysis/{tgt_name}_AUROC.png")
    plt.show()
    #plt.figure()

    # Plot histograms for each dataset

    C10_fmax, C10_edges = create_online_hist(C10_fmax, n_bins=100)
    C100_fmax, C100_edges = create_online_hist(C100_fmax, n_bins=100)
    SVHN_fmax, SVHN_edges = create_online_hist(SVHN_fmax, n_bins=100)
    plt.bar(C10_edges[:-1], C10_fmax, width=np.diff(C10_edges), alpha=0.5, label='CIFAR10', align='edge')
    plt.bar(C100_edges[:-1], C100_fmax, width=np.diff(C100_edges), alpha=0.5, label='CIFAR100', align='edge')
    plt.bar(SVHN_edges[:-1], SVHN_fmax, width=np.diff(SVHN_edges), alpha=0.5, label='SVHN', align='edge')

    # Add a legend and title to the plot
    plt.legend(loc='upper right')
    plt.title(f'{tgt_name}: Histograms of CIFAR10, CIFAR100, and SVHN image feature max values')

    # save the plot
    plt.tight_layout()
    plt.savefig(f"results/feature_analysis/{tgt_name}_feature_hist.png")

    # Show the plot
    plt.show()


# -------------------------------------------------------------------------------------------------------------------- #
# ---------------------------------MAIN FUNCTIONS--------------------------------------------------------------------- #
# -------------------------------------------------------------------------------------------------------------------- #


if __name__ == "__main__":
    args = parser.parse_args()

    model_path = join(args.dirpath, args.name)
    model_dict = torch.load(model_path)
    model_args = model_dict['args']

    # --- DEFINE DNN --- #
    # forming Wide ResNet 28-10, WRN 28-10:
    n = model_args['N'] * 6 + 4
    print("Creating model WRN-{}-{} with N={}".format(n, model_args['k'], model_args['N']))
    dnn = WideResNet(ModernBasicBlock, [model_args['N'], model_args['N'], model_args['N']], input_size=(32, 32, 3),
                     num_classes=10, k=model_args['k'], spectral_normalization=model_args['sn'],
                     n_power_iter=model_args['n_power_iter'], coeff=model_args['coeff'])
    dnn_outshape = dnn.out_size
    if model_args['flatten']: dnn_outshape = [1, 1]
    print(f"outshape: {dnn_outshape}")
    # model.to(device)
    print("number of parameters: {} ({:,})".format(get_model_params(dnn), get_model_params(dnn)))

    # --- DEFINE NF --- #
    print("Creating {} type normalising flow with {} blocks and {} flows each, hidden dimension {}".format(
        model_args['nf_type'],
        model_args['n_blocks'],
        model_args['n_flows'],
        model_args['idim']))
    #nf = define_flow_model(model_args, in_shape=dnn_outshape)
    nf = define_flow_model(model_args['batch'], model_args['k'], model_args['nf_type'], model_args['n_blocks'],
                           model_args['n_flows'], model_args['idim'], model_args['factor_out'], model_args['actnorm'],
                           model_args['act'], model_args['fc_end'], model_args['n_exact_terms'], model_args['affine'],
                           model_args['no_lu'], model_args['squeeze_first'], dnn_outshape, fc=model_args['fc'],
                           input_layer=model_args['init_layer'])
    print("number of parameters: {} ({:,})".format(get_model_params(nf), get_model_params(nf)))

    # --- DEFINE DHM --- #
    print("creating deep hybrid model with WRN-{}-{} dnn and {}-{}x{} nf".format(n, model_args['k'], model_args['nf_type'],
                                                                                 model_args['n_blocks'],
                                                                                 model_args['n_flows']))
    dhm = DHM(dnn, nf, normalise_features=model_args['normalise_features'], flatten_features=model_args['flatten'],
              flow_in_shape=[dnn_outshape[0], dnn_outshape[1], model_args['k'] * 64])
    # dhm.load_state_dict(torch.load("checkpoints/testing/20230404_dry-silence-52.pth"))
    print("number of parameters: {} ({:,})".format(get_model_params(dhm), get_model_params(dhm)))

    #dhm.load_state_dict(model_dict['state_dict'])
    dhm.to(device)
    dhm.eval()
    test_input = torch.randn((1, 3, 32, 32))
    test_output = dhm(test_input.to(device))
    print("test input done")
    dhm.load_state_dict(model_dict['state_dict'])
    print("saved parameters loaded")

    generate_histograms(dhm)
