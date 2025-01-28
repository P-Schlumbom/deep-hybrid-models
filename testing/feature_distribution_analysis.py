from tqdm import tqdm
from math import log
import numpy as np
from numpy import trapezoid as trapz
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
from architectures.deep_hybrid_models.dhm import DHM_iresflows, create_flow_model, normalise_features
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


def plot_histogram(A, B=None, C=None, title="test", tgt_name="test", a_label="a", b_label="b", c_label="c", n_bins=100):
    A, A_edges = create_online_hist(A, n_bins=n_bins)
    plt.bar(A_edges[:-1], A, width=np.diff(A_edges), alpha=0.5, label=a_label, align='edge')

    if B is not None:
        B, B_edges = create_online_hist(B, n_bins=n_bins)
        plt.bar(B_edges[:-1], B, width=np.diff(B_edges), alpha=0.5, label=b_label, align='edge')
    if C is not None:
        C, C_edges = create_online_hist(C, n_bins=n_bins)
        plt.bar(C_edges[:-1], C, width=np.diff(C_edges), alpha=0.5, label=c_label, align='edge')

    # Add a legend and title to the plot
    plt.legend(loc='upper right')
    plt.title(title)

    # save the plot
    plt.tight_layout()
    plt.savefig(f"results/feature_analysis/{tgt_name}_hist.png")

    # Show the plot
    plt.show()


def batch_operation(model, batch):
    inputs, _ = batch
    inputs = inputs.to(device)

    # forward + backward + optimize
    y, logpz, logdet, z, features = model(inputs, return_features=True)
    logpx = compute_logpx(logpz, logdet)
    nf_features = normalise_features(features)

    features = features.view(-1).detach().cpu().numpy().tolist()
    normalised_features = nf_features.view(-1).detach().cpu().numpy().tolist()
    probs = logpx.detach().cpu().numpy().tolist()

    return features, normalised_features, probs

# ---------------------------------------------------------------------------------------------------------------------#
# ---------------------------------TESTING FUNCTIONS-------------------------------------------------------------------#
# ---------------------------------------------------------------------------------------------------------------------#


def feature_analysis(dhm, batch_size=None, tgt_name=None):
    if batch_size is None:
        batch_size = 128
    if tgt_name is None:
        tgt_name = "test"
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.25, 0.25, 0.25)),
         ])

    C10_features = []

    CIFAR10_dataset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                            download=True, transform=transform)

    CIFAR10_loader = torch.utils.data.DataLoader(CIFAR10_dataset, batch_size=batch_size,
                                              shuffle=True, num_workers=2)

    print("computing CIFAR10 feature max vals...")
    for i, data in enumerate(tqdm(CIFAR10_loader)):
        inputs, _ = data
        inputs = inputs.to(device)

        # forward + backward + optimize
        y, logpz, logdet, z, features = dhm(inputs, return_features=True)
        logpx = compute_logpx(logpz, logdet)
        nf_features = normalise_features(features)

        C10_features = features.view(-1).detach().cpu().numpy().tolist()
        C10_normalised_features = nf_features.view(-1).detach().cpu().numpy().tolist()
        C10_probs = logpx.detach().cpu().numpy().tolist()

        break
    C10_featuremean = np.mean(C10_features)
    print(f"CIFAR10 feature mean: {C10_featuremean:.5g}, std: {np.std(C10_features):.5g}")
    print(f"CIFAR10 min, mean, and max feature max: {np.min(C10_features)}, {C10_featuremean:.5g}, {np.max(C10_features):.5g}")

    # Plot histograms for each dataset

    ## sample feature vals

    plot_histogram(C10_features, title=f'{tgt_name}: Histogram of CIFAR10 image features', tgt_name=f"{tgt_name}_feature_vals", a_label='CIFAR10')

    ## normalised feature vals

    plot_histogram(C10_normalised_features, title=f'{tgt_name}: Histogram of CIFAR10 normalised image features', tgt_name=f"{tgt_name}_normalised_feature_vals", a_label='CIFAR10')


def data_compare_feature_analysis(dhm, batch_size=None, tgt_name=None):
    if batch_size is None:
        batch_size = 128
    if tgt_name is None:
        tgt_name = "test"
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.25, 0.25, 0.25)),
         ])

    C10_features = []

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
        C10_features, C10_normalised_features, C10_probs = batch_operation(dhm, data)
        break
    for i, data in enumerate(tqdm(CIFAR100_loader)):
        C100_features, C100_normalised_features, C100_probs = batch_operation(dhm, data)
        break
    for i, data in enumerate(tqdm(SVHN_loader)):
        SVHN_features, SVHN_normalised_features, SVHN_probs = batch_operation(dhm, data)
        break
    C10_featuremean = np.mean(C10_features)
    print(f"CIFAR10 feature mean: {C10_featuremean:.5g}, std: {np.std(C10_features):.5g}")
    print(f"CIFAR10 min, mean, and max feature max: {np.min(C10_features)}, {C10_featuremean:.5g}, {np.max(C10_features):.5g}")

    # Plot histograms for each dataset

    ## sample feature vals

    plot_histogram(C10_features, B=C100_features, C=SVHN_features, title=f'{tgt_name}: Histogram of image features',
                   tgt_name=f"{tgt_name}_comparative_feature_vals", a_label='CIFAR10', b_label='CIFAR100',
                   c_label='SVHN')

    ## normalised feature vals

    plot_histogram(C10_normalised_features, B=C100_normalised_features, C=SVHN_normalised_features,
                   title=f'{tgt_name}: Histogram of normalised image features',
                   tgt_name=f"{tgt_name}_comparative_normalised_feature_vals", a_label='CIFAR10', b_label='CIFAR100',
                   c_label='SVHN')

    ## probabilities

    plot_histogram(C10_probs, B=C100_probs, C=SVHN_probs,
                   title=f'{tgt_name}: Histogram of image probs',
                   tgt_name=f"{tgt_name}_comparative_image_probs", a_label='CIFAR10', b_label='CIFAR100',
                   c_label='SVHN')


def test_feature_scaling(dhm, batch_size=None, tgt_name=None):
    if batch_size is None:
        batch_size = 128
    if tgt_name is None:
        tgt_name = "test"
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.25, 0.25, 0.25)),
         ])

    C10_features = []

    CIFAR10_dataset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                                   download=True, transform=transform)

    CIFAR10_loader = torch.utils.data.DataLoader(CIFAR10_dataset, batch_size=batch_size,
                                                 shuffle=True, num_workers=2)

    print("computing CIFAR10 feature max vals...")
    for i, data in enumerate(tqdm(CIFAR10_loader)):
        inputs, _ = data
        inputs = inputs.to(device)

        # forward + backward + optimize
        y, logpz, logdet, z, features = dhm(inputs, return_features=True)
        logpx = compute_logpx(logpz, logdet)

        # test scaled features
        features = normalise_features(features)
        scaled_features = features * 2
        scaled_logpz, scaled_logdet, scaled_z = dhm.flow(scaled_features)
        scaled_logpx = compute_logpx(scaled_logpz, scaled_logdet)

        features = features.view(-1).detach().cpu().numpy().tolist()
        scaled_features = scaled_features.view(-1).detach().cpu().numpy().tolist()
        probs = logpx.detach().cpu().numpy().tolist()
        scaled_probs = scaled_logpx.detach().cpu().numpy().tolist()
        break
    featuremean = np.mean(features)
    print(f"CIFAR10 feature mean: {featuremean:.5g}, std: {np.std(features):.5g}")
    print(f"CIFAR10 min, mean, and max feature max: {np.min(features)}, {featuremean:.5g}, {np.max(features):.5g}")

    # Plot histograms for each dataset

    plot_histogram(features, B=scaled_features, title=f'{tgt_name}: Histogram of image features',
                   tgt_name=f"{tgt_name}_scaled_features", a_label='default feature vals', b_label='scaled feature vals')

    plot_histogram(probs, B=scaled_probs, title=f'{tgt_name}: Histogram of image probs',
                   tgt_name=f"{tgt_name}_scaled_feature_probs", a_label='default probs', b_label='scaled feature probs')


def test_denser_dist(dhm, batch_size=None, tgt_name=None):
    if batch_size is None:
        batch_size = 128
    if tgt_name is None:
        tgt_name = "test"
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.25, 0.25, 0.25)),
         ])

    CIFAR10_dataset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                                   download=True, transform=transform)

    CIFAR10_loader = torch.utils.data.DataLoader(CIFAR10_dataset, batch_size=batch_size,
                                                 shuffle=True, num_workers=2)

    print("computing CIFAR10 feature max vals...")
    for i, data in enumerate(tqdm(CIFAR10_loader)):
        inputs, _ = data
        inputs = inputs.to(device)

        # forward + backward + optimize
        y, logpz, logdet, z, features = dhm(inputs, return_features=True)
        logpx = compute_logpx(logpz, logdet)

        # test scaled features
        features = normalise_features(features)
        mu_f = torch.mean(features)
        denser_features = ((features - mu_f) / 2) + mu_f
        denser_logpz, denser_logdet, denser_z = dhm.flow(denser_features)
        denser_logpx = compute_logpx(denser_logpz, denser_logdet)

        features = features.view(-1).detach().cpu().numpy().tolist()
        denser_features = denser_features.view(-1).detach().cpu().numpy().tolist()
        probs = logpx.detach().cpu().numpy().tolist()
        denser_probs = denser_logpx.detach().cpu().numpy().tolist()
        break
    featuremean = np.mean(features)
    print(f"CIFAR10 feature mean: {featuremean:.5g}, std: {np.std(features):.5g}")
    print(f"CIFAR10 min, mean, and max feature max: {np.min(features)}, {featuremean:.5g}, {np.max(features):.5g}")

    # Plot histograms for each dataset

    plot_histogram(features, B=denser_features, title=f'{tgt_name}: Histogram of image features',
                   tgt_name=f"{tgt_name}_denser_features", a_label='default feature vals', b_label='denser feature vals')

    plot_histogram(probs, B=denser_probs, title=f'{tgt_name}: Histogram of image probs',
                   tgt_name=f"{tgt_name}_denser_feature_probs", a_label='default probs', b_label='denser feature probs')

