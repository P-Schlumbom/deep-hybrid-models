import io
import base64
import sklearn.mixture
from tqdm import tqdm
from math import log
import numpy as np
from numpy import trapz
from os.path import join
import matplotlib as mpl
from matplotlib import pyplot as plt
import plotly.express as px
import plotly.graph_objs as go
# from dash import Dash, dcc, html, Input, Output, no_update, callback
import pandas as pd

from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import MultiStepLR
import torchvision
import torchvision.transforms as transforms

from sklearn.manifold import TSNE, Isomap, LocallyLinearEmbedding, MDS

import argparse

from architectures.resnets.blocks import ModernBasicBlock
from architectures.resnets.wide_resnet import WideResNet
from architectures.normalising_flows.glow import Glow
from architectures.normalising_flows.residual_flows.residual_flow import ResidualFlow, ACT_FNS, create_resflow
from architectures.normalising_flows.residual_flows.layers.elemwise import LogitTransform, Normalize, IdentityTransform
from architectures.normalising_flows.residual_flows.layers.squeeze import SqueezeLayer
from architectures.deep_hybrid_models.dhm import DHM, define_flow_model
from helpers.utils import running_average, print_model_params, get_model_params
from testing.single_batch_analysis import single_batch_analysis
from testing.feature_distribution_analysis import feature_analysis, data_compare_feature_analysis, test_feature_scaling, \
    test_denser_dist
from datasets.dataset_labels import c10_id2label, c100_id2label

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
        # init_layer = LogitTransform(0.05)  # TODO: need to check how this value is actually found...
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
    # y_logits = nn.functional.softmax(y)
    # y_logits = y_logits.gather(1, labels.unsqueeze(1)).squeeze()  # select the predicted logit values for the correct y
    # log_y = torch.log(y_logits)

    logpy = torch.nn.functional.log_softmax(y, dim=1)  # apply softmax to each y prediction (log_softmax supposed to be
    # more efficient than doing log and softmax separately)
    logpy = torch.max(logpy, dim=1, keepdim=True)[0]  # keep only the max y for each prediction
    # print(logpy.mean(), logp.mean(), logdet.mean())
    return torch.mean(logpy + logp + logdet, 1, True)


def compute_logpx(logp, logdet):
    # question: should I be dividing this by the number of elements? To get the average probabiilty per input element?
    # return torch.mean(logp + logdet, 1, True)
    return logp - logdet  # p(x) = 1/|det| * p(z), logp(x) = -log(|det|) + logp(z), and logdet has already been reversed
    # return logp + logdet


def aggregate_data(x, scale=10):
    # x_min, x_max = min(x), max(x)
    # step = (x_max - x_min) / n_bins
    aggregated = []
    x.sort()
    for i in range(0, len(x) - 1, scale):
        aggregated.append(np.mean(x[i * scale:i * scale + scale]))
    return aggregated


def compute_thresholds(baseline_data, ood_data, n_thresholds):
    # Ensure both inputs are numpy arrays
    baseline_data = np.array(baseline_data)
    ood_data = np.array(ood_data)

    # Combine the baseline and OOD data
    combined_data = np.concatenate([baseline_data, ood_data])

    # n_thresholds should be at most 1/2 the number of samples otherwise it won't really make sense.
    n_thresholds = min(len(combined_data)//2, n_thresholds)

    # Calculate n+1 quantiles to get n thresholds
    quantile_values = np.quantile(combined_data, np.linspace(0, 1, n_thresholds + 1))

    # Return the calculated thresholds
    return quantile_values


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
    hist = np.zeros(n_bins - 1, dtype='int32')
    for i in range(0, len(data), batch):
        d = data[i * batch:i * batch + batch]
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
    # bmin, bmax = np.min(baseline_data), np.max(baseline_data)
    #bmin, bmax = min(np.min(baseline_data), np.min(ood_data)), max(np.max(baseline_data), np.max(ood_data))
    #brange = bmax - bmin
    #bstep = brange / n_samples

    TPR, FPR = [], []
    thresholds = compute_thresholds(baseline_data, ood_data, n_samples)
    for threshold in thresholds:
        tpr = (baseline_data >= threshold).sum() / len(baseline_data)
        fpr = ((ood_data >= threshold).sum()) / (len(ood_data))
        TPR.append(tpr)
        FPR.append(fpr)
    return TPR, FPR


def compute_AUPR(baseline_data, comparison_data, n_samples=500):
    """
    Compute the Area Under Precision curve, precision vs recall (=TPR)
    :param baseline_data:
    :param comparison_data:
    :param n_samples:
    :return:
    """
    bmin, bmax = min(np.min(baseline_data), np.min(comparison_data)), max(np.max(baseline_data), np.max(comparison_data))
    brange = bmax - bmin
    bstep = brange / n_samples

    precision, recall = [], []
    thresholds = compute_thresholds(baseline_data, comparison_data, n_samples)
    for threshold in thresholds:
        tpr = (baseline_data >= threshold).sum() / len(baseline_data)
        prec = ((baseline_data >= threshold).sum()) / ((baseline_data >= threshold).sum() + (comparison_data >= threshold).sum())
        recall.append(tpr)
        precision.append(prec)
    return precision, recall


def compute_AUTC(baseline_data, comparison_data, n_samples=500):
    """
    Compute the Area Under the Threshold Curve, as described by Humblot-Renaux et al. ()
    Computes the AUFPR, AUFNR, and finally AUTC = (AUFPR + AUFNR)/2
    :param baseline_data:
    :param comparison_data:
    :param n_samples:
    :return:
    """
    #bmin, bmax = min(np.min(baseline_data), np.min(comparison_data)), max(np.max(baseline_data),
    #                                                                      np.max(comparison_data))
    #thresholds = np.linspace(bmin, bmax, n_samples)

    FPR, FNR = [], []
    thresholds = compute_thresholds(baseline_data, comparison_data, n_samples)
    for threshold in thresholds:
        fpr = (comparison_data >= threshold).sum() / len(comparison_data)
        fnr = ((baseline_data < threshold).sum()) / (len(baseline_data))
        FPR.append(fpr)
        FNR.append(fnr)
    scaled_thresholds = (thresholds - min(thresholds)) / (max(thresholds) - min(thresholds))
    AUFPR = trapz(FPR, x=scaled_thresholds)
    AUFNR = trapz(FNR, x=scaled_thresholds)
    AUTC = (AUFPR + AUFNR) / 2
    return AUTC


def compute_total_AUROC(baseline_data, A, B, n_samples=20):
    bmin, bmax = np.min(baseline_data), np.max(baseline_data)
    brange = bmax - bmin
    bstep = brange / n_samples

    TPR, FPR = [], []
    for i in range(n_samples + 1):
        threshold = bmin + (i * bstep)
        tpr = (baseline_data >= threshold).sum() / len(baseline_data)
        fpr = ((A >= threshold).sum() + (B >= threshold).sum()) / (len(A) + len(B))
        TPR.append(tpr)
        FPR.append(fpr)

    return TPR, FPR


def shift_towards_mean(inputs):
    # print(inputs.shape)
    s_mu = inputs.mean(axis=1).unsqueeze(-1)
    sh_inputs = ((inputs - s_mu) / 2.0) + s_mu
    return sh_inputs


def get_max_softmaxes(y):
    """
    return the maximum softmax value for each sample
    :param y:
    :return:
    """
    softmaxes = nn.functional.softmax(y, dim=1)
    max_vals, _ = torch.max(softmaxes, dim=1, keepdim=True)
    return max_vals


def get_labelled_softmaxes(y, labels):
    """
    return the softmax value of the correct logit for each sample
    :param y:
    :param labels:
    :return:
    """
    softmaxes = nn.functional.softmax(y, dim=1)
    softmaxes = torch.gather(softmaxes, 1, labels.unsqueeze(1)).squeeze()
    return softmaxes


# ---------------------------------------------------------------------------------------------------------------------#
# ---------------------------------TESTING FUNCTIONS-------------------------------------------------------------------#
# ---------------------------------------------------------------------------------------------------------------------#

def get_logpx_loop(dhm: DHM, inputs, test_unimodal=False):
    # y, logpz, logdet, z = model(inputs)

    # return compute_logpx(logpz, logdet)
    logpx, features = dhm.log_prob(inputs, return_features=test_unimodal)
    if test_unimodal:
        sh_features = shift_towards_mean(features)
        sh_logpx = dhm.feature_logprob(sh_features)
        return logpx, sh_logpx
    return logpx, None


def generate_histograms(dhm, batch_size=None, tgt_name=None, plot=False, test_unimodal=False):
    if batch_size is None:
        batch_size = args.batch
    if tgt_name is None:
        tgt_name = args.name.split('.')[0]
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize(np.array([125.3, 123.0, 113.9]) / 255.0,
                              np.array([63.0, 62.1, 66.7]) / 255.0),
         ])

    CIFAR10_probs, CIFAR10_shifted_probs = [], []
    CIFAR100_probs, CIFAR100_shifted_probs = [], []
    SVHN_probs, SVHN_shifted_probs = [], []

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

    print("computing CIFAR10 probabilities...")
    for i, data in enumerate(tqdm(CIFAR10_loader)):
        inputs, _ = data
        inputs = inputs.to(device)

        """if logpx_loop:
            logpx = logpx_loop(dhm, inputs)
        else:
            logpx = get_logpx_loop(dhm, inputs)"""
        logpx, sh_logpx = get_logpx_loop(dhm, inputs, test_unimodal=test_unimodal)

        if test_unimodal: CIFAR10_shifted_probs.extend(sh_logpx.detach().cpu().numpy().tolist())
        CIFAR10_probs.extend(logpx.detach().cpu().numpy().tolist())
        # if i > 100:
        #    break
    print("computing CIFAR100 probabilities...")
    for i, data in enumerate(tqdm(CIFAR100_loader)):
        inputs, _ = data
        inputs = inputs.to(device)

        logpx, sh_logpx = get_logpx_loop(dhm, inputs, test_unimodal=test_unimodal)

        if test_unimodal: CIFAR100_shifted_probs.extend(sh_logpx.detach().cpu().numpy().tolist())
        CIFAR100_probs.extend(logpx.detach().cpu().numpy().tolist())
        # if i > 100:
        #    break
    print("computing SVHN probabilities...")
    for i, data in enumerate(tqdm(SVHN_loader)):
        inputs, _ = data
        inputs = inputs.to(device)

        logpx, sh_logpx = get_logpx_loop(dhm, inputs, test_unimodal=test_unimodal)

        if test_unimodal: SVHN_shifted_probs.extend(sh_logpx.detach().cpu().numpy().tolist())
        SVHN_probs.extend(logpx.detach().cpu().numpy().tolist())
        # if i > 100:
        #    break
    print(len(CIFAR10_probs))
    C10_meanprob = np.mean(CIFAR10_probs)
    C100_meanprob = np.mean(CIFAR100_probs)
    SVHN_meanprob = np.mean(SVHN_probs)
    print(f"CIFAR10 mean prob: {C10_meanprob:.5g}, std: {np.std(CIFAR10_probs):.5g}")
    print(f"CIFAR100 mean prob: {C100_meanprob:.5g}, std: {np.std(CIFAR100_probs):.5g}")
    print(f"SVHN mean prob: {SVHN_meanprob:.5g}, std: {np.std(SVHN_probs):.5g}")
    print(
        f"(mean(C10) - std(C10)) - (mean(C100) + std(C100)): {(C10_meanprob - np.std(CIFAR10_probs)) - (C100_meanprob + np.std(CIFAR100_probs))}")

    # create AUROC curve
    # tpr, fpr = compute_AUROC(CIFAR10_probs, CIFAR100_probs, SVHN_probs)
    cifar100_tpr, cifar100_fpr = compute_AUROC(CIFAR10_probs, CIFAR100_probs)
    cifar100_area = trapz(cifar100_tpr, x=cifar100_fpr) * -1
    print(f"CIFAR100 AUROC: {cifar100_area}")
    svhn_tpr, svhn_fpr = compute_AUROC(CIFAR10_probs, SVHN_probs)
    svhn_area = trapz(svhn_tpr, x=svhn_fpr) * -1
    print(f"SVHN AUROC: {svhn_area}")

    cifar100_precision = compute_precision(CIFAR10_probs, CIFAR100_probs)
    print(f"CIFAR100 precision: {cifar100_precision:.5f}")
    svhn_precision = compute_precision(CIFAR10_probs, SVHN_probs)
    print(f"SVHN precision: {svhn_precision:.5f}")

    # compute AUPR scores
    # Cifar100 in and out AUPR scores
    cifar100_in_precision, cifar100_recall = compute_AUPR(CIFAR10_probs, CIFAR100_probs)
    cifar100_in_area = trapz(cifar100_in_precision, x=cifar100_recall) * -1
    cifar100_out_precision, cifar100_recall = compute_AUPR(CIFAR100_probs, CIFAR10_probs)
    cifar100_out_area = trapz(cifar100_out_precision, x=cifar100_recall) * -1
    # SVHN in and out AUPR scores
    svhn_in_precision, svhn_recall = compute_AUPR(CIFAR10_probs, SVHN_probs)
    svhn_in_area = trapz(svhn_in_precision, svhn_recall) * -1
    svhn_out_precision, svhn_recall = compute_AUPR(SVHN_probs, CIFAR10_probs)
    svhn_out_area = trapz(svhn_out_precision, svhn_recall) * -1
    # print results
    print(f"CIFAR100 AUPR-in: {cifar100_in_area}; CIFAR100 AUPR-out: {cifar100_out_area}")
    print(f"SVHN AUPR-in: {svhn_in_area}; SVHN AUPR-out: {svhn_out_area}")

    # compute AUTC scores
    cifar100_autc = compute_AUTC(CIFAR10_probs, CIFAR100_probs)
    svhn_autc = compute_AUTC(CIFAR10_probs, SVHN_probs)
    # print results
    print(f"CIFAR100 AUTC: {cifar100_autc}")
    print(f"SVHN AUTC: {svhn_autc}")

    # calculate unimodal evidence
    if test_unimodal:
        C10_unimodal_evidence = C10_meanprob - np.mean(CIFAR10_shifted_probs)
        C100_unimodal_evidence = C100_meanprob - np.mean(CIFAR100_shifted_probs)
        SVHN_unimodal_evidence = SVHN_meanprob - np.mean(SVHN_shifted_probs)

    if plot:
        plt.plot(cifar100_fpr, cifar100_tpr, label=f"CIFAR100:{cifar100_area:.4f}")

        plt.plot(svhn_fpr, svhn_tpr, label=f"SVHN:{svhn_area:.4f}")
        plt.title(f"AUROC curves for CIFAR100 and SVHN")
        plt.legend()
        plt.savefig(f"results/histograms/{tgt_name}_AUROC.png")
        # plt.show()
        plt.figure()

        # Plot histograms for each dataset
        # print(len(CIFAR10_probs), min(CIFAR10_probs), max(CIFAR10_probs))
        # CIFAR10_probs = aggregate_data(CIFAR10_probs)
        # print(len(CIFAR10_probs), min(CIFAR10_probs), max(CIFAR10_probs))
        # CIFAR100_probs = aggregate_data(CIFAR100_probs)
        # SVHN_probs = aggregate_data(SVHN_probs)

        CIFAR10_probs, C10_edges = create_online_hist(CIFAR10_probs, n_bins=100)
        CIFAR100_probs, C100_edges = create_online_hist(CIFAR100_probs, n_bins=100)
        SVHN_probs, SVHN_edges = create_online_hist(SVHN_probs, n_bins=100)
        plt.bar(C10_edges[:-1], CIFAR10_probs, width=np.diff(C10_edges), alpha=0.5, label='CIFAR10', align='edge')
        plt.bar(C100_edges[:-1], CIFAR100_probs, width=np.diff(C100_edges), alpha=0.5, label='CIFAR100', align='edge')
        plt.bar(SVHN_edges[:-1], SVHN_probs, width=np.diff(SVHN_edges), alpha=0.5, label='SVHN', align='edge')

        # plt.hist(CIFAR10_probs, bins=50, alpha=0.5, label='CIFAR10')
        # plt.hist(CIFAR100_probs, bins=50, alpha=0.5, label='CIFAR100')
        # plt.hist(SVHN_probs, bins=50, alpha=0.5, label='SVHN')

        # Add a legend and title to the plot
        plt.legend(loc='upper right')
        plt.title(f'{tgt_name}: Histograms of CIFAR10, CIFAR100, and SVHN image log probabilities')

        # save the plot
        plt.tight_layout()
        plt.savefig(f"results/histograms/{tgt_name}.png")

        # Show the plot
        # plt.show()

    results = {
        'CIFAR100_AUROC': cifar100_area,
        'SVHN_AUROC': svhn_area,
        'CIFAR100_precision': cifar100_precision,
        'SVHN_precision': svhn_precision,
        'CIFAR10_meanprob': C10_meanprob,
        'CIFAR100_meanprob': C100_meanprob,
        'SVHN_meanprob': SVHN_meanprob,
        'CIFAR100_AUPR-in': cifar100_in_area,
        'CIFAR100_AUPR-out': cifar100_out_area,
        'CIFAR100_AUTC': cifar100_autc,
        'SVHN_AUPR-in': svhn_in_area,
        'SVHN_AUPR-out': svhn_out_area,
        'SVHN_AUTC': svhn_autc,
    }
    if test_unimodal:
        results['unimodal_evidence'] = {
            'CIFAR10': C10_unimodal_evidence,
            'CIFAR100': C100_unimodal_evidence,
            'SVHN': SVHN_unimodal_evidence
        }
    return results


def compute_auroc_scores(dhm, ID_dataloader, OOD_dataloaders, return_plot=False):
    ID_probs = []
    OOD_probs = [[] for i in range(len(OOD_dataloaders))]
    ID_softmax_probs, OOD_softmax_probs = [], [[] for i in range(len(OOD_dataloaders))]
    dataset_labels = []

    print("computing ID probabilities...")
    for i, data in enumerate(tqdm(ID_dataloader)):
        inputs, labels = data
        inputs = inputs.to(device)

        #logpx, sh_logpx = get_logpx_loop(dhm, inputs)
        logpx, features = dhm.log_prob(inputs, return_features=True)
        y, logpz, logdet, z, _ = dhm(inputs, return_features=False)

        ID_probs.extend(logpx.detach().cpu().numpy().tolist())

        labels = labels.to(device)
        # calculate softmaxes
        softmaxes = nn.functional.softmax(y, dim=1)
        # get the max softmax for each sample and create a mask
        max_vals, max_indices = torch.max(softmaxes, dim=1, keepdim=True)
        #mask = softmaxes == max_vals
        mask = labels == max_indices.squeeze()

        # set all non-max softmaxes to nan
        #softmaxes_masked_nan = torch.where(mask, softmaxes, torch.tensor(float('nan')).to(softmaxes.device))
        # keep only the value in each sample indexed by the correct class (incorrect predictions become nan)
        #labelled_softmaxes = torch.gather(softmaxes_masked_nan, 1, labels.unsqueeze(1)).squeeze()

        labelled_softmaxes = max_vals.squeeze()
        labelled_softmaxes[~mask] = torch.tensor(float('nan'))

        ID_softmax_probs.extend(labelled_softmaxes.squeeze().detach().cpu().numpy().tolist())
        dataset_labels.extend(['ID' for _ in range(len(labels))])
    for i, dataloader in enumerate(OOD_dataloaders):
        for j, data in enumerate(tqdm(dataloader)):
            inputs, _ = data
            inputs = inputs.to(device)

            logpx, features = dhm.log_prob(inputs, return_features=True)
            y, logpz, logdet, z, _ = dhm(inputs, return_features=False)

            OOD_probs[i].extend(logpx.detach().cpu().numpy().tolist())
            softmaxes = get_max_softmaxes(y)  # softmax val of the max val for each sample
            OOD_softmax_probs[i].extend(softmaxes.squeeze().detach().cpu().numpy().tolist())
            dataset_labels.extend([f'ID_{i}' for _ in range(len(inputs))])

    print(len(ID_probs))
    ID_meanprob = np.mean(ID_probs)
    OOD_meanprobs = [np.mean(probs) for probs in OOD_probs]
    print(f"ID mean prob: {ID_meanprob:.5g}, std: {np.std(ID_probs):.5g}")
    results = {
        'ID_meanprob': ID_meanprob
    }
    for i, probs in enumerate(OOD_probs):
        print(f"OOD set {i} mean prob: {OOD_meanprobs[i]:.5g}, std: {np.std(probs):.5g}")
        results[f'OOD_{i}_meanprob'] = OOD_meanprobs[i]

    print(
        f"(mean(C10) - std(C10)) - (mean(C100) + std(C100)): {(ID_meanprob - np.std(ID_probs)) - (OOD_meanprobs[0] + np.std(OOD_probs[0]))}")

    # create histogram...
    probs_list = [item[0] for item in ID_probs] + [item[0] for sublist in OOD_probs for item in sublist]
    #probs_labels = ['ID'] * len(ID_probs) + ['OOD'] * sum(map(len, OOD_probs))
    probs_df = pd.DataFrame({'log_prob': probs_list, 'data_type': dataset_labels})
    fig = px.histogram(probs_df, x="log_prob", color='data_type', opacity=0.75, barmode='overlay')

    # create AUROC curve
    for i, probs in enumerate(OOD_probs):
        tpr, fpr = compute_AUROC(ID_probs, probs)
        area = trapz(tpr, x=fpr) * -1
        print(f"OOD set {i} AUROC: {area:.5g}")

        precision = compute_precision(ID_probs, probs)
        print(f"OOD set {i} precision: {precision:.5f}")

        results[f'OOD_{i}_AUROC'] = area
        results[f'OOD_{i}_precision'] = precision

        # compute AUPR scores
        in_precision, in_recall = compute_AUPR(ID_probs, probs)
        in_area = trapz(in_precision, x=in_recall) * -1
        out_precision, out_recall = compute_AUPR(probs, ID_probs)
        out_area = trapz(out_precision, x=out_recall) * -1
        # print results
        print(f"OOD set {i} AUPR-in: {in_area}; OOD set {i} AUPR-out: {out_area}")
        results[f'OOD_{i}_AUPR-in'] = in_area
        results[f'OOD_{i}_AUPR-out'] = out_area

        # compute AUTC scores
        autc = compute_AUTC(ID_probs, probs)
        # print results
        print(f"OOD_{i} AUTC: {autc}")
        results[f'OOD_{i}_AUTC'] = autc

    #
    # baseline results
    #
    ID_softmax_probs = np.asarray(ID_softmax_probs)
    ID_softmax_probs = ID_softmax_probs[~np.isnan(ID_softmax_probs)]  # remove nan entries
    baseline_results = {}
    for i, softprobs in enumerate(OOD_softmax_probs):
        softprobs = np.asarray(softprobs)
        print("trying softmax results...")
        print(min(ID_softmax_probs), max(ID_softmax_probs))
        print(min(softprobs), max(softprobs))
        tpr, fpr = compute_AUROC(ID_softmax_probs, softprobs)
        area = trapz(tpr, x=fpr) * -1
        baseline_results[f'OOD_{i}_AUROC'] = area
        in_precision, in_recall = compute_AUPR(ID_softmax_probs, softprobs)
        in_area = trapz(in_precision, x=in_recall) * -1
        out_precision, out_recall = compute_AUPR(softprobs, ID_softmax_probs)
        out_area = trapz(out_precision, x=out_recall) * -1
        baseline_results[f'OOD_{i}_AUPR-in'] = in_area
        baseline_results[f'OOD_{i}_AUPR-out'] = out_area
        autc = compute_AUTC(ID_softmax_probs, softprobs)
        baseline_results[f'OOD_{i}_AUTC'] = autc
    results['baseline_results'] = baseline_results
    return results, fig if return_plot else None


def generate_tsne_plots(dhm, batch_size=128, max_samples=1000):
    feature_set = []
    feature_labels = []
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.25, 0.25, 0.25)),
         ])
    CIFAR10_dataset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                                   download=True, transform=transform)
    CIFAR10_loader = torch.utils.data.DataLoader(CIFAR10_dataset, batch_size=batch_size,
                                                 shuffle=True, num_workers=4)
    for i, data in enumerate(tqdm(CIFAR10_loader)):
        inputs, labels = data
        inputs = inputs.to(device)

        logpx, features = dhm.log_prob(inputs, return_features=True)

        feature_set.extend(features.detach().cpu().numpy())
        feature_labels.extend(labels.numpy())
        if i * batch_size > max_samples:
            break  # limit the number of samples...
    if np.prod(feature_set[0].shape) > 1000:
        print(f"features (with {feature_set[0].shape} elements) are too large! Skipping tsne generation...")
        return None
    features = np.stack(feature_set, axis=0)
    features = features[:max_samples]

    labels = np.stack(feature_labels, axis=0)
    labels = labels[:max_samples]

    # create tsne plot
    tsne = TSNE(n_components=2, random_state=0)
    tsne_data = tsne.fit_transform(features)

    plt_data = pd.DataFrame({'x': tsne_data[:, 0], 'y': tsne_data[:, 1], 'labels': labels})
    plt_data['labels'] = plt_data['labels'].astype(str)
    fig = px.scatter(plt_data, x='x', y='y', color='labels')

    return fig


def metropolis_walk(dhm, feature_list, n_samples=1000, max_attempts=1):
    if isinstance(feature_list, list):
        feature_list = np.stack(feature_list, axis=0)
    max_val = np.max(feature_list)
    mean_val, std_val = np.mean(feature_list, axis=0), np.std(feature_list, axis=0) / 3
    #x_0 = np.random.uniform(low=-max_val, high=max_val, size=feature_list.shape[1])
    #x_last = np.random.normal(mean_val, std_val)
    x_last = feature_list[0]

    samples = [x_last]
    last_prob = float(dhm.feature_logprob(torch.tensor([x_last], dtype=torch.float32).to(device)).detach().cpu().squeeze().numpy())
    print(last_prob, type(last_prob))
    probs = [last_prob]
    for i in tqdm(range(n_samples)):
        success = False
        counter = 0
        while not success and counter < max_attempts:
            counter += 1
            #x_i = np.random.uniform(low=-max_val, high=max_val, size=feature_list.shape[1])
            x_i = np.random.normal(x_last, std_val)
            new_prob = float(dhm.feature_logprob(torch.tensor([x_i], dtype=torch.float32).to(device)).detach().cpu().squeeze().numpy())
            A = min(1, np.exp(new_prob - last_prob))
            if np.random.uniform() < A:  # accept the sample
                success = True
                samples.append(x_i)
                probs.append(new_prob)
                x_last = x_i
                last_prob = new_prob
    print(f"{len(samples)} samples found by Metropolis")
    return samples, probs


def generate_multidata_tsne_plots(dhm, batch_size=128, max_samples=1000, return_table=True, plot_classes=False,
                                  max_size=20, use_metropolis=False):
    feature_set = []
    dataset_labels = []
    class_labels = []
    logprobs = []
    softmax_probs = []
    # images = []
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize(np.array([125.3, 123.0, 113.9]) / 255.0,
                              np.array([63.0, 62.1, 66.7]) / 255.0),
         ])
    CIFAR10_dataset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                                   download=True, transform=transform)
    CIFAR100_dataset = torchvision.datasets.CIFAR100(root='./data', train=False,
                                                     download=True, transform=transform)
    SVHN_dataset = torchvision.datasets.SVHN(root='./data', split='test',
                                             download=True, transform=transform)

    CIFAR10_loader = torch.utils.data.DataLoader(CIFAR10_dataset, batch_size=batch_size,
                                                 shuffle=True, num_workers=4)
    CIFAR100_loader = torch.utils.data.DataLoader(CIFAR100_dataset, batch_size=batch_size,
                                                  shuffle=True, num_workers=4)
    SVHN_loader = torch.utils.data.DataLoader(SVHN_dataset, batch_size=batch_size,
                                              shuffle=True, num_workers=4)

    for i, data in enumerate(tqdm(CIFAR10_loader)):
        inputs, labels = data
        # images.extend(inputs.numpy())
        inputs = inputs.to(device)

        logpx, features = dhm.log_prob(inputs, return_features=True)
        y, logpz, logdet, z, _ = dhm(inputs, return_features=False)

        feature_set.extend(features.detach().cpu().numpy())
        dataset_labels.extend(['cifar10' for k in range(len(features))])
        class_labels.extend([c10_id2label[k] for k in list(labels.numpy())])
        logprobs.extend(logpx.squeeze().detach().cpu().numpy().tolist())

        labels = labels.to(device)
        # calculate softmaxes
        softmaxes = nn.functional.softmax(y, dim=1)
        # get the max softmax for each sample and create a mask
        max_vals, max_indices = torch.max(softmaxes, dim=1, keepdim=True)
        mask = softmaxes == max_vals
        # set all non-max softmaxes to nan
        softmaxes_masked_nan = torch.where(mask, softmaxes, torch.tensor(float('nan')).to(softmaxes.device))
        # keep only the value in each sample indexed by the correct class (incorrect predictions become nan)
        labelled_softmaxes = torch.gather(softmaxes_masked_nan, 1, labels.unsqueeze(1)).squeeze()
        softmax_probs.extend(labelled_softmaxes.squeeze().detach().cpu().numpy().tolist())

        if i * batch_size > max_samples:
            break  # limit the number of samples...
    for i, data in enumerate(tqdm(CIFAR100_loader)):
        inputs, labels = data
        # images.extend(inputs.numpy())
        inputs = inputs.to(device)

        logpx, features = dhm.log_prob(inputs, return_features=True)
        y, logpz, logdet, z, _ = dhm(inputs, return_features=False)

        feature_set.extend(features.detach().cpu().numpy())
        dataset_labels.extend(['cifar100' for k in range(len(features))])
        class_labels.extend([c100_id2label[k] for k in list(labels.numpy())])
        logprobs.extend(logpx.squeeze().detach().cpu().numpy().tolist())

        softmaxes = get_max_softmaxes(y)  # softmax val of the max val for each sample
        softmax_probs.extend(softmaxes.squeeze().detach().cpu().numpy().tolist())
        if i * batch_size > max_samples:
            break  # limit the number of samples...
    for i, data in enumerate(tqdm(SVHN_loader)):
        inputs, labels = data
        # images.extend(inputs.numpy())
        inputs = inputs.to(device)

        logpx, features = dhm.log_prob(inputs, return_features=True)
        y, logpz, logdet, z, _ = dhm(inputs, return_features=False)

        feature_set.extend(features.detach().cpu().numpy())
        dataset_labels.extend(['svhn' for k in range(len(features))])
        class_labels.extend([str(k) for k in list(labels.numpy())])
        logprobs.extend(logpx.squeeze().detach().cpu().numpy().tolist())

        softmaxes = get_max_softmaxes(y)  # softmax val of the max val for each sample
        softmax_probs.extend(softmaxes.squeeze().detach().cpu().numpy().tolist())
        if i * batch_size > max_samples:
            break  # limit the number of samples...
    if use_metropolis:
        metropolis_samples, metropolis_probs = metropolis_walk(dhm, feature_set, n_samples=1000)
        feature_set.extend(metropolis_samples)
        dataset_labels.extend(['metro_walk' for k in range(len(metropolis_samples))])
        class_labels.extend(['sample' for k in range(len(metropolis_samples))])
        logprobs.extend(metropolis_probs)
        softmax_probs.extend([0 for k in range(len(metropolis_samples))])
    if np.prod(feature_set[0].shape) > 3072:
        print(f"features (with {feature_set[0].shape} elements) are too large! Skipping tsne generation...")
        return None
    features = np.stack(feature_set, axis=0)
    # images = [image.transpose() for image in images]
    # features = features[:max_samples]

    # create tsne plot
    # tsne = TSNE(n_components=2, random_state=0)
    # tsne_data = tsne.fit_transform(features)
    #embedding = TSNE(n_components=2, random_state=0)
    #embedding = MDS(n_components=2, normalized_stress='auto', n_jobs=-1)
    embedding = MDS(n_components=2)
    tsne_data = embedding.fit_transform(features)

    display_sizes = [x for x in logprobs]
    minp, maxp = min(display_sizes), max(display_sizes)
    # print(type(minp))
    # print(type(display_sizes[0]))
    marker_sizes = [np.power((x - minp) / (maxp - minp), 2) * max_size for x in display_sizes]
    plt_data = pd.DataFrame({'x': tsne_data[:, 0], 'y': tsne_data[:, 1], 'dataset': dataset_labels,
                             'class': class_labels, 'logprob': logprobs,
                             'index': [i for i in range(len(dataset_labels))], 'softmax_probs': softmax_probs})

    #
    # get ID class probs
    #
    class_means = plt_data[plt_data['dataset'] == 'cifar10'].groupby('class')['logprob'].mean()
    print("Class mean logprobs:", class_means)
    print(f"min: {min(class_means)}, max: {max(class_means)}, diff: {abs(max(class_means) - min(class_means))}")
    print(f"Original logprob: {np.mean(class_means)}; Adjusted logprob: {np.mean(class_means) - abs(max(class_means) - min(class_means))}")


    #
    # fit gaussian model
    #
    gm = sklearn.mixture.GaussianMixture(n_components=10, random_state=0)
    X = plt_data[plt_data['dataset'] == 'cifar10']
    X = X[['x', 'y']]
    gm.fit(X)
    # compute probs
    c10_logprobs = gm.score_samples(X)
    C10_meanprob = np.mean(c10_logprobs)
    X_c100 = plt_data[plt_data['dataset'] == 'cifar100'][['x', 'y']]
    c100_logprobs = gm.score_samples(X_c100)
    C100_meanprob = np.mean(c100_logprobs)
    X_svhn = plt_data[plt_data['dataset'] == 'svhn'][['x', 'y']]
    svhn_logprobs = gm.score_samples(X_svhn)
    SVHN_meanprob = np.mean(svhn_logprobs)

    print(f"t-SNE CIFAR10 mean prob: {C10_meanprob:.5g}")
    print(f"t-SNE CIFAR100 mean prob: {C100_meanprob:.5g}")
    print(f"t-SNE SVHN mean prob: {SVHN_meanprob:.5g}")
    # compute AUROC scores
    cifar100_tpr, cifar100_fpr = compute_AUROC(c10_logprobs, c100_logprobs)
    cifar100_area = trapz(cifar100_tpr, x=cifar100_fpr) * -1
    print(f"t-SNE CIFAR100 AUROC: {cifar100_area}")
    svhn_tpr, svhn_fpr = compute_AUROC(c10_logprobs, svhn_logprobs)
    svhn_area = trapz(svhn_tpr, x=svhn_fpr) * -1
    print(f"t-SNE SVHN AUROC: {svhn_area}")

    # plt_data['labels'] = plt_data['labels'].astype(str)
    # fig = px.scatter(plt_data, x='x', y='y', color='dataset')
    if plot_classes:
        colour_scales = {
            'cifar10': px.colors.sequential.Peach,
            'cifar100': px.colors.sequential.Mint,
            'svhn': px.colors.sequential.Purp,
        }
        c10_colours = px.colors.sample_colorscale(colour_scales['cifar10'], np.linspace(0, 1,
                                                                                        len(c10_id2label)))  # sample a colour for each c10 class
        c100_colours = px.colors.sample_colorscale(colour_scales['cifar100'], np.linspace(0, 1, len(c100_id2label)))
        svhn_colours = px.colors.sample_colorscale(colour_scales['svhn'], np.linspace(0, 1, 10))
        c10dict = {class_name: c10_colours[i] for i, class_name in enumerate(c10_id2label.values())}
        c100dict = {class_name: c100_colours[i] for i, class_name in enumerate(c100_id2label.values())}
        svhndict = {str(i): svhn_colours[i] for i in range(10)}
        c10dict.update(c100dict)
        c10dict.update(svhndict)
        colourdict = c10dict

        fig = px.scatter(plt_data, x='x', y='y', color='class', color_discrete_map=colourdict, size=marker_sizes)
    else:
        fig = px.scatter(plt_data, x='x', y='y', color='dataset', size=marker_sizes,
                         hover_data=['class', 'dataset', 'logprob', 'index'])

    # save images...?
    """for i, image in enumerate(images):
        if i in (720, 1272):
            im = image * 0.25 + 0.5
            im = np.rot90(im, k=-1)
            im = (im * 255).astype(np.uint8)
            im = Image.fromarray(im)
            im.save(f"results/demo_ims/id{i}_prob{int(logprobs[i])}.png")"""

    # generate histogram...
    histogram = px.histogram(plt_data, x='logprob', color='dataset', opacity=0.75, barmode='overlay')

    #
    # compute Hendrycks baseline result
    #
    C10_softprobs = np.asarray(plt_data[plt_data['dataset'] == 'cifar10']['softmax_probs'])
    C10_softprobs = C10_softprobs[~np.isnan(C10_softprobs)]  # remove nan entries
    C100_softprobs = np.asarray(plt_data[plt_data['dataset'] == 'cifar100']['softmax_probs'])
    SVHN_softprobs = np.asarray(plt_data[plt_data['dataset'] == 'svhn']['softmax_probs'])
    h_C10_mean, h_C10_stdev = np.nanmean(C10_softprobs), np.nanstd(C10_softprobs)
    h_C100_mean, h_C100_stdev = np.mean(C100_softprobs), np.std(C100_softprobs)
    h_SVHN_mean, h_SVHN_stdev = np.mean(SVHN_softprobs), np.std(SVHN_softprobs)
    print(f"Baseline CIFAR10 mean prob: {h_C10_mean:.5g}")
    print(f"Baseline CIFAR100 mean prob: {h_C100_mean:.5g}")
    print(f"Baseline SVHN mean prob: {h_SVHN_mean:.5g}")
    # compute AUROC scores
    h_cifar100_tpr, h_cifar100_fpr = compute_AUROC(C10_softprobs, C100_softprobs)
    h_cifar100_area = trapz(h_cifar100_tpr, x=h_cifar100_fpr) * -1
    print(f"Baseline CIFAR100 AUROC: {h_cifar100_area}")
    h_svhn_tpr, h_svhn_fpr = compute_AUROC(C10_softprobs, SVHN_softprobs)
    h_svhn_area = trapz(h_svhn_tpr, x=h_svhn_fpr) * -1
    print(f"Baseline SVHN AUROC: {h_svhn_area}")

    results = {
        'CIFAR100_AUROC': cifar100_area,
        'SVHN_AUROC': svhn_area,
        'CIFAR10_meanprob': C10_meanprob,
        'CIFAR100_meanprob': C100_meanprob,
        'SVHN_meanprob': SVHN_meanprob,
        'Baseline_results': {
            'CIFAR10_meanprob': h_C10_mean,
            'CIFAR100_meanprob': h_C100_mean,
            'SVHN_meanprob': h_SVHN_mean,
            'C100_AUROC': h_cifar100_area,
            'SVHN_AUROC': h_svhn_area
        }
    }
    if return_table:
        return fig, results, plt_data, histogram
    return fig, results, None, histogram


def compute_embeddings(dhm, ID_dataloader, OOD_dataloaders, batch_size=128, max_samples=1000, return_table=True,
                       return_gmm_results=False, max_size=20, embedding_type='mds'):
    feature_set = []
    dataset_labels = []
    class_labels = []
    logprobs = []
    softmax_probs = []
    #ID_softmax_probs, OOD_softmax_probs = [], [[] for i in range(len(OOD_dataloaders))]

    for i, data in enumerate(tqdm(ID_dataloader)):
        inputs, labels = data
        inputs = inputs.to(device)

        logpx, features = dhm.log_prob(inputs, return_features=True)
        y, logpz, logdet, z, _ = dhm(inputs, return_features=False)

        feature_set.extend(features.detach().cpu().numpy())
        dataset_labels.extend(['ID' for k in range(len(features))])
        class_labels.extend([k for k in list(labels.numpy())])
        logprobs.extend(logpx.squeeze().detach().cpu().numpy().tolist())

        labels = labels.to(device)
        softmaxes = nn.functional.softmax(y, dim=1)
        max_vals, max_indices = torch.max(softmaxes, dim=1, keepdim=True)
        mask = labels == max_indices.squeeze()
        labelled_softmaxes = max_vals.squeeze()
        labelled_softmaxes[~mask] = torch.tensor(float('nan'))
        softmax_probs.extend(labelled_softmaxes.squeeze().detach().cpu().numpy().tolist())
        if i * batch_size > max_samples:
            break  # limit the number of samples...
    num_ID_classes = len(set(class_labels))
    num_ID_samples = len(feature_set)
    for i, dataloader in enumerate(OOD_dataloaders):
        dataset_name = f"OOD_{i}"
        for j, data in enumerate(tqdm(dataloader)):
            inputs, labels = data
            inputs = inputs.to(device)

            logpx, features = dhm.log_prob(inputs, return_features=True)
            y, logpz, logdet, z, _ = dhm(inputs, return_features=False)

            feature_set.extend(features.detach().cpu().numpy())
            dataset_labels.extend([dataset_name for k in range(len(features))])
            class_labels.extend([k for k in list(labels.numpy())])
            logprobs.extend(logpx.squeeze().detach().cpu().numpy().tolist())
            softmaxes = get_max_softmaxes(y)  # softmax val of the max val for each sample
            softmax_probs.extend(softmaxes.squeeze().detach().cpu().numpy().tolist())
            if j * batch_size > max_samples:
                break  # limit the number of samples...

    if np.prod(feature_set[0].shape) > 3072:
        print(f"features (with {feature_set[0].shape} elements) are too large! Skipping embedding generation...")
        return None
    features = np.stack(feature_set, axis=0)

    # create embedding plot
    if embedding_type == 'tsne':
        embedding = TSNE(n_components=2, random_state=0)
    else:
        embedding = MDS(n_components=2)#, normalized_stress='auto', n_jobs=-1)
    embedding_data = embedding.fit_transform(features)

    display_sizes = [x for x in logprobs]
    minp, maxp = min(display_sizes), max(display_sizes)

    marker_sizes = [np.power((x - minp) / (maxp - minp), 2) * max_size for x in display_sizes]
    plt_data = pd.DataFrame({'x': embedding_data[:, 0], 'y': embedding_data[:, 1], 'dataset': dataset_labels,
                             'class': class_labels, 'logprob': logprobs,
                             'index': [i for i in range(len(dataset_labels))], 'softmax_prob': softmax_probs})

    fig = px.scatter(plt_data, x='x', y='y', color='dataset', size=marker_sizes,
                     hover_data=['class', 'dataset', 'logprob', 'index', 'softmax_prob'])

    if return_gmm_results:
        gmm_results = fit_gmm_for_auroc(features[:num_ID_samples], features[num_ID_samples:],
                                        n_components=num_ID_classes)
    return fig, plt_data if return_table else None, gmm_results if return_gmm_results else None


def fit_gmm_for_auroc(ID_data, OOD_data, n_components=10):
    # fit gmm model
    gm = sklearn.mixture.GaussianMixture(n_components=n_components, random_state=0)
    gm.fit(ID_data)

    # compute probs
    ID_logprobs = gm.score_samples(ID_data)
    ID_meanprob = np.mean(ID_logprobs)
    OOD_logprobs = gm.score_samples(OOD_data)
    OOD_meanprob = np.mean(OOD_logprobs)
    print(f"GMM ID mean prob: {ID_meanprob:.5g}")
    print(f"GMM OOD mean prob: {OOD_meanprob:.5g}")

    # compute AUROC score
    tpr, fpr = compute_AUROC(ID_logprobs, OOD_logprobs)
    area = trapz(tpr, x=fpr) * -1
    print(f"GMM AUROC score: {area:.5g}")

    return {"ID_meanprob": ID_meanprob, "OOD_meanprob": OOD_meanprob, "AUROC": area}


def fit_transformed_data(embeddings, dataset_labels, method_label='unknown method'):
    plt_data = pd.DataFrame({'x': embeddings[:, 0], 'y': embeddings[:, 1], 'dataset': dataset_labels})

    #
    # fit gaussian model
    #
    gm = sklearn.mixture.GaussianMixture(n_components=10, random_state=0)
    X = plt_data[plt_data['dataset'] == 'cifar10']
    X = X[['x', 'y']]
    gm.fit(X)
    # compute probs
    c10_logprobs = gm.score_samples(X)
    C10_meanprob = np.mean(c10_logprobs)
    X_c100 = plt_data[plt_data['dataset'] == 'cifar100'][['x', 'y']]
    c100_logprobs = gm.score_samples(X_c100)
    C100_meanprob = np.mean(c100_logprobs)
    X_svhn = plt_data[plt_data['dataset'] == 'svhn'][['x', 'y']]
    svhn_logprobs = gm.score_samples(X_svhn)
    SVHN_meanprob = np.mean(svhn_logprobs)

    print(f"{method_label} CIFAR10 mean prob: {C10_meanprob:.5g}")
    print(f"{method_label} CIFAR100 mean prob: {C100_meanprob:.5g}")
    print(f"{method_label} SVHN mean prob: {SVHN_meanprob:.5g}")
    # compute AUROC scores
    cifar100_tpr, cifar100_fpr = compute_AUROC(c10_logprobs, c100_logprobs)
    cifar100_area = trapz(cifar100_tpr, x=cifar100_fpr) * -1
    print(f"{method_label} CIFAR100 AUROC: {cifar100_area}")
    svhn_tpr, svhn_fpr = compute_AUROC(c10_logprobs, svhn_logprobs)
    svhn_area = trapz(svhn_tpr, x=svhn_fpr) * -1
    print(f"{method_label} SVHN AUROC: {svhn_area}")


def test_embedding_methods(dhm, batch_size=128, max_samples=1000):
    feature_set = []
    dataset_labels = []
    class_labels = []
    logprobs = []
    # images = []
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.25, 0.25, 0.25)),
         ])
    CIFAR10_dataset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                                   download=True, transform=transform)
    CIFAR100_dataset = torchvision.datasets.CIFAR100(root='./data', train=False,
                                                     download=True, transform=transform)
    SVHN_dataset = torchvision.datasets.SVHN(root='./data', split='test',
                                             download=True, transform=transform)

    CIFAR10_loader = torch.utils.data.DataLoader(CIFAR10_dataset, batch_size=batch_size,
                                                 shuffle=True, num_workers=4)
    CIFAR100_loader = torch.utils.data.DataLoader(CIFAR100_dataset, batch_size=batch_size,
                                                  shuffle=True, num_workers=4)
    SVHN_loader = torch.utils.data.DataLoader(SVHN_dataset, batch_size=batch_size,
                                              shuffle=True, num_workers=4)
    for i, data in enumerate(tqdm(CIFAR10_loader)):
        inputs, labels = data
        # images.extend(inputs.numpy())
        inputs = inputs.to(device)

        logpx, features = dhm.log_prob(inputs, return_features=True)

        feature_set.extend(features.detach().cpu().numpy())
        dataset_labels.extend(['cifar10' for i in range(len(features))])
        class_labels.extend([c10_id2label[i] for i in list(labels.numpy())])
        logprobs.extend(logpx.detach().cpu().numpy().tolist())
        if i * batch_size > max_samples:
            break  # limit the number of samples...
    for i, data in enumerate(tqdm(CIFAR100_loader)):
        inputs, labels = data
        # images.extend(inputs.numpy())
        inputs = inputs.to(device)

        logpx, features = dhm.log_prob(inputs, return_features=True)

        feature_set.extend(features.detach().cpu().numpy())
        dataset_labels.extend(['cifar100' for i in range(len(features))])
        class_labels.extend([c100_id2label[i] for i in list(labels.numpy())])
        logprobs.extend(logpx.detach().cpu().numpy().tolist())
        if i * batch_size > max_samples:
            break  # limit the number of samples...
    for i, data in enumerate(tqdm(SVHN_loader)):
        inputs, labels = data
        # images.extend(inputs.numpy())
        inputs = inputs.to(device)

        logpx, features = dhm.log_prob(inputs, return_features=True)

        feature_set.extend(features.detach().cpu().numpy())
        dataset_labels.extend(['svhn' for i in range(len(features))])
        class_labels.extend([str(i) for i in list(labels.numpy())])
        logprobs.extend(logpx.detach().cpu().numpy().tolist())
        if i * batch_size > max_samples:
            break  # limit the number of samples...
    if np.prod(feature_set[0].shape) > 1000:
        print(f"features (with {feature_set[0].shape} elements) are too large! Skipping tsne generation...")
        return None

    features = np.stack(feature_set, axis=0)

    # test isomap embeddings
    iso = Isomap(n_components=2)
    iso_data = iso.fit_transform(features)
    fit_transformed_data(iso_data, dataset_labels, method_label='Isomap')

    # test locally linear embedding
    lle = LocallyLinearEmbedding(n_components=2)
    lle_data = lle.fit_transform(features)
    fit_transformed_data(lle_data, dataset_labels, method_label='LLE')

    # test modified LLE
    lle = LocallyLinearEmbedding(n_components=2, method='modified')
    lle_data = lle.fit_transform(features)
    fit_transformed_data(lle_data, dataset_labels, method_label='modified LLE')

    # test hessian eigenmapping
    # lle = LocallyLinearEmbedding(n_components=2, method='hessian', n_neighbors=6)
    # lle_data = lle.fit_transform(features)
    # fit_transformed_data(lle_data, dataset_labels, method_label='Hessian LLE')

    # test ltsa
    # lle = LocallyLinearEmbedding(n_components=2, method='ltsa')
    # lle_data = lle.fit_transform(features)
    # fit_transformed_data(lle_data, dataset_labels, method_label='LTSA')

    # test tsne embeddings
    tsne = TSNE(n_components=2, random_state=0)
    tsne_data = tsne.fit_transform(features)
    fit_transformed_data(tsne_data, dataset_labels, method_label='t-SNE')


def test_custom_datasets(dhm, dataloaders, dataset_names=None, batch_size=128, max_samples=1000, return_table=True,
                         max_size=20, test_gmm=False):
    feature_set = []
    dataset_labels = []
    class_labels = []
    logprobs = []
    # images = []

    # train a gmm on the ID features
    id_features = []
    if test_gmm:
        feature_gm = sklearn.mixture.GaussianMixture(n_components=2, random_state=0, warm_start=True)
        for i, data in enumerate(tqdm(dataloaders[0])):
            inputs, _ = data
            inputs = inputs.to(device)
            _, features = dhm.log_prob(inputs, return_features=True)
            # features = features.detach().cpu().numpy()
            # feature_gm.fit(features)
            id_features.extend(features.detach().cpu().numpy())
        id_features = np.stack(id_features, axis=0)
        feature_gm.fit(id_features)

    for i, dataloader in enumerate(dataloaders):
        for j, data in enumerate(tqdm(dataloader)):
            inputs, labels = data
            # images.extend(inputs.numpy())
            inputs = inputs.to(device)

            logpx, features = dhm.log_prob(inputs, return_features=True)

            logprobs.extend(logpx.detach().cpu().numpy().tolist())

            feature_set.extend(features.detach().cpu().numpy())
            if dataset_names:
                dataset_labels.extend([dataset_names[i] for k in range(len(features))])
            class_labels.extend([str(label) for label in list(labels.numpy())])
            # if i * batch_size > max_samples:
            #    break  # limit the number of samples...

    if np.prod(feature_set[0].shape) > 3072:
        print(f"features (with {feature_set[0].shape} elements) are too large! Skipping tsne generation...")
        return None
    features = np.stack(feature_set, axis=0)
    if test_gmm:
        gmm_logprobs = feature_gm.score_samples(features)
    else:
        gmm_logprobs = [None for i in range(len(features))]

    # create tsne plot
    use_tsne = False
    if np.prod(feature_set[0].shape) > 2:
        use_tsne = True
        tsne = TSNE(n_components=2, random_state=0)
        tsne_data = tsne.fit_transform(features)
    else:
        tsne_data = features

    display_sizes = [x for x in logprobs]
    if test_gmm:
        display_sizes = [x for x in gmm_logprobs]
    minp, maxp = min(display_sizes), max(display_sizes)
    marker_sizes = [np.power((x - minp) / (maxp - minp), 2) * max_size for x in display_sizes]
    plt_data = pd.DataFrame({'x': tsne_data[:, 0], 'y': tsne_data[:, 1], 'dataset': dataset_labels,
                             'class': class_labels, 'logprob': logprobs, 'gmm_logprob': gmm_logprobs,
                             'index': [i for i in range(len(dataset_labels))]})

    #
    # calculate tsne OOD performance
    #
    id_name = dataset_names[0]

    # fit gaussian model
    gm = sklearn.mixture.GaussianMixture(n_components=10, random_state=0)
    X = plt_data[plt_data['dataset'] == id_name]
    X = X[['x', 'y']]
    gm.fit(X)
    gm_logprobs = gm.score_samples(X)
    gm_meanprob = np.mean(gm_logprobs)

    #
    # calculate meanprob and AUROC stats
    #
    # dataset_names = plt_data['dataset'].unique()
    id_probs = plt_data[plt_data['dataset'] == id_name]['logprob'].to_list()
    id_mean = np.mean(id_probs)
    id_std = np.std(id_probs)
    results = {dataset_name: {} for dataset_name in dataset_names}
    results[id_name]['meanprob'] = id_mean
    results[id_name]['stdprob'] = id_std
    results[id_name]['tsne_meanprob'] = gm_meanprob
    for i, dataset_name in enumerate(dataset_names[1:]):
        target_data = plt_data[plt_data['dataset'] == dataset_name]
        logprobs = target_data['logprob'].tolist()
        meanprob = np.mean(logprobs)
        stdprob = np.std(logprobs)

        # calculate AUROC scores
        tpr, fpr = compute_AUROC(id_probs, logprobs)
        area = trapz(tpr, x=fpr) * -1
        precision = compute_precision(id_probs, logprobs)

        # compute tsne stats

        # compute probs
        data_X = target_data[['x', 'y']]
        tsne_logprobs = gm.score_samples(data_X)
        tsne_meanprob = np.mean(tsne_logprobs)
        # compute AUROC score
        tsne_tpr, tsne_fpr = compute_AUROC(gm_logprobs, tsne_logprobs)
        tsne_area = trapz(tsne_tpr, x=tsne_fpr) * -1

        results[dataset_name]['meanprob'] = meanprob
        results[dataset_name]['stdprob'] = stdprob
        results[dataset_name]['AUROC'] = area
        results[dataset_name]['precision'] = precision
        results[dataset_name]['tsne_meanprob'] = tsne_meanprob
        results[dataset_name]['tsne_AUROC'] = tsne_area

    if test_gmm:
        #
        # calculate meanprob and AUROC stats
        #
        # dataset_names = plt_data['dataset'].unique()
        id_probs = plt_data[plt_data['dataset'] == id_name]['gmm_logprob'].to_list()
        id_mean = np.mean(id_probs)
        id_std = np.std(id_probs)
        # results = {dataset_name: {} for dataset_name in dataset_names}
        results[id_name]['gmm_meanprob'] = id_mean
        results[id_name]['gmm_stdprob'] = id_std
        for i, dataset_name in enumerate(dataset_names[1:]):
            target_data = plt_data[plt_data['dataset'] == dataset_name]
            logprobs = target_data['gmm_logprob'].tolist()
            meanprob = np.mean(logprobs)
            stdprob = np.std(logprobs)

            # calculate AUROC scores
            tpr, fpr = compute_AUROC(id_probs, logprobs)
            area = trapz(tpr, x=fpr) * -1
            precision = compute_precision(id_probs, logprobs)
            results[dataset_name]['gmm_meanprob'] = meanprob
            results[dataset_name]['gmm_stdprob'] = stdprob
            results[dataset_name]['gmm_AUROC'] = area
            results[dataset_name]['gmm_precision'] = precision

    plot_title = "feature tsne plot" if use_tsne else "feature plot"
    if test_gmm:
        plot_title = "direct GMM " + plot_title
    fig = px.scatter(plt_data, x='x', y='y', color='dataset', size=marker_sizes,
                     hover_data=['class', 'dataset', 'logprob', 'gmm_logprob', 'index'], title=plot_title)
    if return_table:
        return fig, results, plt_data
    return fig, results, None


def generate_histograms_remove_top_features(dhm, batch_size=None, tgt_name=None):
    get_features = True
    k = 127  # remove the top k samples from the batch
    if batch_size is None:
        batch_size = args.batch
    if tgt_name is None:
        tgt_name = args.name.split('.')[0]
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.25, 0.25, 0.25)),
         # transforms.RandomHorizontalFlip(0.5),
         # transforms.RandomCrop(size=32, padding=4, padding_mode='reflect')
         ])

    CIFAR10_probs = []
    CIFAR100_probs = []
    SVHN_probs = []

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

    print("computing CIFAR10 probabilities...")
    for i, data in enumerate(tqdm(CIFAR10_loader)):
        inputs, _ = data
        inputs = inputs.to(device)

        # forward + backward + optimize
        if get_features:
            y, logpz, logdet, z, features = dhm(inputs, return_features=True)
        else:
            y, logpz, logdet, z = dhm(inputs)

        # get max value over each sample to identify maxval sample and remove it
        fmax = torch.amax(features, dim=(1, 2, 3), keepdim=True)  # per-sample max
        # max_sample_idx = torch.argmax(fmax)
        # Get the indices of the top k maximum values
        clamped_k = min(k, len(fmax) - 2)
        topk_values, topk_indices = torch.topk(fmax, clamped_k, dim=0)

        new_batch = torch.index_select(inputs, dim=0,
                                       index=torch.tensor([i for i in range(len(inputs)) if i not in topk_indices]).to(
                                           device))
        # print(len(new_batch))
        y, logpz, logdet, z = dhm(new_batch)

        # print(logpz.shape, logdet.shape)
        logpx = compute_logpx(logpz, logdet)

        CIFAR10_probs.extend(logpx.detach().cpu().numpy().tolist())
        # if i > 100:
        #    break
    print("computing CIFAR100 probabilities...")
    for i, data in enumerate(tqdm(CIFAR100_loader)):
        inputs, _ = data
        inputs = inputs.to(device)

        # forward + backward + optimize
        if get_features:
            y, logpz, logdet, z, features = dhm(inputs, return_features=True)
        else:
            y, logpz, logdet, z = dhm(inputs)

        # get max value over each sample to identify maxval sample and remove it
        fmax = torch.amax(features, dim=(1, 2, 3), keepdim=True)  # per-sample max
        # max_sample_idx = torch.argmax(fmax)
        # Get the indices of the top k maximum values
        clamped_k = min(k, len(fmax) - 2)
        topk_values, topk_indices = torch.topk(fmax, clamped_k, dim=0)

        new_batch = torch.index_select(inputs, dim=0,
                                       index=torch.tensor([i for i in range(len(inputs)) if i not in topk_indices]).to(
                                           device))
        y, logpz, logdet, z = dhm(new_batch)

        logpx = compute_logpx(logpz, logdet)
        # _, logpx = dhm(z, logpz, inverse=True)
        # logpx = compute_uncertainty(y, logpz, logdet)
        CIFAR100_probs.extend(logpx.detach().cpu().numpy().tolist())
        # if i > 100:
        #    break
    print("computing SVHN probabilities...")
    for i, data in enumerate(tqdm(SVHN_loader)):
        inputs, _ = data
        inputs = inputs.to(device)

        # forward + backward + optimize
        if get_features:
            y, logpz, logdet, z, features = dhm(inputs, return_features=True)
        else:
            y, logpz, logdet, z = dhm(inputs)

        # get max value over each sample to identify maxval sample and remove it
        fmax = torch.amax(features, dim=(1, 2, 3), keepdim=True)  # per-sample max
        # max_sample_idx = torch.argmax(fmax)
        # Get the indices of the top k maximum values
        clamped_k = min(k, len(fmax) - 2)
        topk_values, topk_indices = torch.topk(fmax, clamped_k, dim=0)

        new_batch = torch.index_select(inputs, dim=0,
                                       index=torch.tensor([i for i in range(len(inputs)) if i not in topk_indices]).to(
                                           device))
        y, logpz, logdet, z = dhm(new_batch)

        logpx = compute_logpx(logpz, logdet)
        # _, logpx = dhm(z, logpz, inverse=True)
        # logpx = compute_uncertainty(y, logpz, logdet)
        SVHN_probs.extend(logpx.detach().cpu().numpy().tolist())
        # if i > 100:
        #    break
    C10_meanprob = np.mean(CIFAR10_probs)
    C100_meanprob = np.mean(CIFAR100_probs)
    SVHN_meanprob = np.mean(SVHN_probs)
    print(f"CIFAR10 mean prob: {C10_meanprob:.5g}, std: {np.std(CIFAR10_probs):.5g}")
    print(f"CIFAR100 mean prob: {C100_meanprob:.5g}, std: {np.std(CIFAR100_probs):.5g}")
    print(f"SVHN mean prob: {SVHN_meanprob:.5g}, std: {np.std(SVHN_probs):.5g}")
    print(
        f"(mean(C10) - std(C10)) - (mean(C100) + std(C100)): {(C10_meanprob - np.std(CIFAR10_probs)) - (C100_meanprob + np.std(CIFAR100_probs))}")
    # create AUROC curve
    # tpr, fpr = compute_AUROC(CIFAR10_probs, CIFAR100_probs, SVHN_probs)
    cifar100_tpr, cifar100_fpr = compute_AUROC(CIFAR10_probs, CIFAR100_probs)
    cifar100_area = trapz(cifar100_tpr, x=cifar100_fpr) * -1
    print(f"CIFAR100 AUROC: {cifar100_area}")
    plt.plot(cifar100_fpr, cifar100_tpr, label=f"CIFAR100:{cifar100_area:.4f}")
    svhn_tpr, svhn_fpr = compute_AUROC(CIFAR10_probs, SVHN_probs)
    svhn_area = trapz(svhn_tpr, x=svhn_fpr) * -1
    print(f"SVHN AUROC: {svhn_area}")
    plt.plot(svhn_fpr, svhn_tpr, label=f"SVHN:{svhn_area:.4f}")
    plt.title(f"AUROC curves for CIFAR100 and SVHN")
    plt.legend()
    plt.savefig(f"results/histograms/{tgt_name}_AUROC.png")
    # plt.show()
    plt.figure()

    # Plot histograms for each dataset
    # print(len(CIFAR10_probs), min(CIFAR10_probs), max(CIFAR10_probs))
    # CIFAR10_probs = aggregate_data(CIFAR10_probs)
    # print(len(CIFAR10_probs), min(CIFAR10_probs), max(CIFAR10_probs))
    # CIFAR100_probs = aggregate_data(CIFAR100_probs)
    # SVHN_probs = aggregate_data(SVHN_probs)
    cifar100_precision = compute_precision(CIFAR10_probs, CIFAR100_probs)
    print(f"CIFAR100 precision: {cifar100_precision:.5f}")
    svhn_precision = compute_precision(CIFAR10_probs, SVHN_probs)
    print(f"SVHN precision: {svhn_precision:.5f}")

    CIFAR10_probs, C10_edges = create_online_hist(CIFAR10_probs, n_bins=100)
    CIFAR100_probs, C100_edges = create_online_hist(CIFAR100_probs, n_bins=100)
    SVHN_probs, SVHN_edges = create_online_hist(SVHN_probs, n_bins=100)
    plt.bar(C10_edges[:-1], CIFAR10_probs, width=np.diff(C10_edges), alpha=0.5, label='CIFAR10', align='edge')
    plt.bar(C100_edges[:-1], CIFAR100_probs, width=np.diff(C100_edges), alpha=0.5, label='CIFAR100', align='edge')
    plt.bar(SVHN_edges[:-1], SVHN_probs, width=np.diff(SVHN_edges), alpha=0.5, label='SVHN', align='edge')

    # plt.hist(CIFAR10_probs, bins=50, alpha=0.5, label='CIFAR10')
    # plt.hist(CIFAR100_probs, bins=50, alpha=0.5, label='CIFAR100')
    # plt.hist(SVHN_probs, bins=50, alpha=0.5, label='SVHN')

    # Add a legend and title to the plot
    plt.legend(loc='upper right')
    plt.title(f'{tgt_name}: Histograms of CIFAR10, CIFAR100, and SVHN image log probabilities')

    # save the plot
    plt.tight_layout()
    plt.savefig(f"results/histograms/{tgt_name}.png")

    # Show the plot
    # plt.show()
    results = {
        'CIFAR100_AUROC': cifar100_area,
        'SVHN_AUROC': svhn_area,
        'CIFAR100_precision': cifar100_precision,
        'SVHN_precision': svhn_precision,
        'CIFAR10_meanprob': C10_meanprob,
        'CIFAR100_meanprob': C100_meanprob,
        'SVHN_meanprob': SVHN_meanprob
    }
    return results


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
    # nf = define_flow_model(model_args, in_shape=dnn_outshape)
    nf = define_flow_model(model_args['batch'], model_args['k'], model_args['nf_type'], model_args['n_blocks'],
                           model_args['n_flows'], model_args['idim'], model_args['factor_out'], model_args['actnorm'],
                           model_args['act'], model_args['fc_end'], model_args['n_exact_terms'], model_args['affine'],
                           model_args['no_lu'], model_args['squeeze_first'], dnn_outshape, fc=model_args['fc'],
                           input_layer=model_args['init_layer'])
    print("number of parameters: {} ({:,})".format(get_model_params(nf), get_model_params(nf)))

    # --- DEFINE DHM --- #
    print("creating deep hybrid model with WRN-{}-{} dnn and {}-{}x{} nf".format(n, model_args['k'],
                                                                                 model_args['nf_type'],
                                                                                 model_args['n_blocks'],
                                                                                 model_args['n_flows']))
    print(f"flatten: {model_args['flatten']}, init layer: {model_args['init_layer']}, normalise features: "
          f"{model_args['normalise_features']}")
    dhm = DHM(dnn, nf, normalise_features=model_args['normalise_features'], flatten_features=model_args['flatten'],
              flow_in_shape=[dnn_outshape[0], dnn_outshape[1], model_args['k'] * 64])
    # dhm.load_state_dict(torch.load("checkpoints/testing/20230404_dry-silence-52.pth"))
    print("number of parameters: {} ({:,})".format(get_model_params(dhm), get_model_params(dhm)))

    # dhm.load_state_dict(model_dict['state_dict'])
    dhm.to(device)
    dhm.eval()
    test_input = torch.randn((1, 3, 32, 32))
    test_output = dhm(test_input.to(device))
    print("test input done")
    dhm.load_state_dict(model_dict['state_dict'])
    print("saved parameters loaded")

    # generate_histograms(dhm)
    # single_batch_analysis(dhm, model_args['batch'], args.name.split('.')[0])
    # feature_analysis(dhm, batch_size=args.batch, tgt_name=args.name.split('.')[0])
    # data_compare_feature_analysis(dhm, batch_size=128, tgt_name=args.name.split('.')[0])
    # test_feature_scaling(dhm, batch_size=128, tgt_name=args.name.split('.')[0])
    test_denser_dist(dhm, batch_size=128, tgt_name=args.name.split('.')[0])
