import os.path

import numpy as np
import scipy.stats as stats

import wandb
from tqdm import tqdm
from os.path import join, exists

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import MultiStepLR
import torchvision
import torchvision.transforms as transforms

import argparse

from architectures.normalising_flows.residual_flows.layers import base as base_layers
from architectures.deep_hybrid_models.dhm import create_ires_dhm, DHM_iresflows
from test_OOD import generate_histograms, generate_tsne_plots, generate_multidata_tsne_plots, test_embedding_methods, \
    test_custom_datasets, compute_auroc_scores, compute_embeddings
from helpers.utils import running_average, print_model_params, get_model_params, set_seed
from datasets.synthetic_data import CustomDataset
from datasets.natural_data import LocalDataset, InAndOutLocalDataset
from datasets.data_classes import StripedImages, StripedOODImages

# in case CIFAR10 fails to download
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

parser = argparse.ArgumentParser(description="DHM trainer")

# wandb controls
parser.add_argument("--mode", default="online", type=str, choices=["online", "offline", "disabled"],
                    help="Wandb logging setting. Set to 'disabled' to prevent all logging (useful for debugging)")
parser.add_argument("--project_name", default="DHM-v2", type=str, help="project name to log the results to.")
parser.add_argument("--group_name", default="train_dhm_iresflow", type=str, help="group name to apply to run;"
                                                                                 " default is script name.")
# general controls
parser.add_argument("--src_name", type=str, default=None, help="(optional) name of the model file to load, won't load "
                                                               "anything if nothing is provided")
parser.add_argument("--save_checkpoints", type=eval, choices=[True, False], default=False,
                    help="if chosen, saves best (in terms of validation "
                         "accuracy) and final model checkpoints")
parser.add_argument("--dirpath", type=str, default="checkpoints/testing",
                    help="path of the dir to load and save the model to (checkpoints/testing by default)")
parser.add_argument("--test_model", type=eval, choices=[True, False], default=False,
                    help="if chosen, run a final test to calculate AUROC and "
                         "precision wrt CIFAR100 and SVHN datasets after training")
parser.add_argument("--test_every_epoch", type=eval, choices=[True, False], default=False,
                    help="if chosen, test and log OOD performance after "
                         "every epoch")
parser.add_argument("--seed", type=int, default=0, help="set seed; 0 by default, if set to -1 will use random seed")
parser.add_argument("--distribution_model", type=str, default="normflow", choices=["normflow", "mvn", "gmm"],
                    help="Pick which distribution model to use in the DHM; by default, uses normalising flow.")
parser.add_argument("--dataset", type=str, choices=['CIFAR10', 'stink-bugs', 'stripes'], default='CIFAR10',
                    help="Which dataset to train on; CIFAR10 by default.")
# training controls
parser.add_argument("--batch", default=16, type=int, help="batch size")
parser.add_argument("--epochs", default=10, type=int, help="maximum epochs")
parser.add_argument("--lr", default=1e-4, type=float, help="learning rate used for nf optimiser")
parser.add_argument("--n_classes", default=10, type=int, help="number of classes to classify")
parser.add_argument("--lamb", default=0.06, type=float, help="the constant which determines the influence of the "
                                                             "normalising flow loss")
parser.add_argument("--lr_schedule", default="60-120-160", type=str, help="set the schedule for when the learning "
                                                                          "rate should be dropped (SGD only). "
                                                                          "Format is '-' delimited epoch, e.g. "
                                                                          "25-65-115-180")
parser.add_argument("--additional_loss", default="none", choices=["none", "max_dist", "class_adjust"],
                    help="Include an additional loss term (for experimental losses)")
# dhm args
parser.add_argument("--flatten", type=eval, choices=[True, False], default=False,
                    help="flatten the DNN features before passing them on to the flow"
                         " model")
parser.add_argument("--normalise_features", type=eval, choices=[True, False], default=False,
                    help="normalise DNN features (rescale to 0-1) before "
                         "passing them on to the flow model")
parser.add_argument("--norm_ord", type=int, default=torch.inf, help="normalisation type, infinity norm by default")
parser.add_argument("--common_features", type=eval, choices=[True, False], default=True, help="If true, the "
                                                                                              "classifier segment receives the "
                                                                                              "features after they've been "
                                                                                              "prepared for the normalising flow.")
parser.add_argument("--adaptive_lambda", type=eval, choices=[True, False], default=False,
                    help="If active, will automatically adjust lambda to the expected future flow loss. "
                         "Given the lambda parameter l, adaptive lambda = l / (est_flow_loss)")
# resnet params
parser.add_argument("--N", default=4, type=int, help="number of blocks per group; total number of convolutional layers "
                                                     "n = 6N + 4, so n = 28 -> N = 4")
parser.add_argument("--k", default=10, type=int, help="multiplier for the baseline width of each block; k=1 -> basic "
                                                      "resnet, k>1 -> wide resnet")
parser.add_argument("--sn", type=eval, choices=[True, False], default=True,
                    help="Whether or not to use Spectral Normalization (SN)")
parser.add_argument("--n_power_iter", default=1, type=int, help="Spectral Normalization parameter, number of power "
                                                                "iterations")
parser.add_argument("--dnn_coeff", default=6.0, type=float, help="upper bound coefficient for spectral normalisation - "
                                                                 "coeff=1.0 is basic SN implementation")
parser.add_argument('--activate_features', type=eval, choices=[True, False], default=False,
                    help="Whether or not to pass the classifier features to the relu "
                         "activation function before sending them to the normalising flow.")
parser.add_argument("--bottleneck", default=None, type=int, help="Use a bottleneck layer to reduce the "
                                                                 "feature dimensions to the chosen size. Disabled by "
                                                                 "default, enabled when a bottleneck size is provided.")

# resflow args
# build nnet parameters
parser.add_argument('--vnorms', type=str, default='222222')
parser.add_argument('--learn-p', type=eval, choices=[True, False], default=False)
parser.add_argument('--mixed', type=eval, choices=[True, False], default=True)
parser.add_argument('--nf_coeff', type=float, default=0.9)
parser.add_argument('--n-lipschitz-iters', type=int, default=5)
parser.add_argument('--atol', type=float, default=None)
parser.add_argument('--rtol', type=float, default=None)

# define nf model parameters
parser.add_argument('--dims', type=str, default='128-128-128')
parser.add_argument('--act', type=str, choices=ACTIVATION_FNS.keys(), default='swish')
parser.add_argument('--actnorm', type=eval, choices=[True, False], default=False)  # ?
parser.add_argument('--n_blocks', type=int, default=10)
parser.add_argument('--n-dist', choices=['geometric', 'poisson'], default='geometric')
parser.add_argument('--n-power-series', type=int, default=None)
parser.add_argument('--exact-trace', type=eval, choices=[True, False], default=False)
parser.add_argument('--brute-force', type=eval, choices=[True, False], default=False)
parser.add_argument('--n-samples', type=int, default=1)
parser.add_argument('--batchnorm', type=eval, choices=[True, False], default=False)
parser.add_argument("--init_layer", default=None, type=str, choices=[None, "logit", "norm"])


# ---------------------------HELPERS---------------------------------------------------------------------------------- #

class AddUniformNoise(object):
    def __init__(self, scale=1.0):
        self.scale = scale

    def __call__(self, tensor):
        return tensor + (torch.rand(tensor.shape) * self.scale)

    def __repr__(self):
        return self.__class__.__name__ + 'uniform_noise'


def freeze_dhm(dhm_model, dnn_setting=True, nf_setting=True):
    for param in dhm_model.dnn.parameters():
        param.requires_grad = dnn_setting
    for param in dhm_model.fc.parameters():
        param.requires_grad = dnn_setting
    for param in dhm_model.flow.parameters():
        param.requires_grad = nf_setting


def get_gradient_norm(model):
    """
    Compute the norm of the model gradients (should be useful for identifying exploding/vanishing gradients)
    :param model:
    :return:
    """
    total_norm = 0
    for param in model.parameters():
        if param.grad is not None:
            param_norm = param.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
    total_norm = total_norm ** 0.5
    return total_norm


def update_feature_stats(feature_stats, features: np.array, i):
    min = np.mean(np.min(features, axis=1))  # assumes features are along axis 1
    mean = np.mean(features)
    max = np.mean(np.max(features, axis=1))
    skew = np.mean(stats.skew(features, axis=1, bias=False))
    lq = np.mean(np.quantile(features, 0.25, axis=1))
    uq = np.mean(np.quantile(features, 0.75, axis=1))
    median = np.mean(np.median(features, axis=1))
    stdev = np.mean(np.std(features, axis=1))
    l2 = np.mean(np.sqrt(np.sum(np.power(features, 2), axis=1)))
    # mean_loss = running_average(loss.item(), mean_loss, i)
    feature_stats['feature_min'] = running_average(min, feature_stats['feature_min'], i)
    feature_stats['feature_mean'] = running_average(mean, feature_stats['feature_mean'], i)
    feature_stats['feature_max'] = running_average(max, feature_stats['feature_max'], i)
    feature_stats['feature_skew'] = running_average(skew, feature_stats['feature_skew'], i)
    feature_stats['feature_lq'] = running_average(lq, feature_stats['feature_lq'], i)
    feature_stats['feature_uq'] = running_average(uq, feature_stats['feature_uq'], i)
    feature_stats['feature_median'] = running_average(median, feature_stats['feature_median'], i)
    feature_stats['feature_stdev'] = running_average(stdev, feature_stats['feature_stdev'], i)
    feature_stats['feature_l2_norm'] = running_average(l2, feature_stats['feature_l2_norm'], i)
    return feature_stats


def update_latent_stats(latent_stats, z, i):
    min = np.mean(np.min(z, axis=1))  # assumes features are along axis 1
    mean = np.mean(z)
    max = np.mean(np.max(z, axis=1))
    lq = np.mean(np.quantile(z, 0.25, axis=1))
    uq = np.mean(np.quantile(z, 0.75, axis=1))
    stdev = np.mean(np.std(z, axis=1))
    l2 = np.mean(np.sqrt(np.sum(np.power(z, 2), axis=1)))
    # mean_loss = running_average(loss.item(), mean_loss, i)
    latent_stats['latent_min'] = running_average(min, latent_stats['latent_min'], i)
    latent_stats['latent_mean'] = running_average(mean, latent_stats['latent_mean'], i)
    latent_stats['latent_max'] = running_average(max, latent_stats['latent_max'], i)
    latent_stats['latent_lq'] = running_average(lq, latent_stats['latent_lq'], i)
    latent_stats['latent_uq'] = running_average(uq, latent_stats['latent_uq'], i)
    latent_stats['latent_stdev'] = running_average(stdev, latent_stats['latent_stdev'], i)
    latent_stats['latent_l2_norm'] = running_average(l2, latent_stats['latent_l2_norm'], i)
    return latent_stats


# ---------------------------HELPERS---------------------------------------------------------------------------------- #


def create_custom_dataloaders(datamodel, batch=32, num_samples=1000, num_test_samples=500):
    train_dataset = CustomDataset(datamodel, num_samples)
    train_dataloader = DataLoader(train_dataset, batch_size=batch, shuffle=True)
    test_dataset = CustomDataset(datamodel, num_test_samples)
    test_dataloader = DataLoader(test_dataset, batch_size=batch, shuffle=True)
    return train_dataloader, test_dataloader


def compute_nf_loss(logpz, logdet):
    return (logdet - logpz).mean()


def experimental_nf_loss(logpz, logdet, labels):
    logprob = logpz - logdet
    mean_logprob = torch.mean(logprob)
    # compute per-class probability and overall equivalence score
    class_probs = torch.zeros(args.n_classes)
    for c in range(args.n_classes):
        class_probs[c] = torch.mean(logprob[labels == c])
    class_diff = max(class_probs[~class_probs.isnan()]) - min(class_probs[~class_probs.isnan()])
    adjusted_logprob = mean_logprob - class_diff
    return -adjusted_logprob


def compute_dhm_loss(dnn_loss, nf_loss, lamb=1.0):
    """
    According to paper, DHM objective is logp(y|x) + lambda*logp(x),
    Where logp(x) = |det(x)|+logp(z)
    :param y:
    :param logpz:
    :param logdet:
    :return:
    """
    return dnn_loss + (lamb * nf_loss)


def compute_distance_loss(feature_batch, order=2):
    # print(feature_batch.shape)
    B = feature_batch.shape[0]
    D = feature_batch.shape[1]
    mean_dist = torch.mean(torch.cdist(feature_batch, feature_batch, p=2))
    # discount 0 distances
    mean_distance = (mean_dist * B * B) / (B * B - B)  # disregard the diagonal of zeros for the mean
    return -torch.pow(mean_distance, order) / D  # the greater the average distance, the better


def compute_logpx(logp, logdet):
    return logp - logdet


def update_lipschitz(model, n_iterations):
    for m in model.modules():
        if isinstance(m, base_layers.SpectralNormConv2d) or isinstance(m, base_layers.SpectralNormLinear):
            m.compute_weight(update=True, n_iterations=n_iterations)
        if isinstance(m, base_layers.InducedNormConv2d) or isinstance(m, base_layers.InducedNormLinear):
            m.compute_weight(update=True, n_iterations=n_iterations)


def get_grid_volume(X, Y, Z):
    total_volume = 0
    for i, x in enumerate(X[:-1]):
        for j, y in enumerate(Y[:-1]):
            diffx = X[i + 1] - x
            diffy = Y[j + 1] - y
            meanZ = (Z[i, j] + Z[i + 1, j] + Z[i, j + 1] + Z[i + 1, j + 1]) / 4
            grid_square_volume = diffx * diffy * meanZ
            total_volume += grid_square_volume
    return total_volume


# ---------------------------DATA LOADERS----------------------------------------------------------------------------- #

def prepare_stinkbug_data(size=64, num_workers=6):
    train_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Resize(size=size),
            transforms.RandomHorizontalFlip(0.5),
            transforms.RandomCrop(size=size, padding=4),
            AddUniformNoise(scale=1.0),  # images should be in 0-255 range
            transforms.Normalize(np.array([0.5, 0.5, 0.5]), np.array([0.25, 0.25, 0.25])),
        ]
    )
    val_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Resize(size=size),
            transforms.CenterCrop(size=size),
            transforms.Normalize(np.array([0.5, 0.5, 0.5]), np.array([0.25, 0.25, 0.25])),
        ]
    )
    trainset = InAndOutLocalDataset("data/stink-bugs", data_ref_src="data/stink-bugs/species-labels.csv", train=True,
                                    in_distribution=True, transform=train_transform)
    testset = InAndOutLocalDataset("data/stink-bugs", data_ref_src="data/stink-bugs/species-labels.csv", train=False,
                                   in_distribution=True, transform=val_transform)
    OODset = InAndOutLocalDataset("data/stink-bugs", data_ref_src="data/stink-bugs/species-labels.csv", train=True,
                                  in_distribution=False, train_prop=1.0,
                                  transform=val_transform)  # use 100% of the OOD samples
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch, shuffle=True, num_workers=num_workers)
    testloader = torch.utils.data.DataLoader(testset, batch_size=args.batch, shuffle=True, num_workers=num_workers)
    OODloader = torch.utils.data.DataLoader(OODset, batch_size=args.batch, shuffle=True, num_workers=num_workers)

    return trainloader, testloader, OODloader


def prepare_CIFAR10_data():
    train_transform = transforms.Compose(
        [transforms.ToTensor(),
         AddUniformNoise(scale=1.0 / 256.0),
         transforms.Normalize(np.array([125.3, 123.0, 113.9]) / 255.0,
                              np.array([63.0, 62.1, 66.7]) / 255.0),
         transforms.RandomHorizontalFlip(0.5),
         transforms.RandomCrop(size=32, padding=4, padding_mode='reflect')])
    val_transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize(np.array([125.3, 123.0, 113.9]) / 255.0,
                              np.array([63.0, 62.1, 66.7]) / 255.0),
         # transforms.RandomHorizontalFlip(0.5),
         # transforms.RandomCrop(size=32, padding=4, padding_mode='reflect')
         ])

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                            download=True, transform=train_transform)
    testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                           download=True, transform=val_transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch,
                                              shuffle=True, num_workers=6)
    testloader = torch.utils.data.DataLoader(testset, batch_size=args.batch,
                                             shuffle=False, num_workers=6)
    return trainloader, testloader


def prepare_CIFAR100_OOD_data():
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize(np.array([125.3, 123.0, 113.9]) / 255.0,
                              np.array([63.0, 62.1, 66.7]) / 255.0),
         ])
    CIFAR100_dataset = torchvision.datasets.CIFAR100(root='./data', train=False,
                                                     download=True, transform=transform)
    CIFAR100_loader = torch.utils.data.DataLoader(CIFAR100_dataset, batch_size=args.batch,
                                                  shuffle=True, num_workers=6)
    return CIFAR100_loader


def prepare_striped_data(datamodel, num_samples=15000, num_test_samples=500):
    trainset = CustomDataset(datamodel, num_samples)
    train_dataloader = DataLoader(trainset, batch_size=args.batch, shuffle=True)
    if num_test_samples is not None:
        testset = CustomDataset(datamodel, num_test_samples)
        test_dataloader = DataLoader(testset, batch_size=args.batch, shuffle=True)
        return train_dataloader, test_dataloader
    return train_dataloader, None


def get_lambda(est_final_loss):
    if args.adaptive_lambda:
        return abs(args.lamb / est_final_loss)
    return args.lamb


# ---------------------------MAIN PROGRAM----------------------------------------------------------------------------- #


def train(args, model: DHM_iresflows, flow_optimizer, dnn_optimizer, trainloader, testloader, scheduler=None):
    update_l = True
    test_results = None
    best_val_acc = 0
    best_val_loss = np.inf
    M = 64 * args.k * (model.dnn_out_size[0] * model.dnn_out_size[1])  # number of elements per sample
    if args.flatten:  # when using the avgpooling layer, output is the number of channels
        M = 64 * args.k  # if flatten, M is smaller
    print(f"args.k = {args.k}, model.dnn_out_size = {model.dnn_out_size}")
    print(f"M: {M}")

    # flow optimiser scheduler
    # flow_scheduler = MultiStepLR(flow_optimizer, milestones=schedule, gamma=0.2)

    feature_stats_dict = {
        'feature_min': 0.0,
        'feature_mean': 0.0,
        'feature_max': 0.0,
        'feature_skew': 0.0,
        'feature_lq': 0.0,
        'feature_uq': 0.0,
        'feature_median': 0.0,
        'feature_stdev': 0.0,
        'feature_l2_norm': 0.0,
    }

    latent_stats_dict = {
        'latent_min': 0.0,
        'latent_mean': 0.0,
        'latent_max': 0.0,
        'latent_lq': 0.0,
        'latent_uq': 0.0,
        'latent_stdev': 0.0,
        'latent_l2_norm': 0.0,
    }

    first_loss = 0.0
    mean_global_loss = 0.0
    est_final_loss = 1.0

    # with tqdm(range(args.epochs)) as pbar:
    for epoch in range(args.epochs):
        mean_loss = 0.0
        mean_classifier_loss = 0.0
        mean_flow_loss = 0.0
        mean_acc = 0.0

        mean_logprob = 0.0
        mean_logdet = 0.0

        train_feature_stats = feature_stats_dict.copy()
        train_latent_stats = latent_stats_dict.copy()
        model.train()
        epoch_labels = []
        for i, data in enumerate(tqdm(trainloader)):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data
            epoch_labels.extend(list(labels))
            inputs = inputs.to(device)  # .float()
            labels = labels.to(device)

            # zero the parameter gradients
            flow_optimizer.zero_grad()
            dnn_optimizer.zero_grad()

            # forward + backward + optimize
            y, logpz, logdet, z, features = model(inputs, return_features=True)

            loss_classifier = criterion(y, labels)
            if args.additional_loss == "class_adjust":
                loss_flow = experimental_nf_loss(logpz, logdet, labels)
            else:
                loss_flow = compute_nf_loss(logpz, logdet)
            loss = compute_dhm_loss(loss_classifier, loss_flow, lamb=get_lambda(est_final_loss))
            if args.additional_loss == "max_dist":
                feature_loss = compute_distance_loss(features)
                loss += feature_loss
            loss.backward()

            flow_optimizer.step()
            dnn_optimizer.step()

            update_lipschitz(model, n_iterations=args.n_lipschitz_iters)

            # calculate statistics
            if torch.isnan(loss):
                print(f"classifier loss: {loss_classifier}, flow loss: {loss_flow}")
                print("Loss is NaN! Breaking...")
                break
            mean_loss = running_average(loss.item(), mean_loss, i)
            mean_classifier_loss = running_average(loss_classifier.item(), mean_classifier_loss, i)
            mean_flow_loss = running_average(loss_flow.item(), mean_flow_loss, i)
            mean_acc = running_average(torch.sum(torch.argmax(y, dim=1) == labels) / labels.shape[0], mean_acc, i)

            train_feature_stats = update_feature_stats(train_feature_stats, features.detach().cpu().numpy(), i)
            train_latent_stats = update_latent_stats(train_latent_stats, z.detach().cpu().numpy(), i)
            mean_logprob = running_average(logpz.mean().item(), mean_logprob, i)
            mean_logdet = running_average(logdet.mean().item(), mean_logdet, i)

        val_loss = 0.0
        val_classifier_loss = 0.0
        val_flow_loss = 0.0
        val_acc = 0.0
        val_feature_stats = feature_stats_dict.copy()
        val_latent_stats = latent_stats_dict.copy()
        model.eval()
        with torch.no_grad():
            for j, data in enumerate(testloader, 0):
                inputs, labels = data
                inputs = inputs.to(device)  # .float()
                labels = labels.to(device)
                flow_optimizer.zero_grad()
                dnn_optimizer.zero_grad()
                y, logpz, logdet, z, features = model(inputs, return_features=True)

                loss_classifier = criterion(y, labels)
                if args.additional_loss == "class_adjust":
                    loss_flow = experimental_nf_loss(logpz, logdet, labels)
                else:
                    loss_flow = compute_nf_loss(logpz, logdet)
                loss = compute_dhm_loss(loss_classifier, loss_flow, lamb=get_lambda(est_final_loss))
                if args.additional_loss == "max_dist":
                    feature_loss = compute_distance_loss(features)
                    loss += feature_loss

                val_loss = running_average(loss.item(), val_loss, j)
                val_classifier_loss = running_average(loss_classifier.item(), val_classifier_loss, j)
                val_flow_loss = running_average(loss_flow.item(), val_flow_loss, j)
                val_acc = running_average(torch.sum(torch.argmax(y, dim=1) == labels) / labels.shape[0], val_acc, j)
                val_feature_stats = update_feature_stats(val_feature_stats, features.detach().cpu().numpy(), j)
                val_latent_stats = update_latent_stats(val_latent_stats, z.detach().cpu().numpy(), j)
            if args.test_every_epoch:
                test_results = generate_histograms(model, batch_size=args.batch, tgt_name=f"test", test_unimodal=True)

        # update final loss estimate
        mean_global_loss = running_average(mean_flow_loss, mean_global_loss, epoch)

        if args.save_checkpoints and val_acc > best_val_acc:
            model_dict = {
                'state_dict': model.state_dict(),
                'args': vars(args)
            }
            name = wandb.run.name
            torch.save(model_dict, join(args.dirpath, f"{name}_best-acc.pth"))
            best_val_acc = val_acc
        if args.save_checkpoints and val_loss < best_val_loss:
            model_dict = {
                'state_dict': model.state_dict(),
                'args': vars(args)
            }
            name = wandb.run.name
            torch.save(model_dict, join(args.dirpath, f"{name}_best-loss.pth"))
            best_val_loss = val_loss

        if scheduler:
            scheduler.step()

        tqdm.write(
            f"epoch {epoch}: train loss: {mean_loss:.5f}, classifier loss: {mean_classifier_loss:.5f}, "
            f"NF loss: {mean_flow_loss:.5f}, train acc: {mean_acc:.5f}, "
            f"val loss: {val_loss:.5f}, val acc: {val_acc:.5f}, logprob: {mean_logprob}, logdet: {mean_logdet}"
        )
        print(train_feature_stats, val_feature_stats)
        print(val_latent_stats)
        print(f"epoch label counts: {np.bincount(np.asarray(epoch_labels))}")
        results = {
            "epoch": epoch,
            "train_loss": mean_loss,
            "train_classifier_loss": mean_classifier_loss,
            "train_flow_loss": mean_flow_loss,
            "train_acc": mean_acc,
            "train_mean_logprob": mean_logprob,
            "train_mean_logdet": mean_logdet,
            "val_loss": val_loss,
            "val_classifier_loss": val_classifier_loss,
            "val_flow_loss": val_flow_loss,
            "val_acc": val_acc,
            "train_feature_stats": train_feature_stats,
            "val_feature_stats": val_feature_stats,
            "train_latent_stats": train_latent_stats,
            "val_latent_stats": val_latent_stats,
            "lr": scheduler.get_last_lr()[0] if scheduler else args.lr,
            "est_final_loss": est_final_loss,
            "lambda": get_lambda(est_final_loss)
        }
        if args.test_every_epoch:
            results.update(test_results)
        wandb.log(results)

        print(f"Best validation accuracy: {best_val_acc}")
    print('Finished Training')
    return results


def train_alternating(args, model: DHM_iresflows, flow_optimizer, dnn_optimizer, trainloader, testloader,
                      scheduler=None):
    nf_epochs = 50
    train_nf = False
    update_l = True
    test_results = None
    best_val_acc = 0
    best_val_loss = np.inf
    M = 64 * args.k * (model.dnn_out_size[0] * model.dnn_out_size[1])  # number of elements per sample
    if args.flatten:  # when using the avgpooling layer, output is the number of channels
        M = 64 * args.k  # if flatten, M is smaller
    print(f"args.k = {args.k}, model.dnn_out_size = {model.dnn_out_size}")
    print(f"M: {M}")

    # flow optimiser scheduler
    # flow_scheduler = MultiStepLR(flow_optimizer, milestones=schedule, gamma=0.2)

    feature_stats_dict = {
        'feature_min': 0.0,
        'feature_mean': 0.0,
        'feature_max': 0.0,
        'feature_skew': 0.0,
        'feature_lq': 0.0,
        'feature_uq': 0.0,
        'feature_median': 0.0,
        'feature_stdev': 0.0,
        'feature_l2_norm': 0.0,
    }

    first_loss = 0.0
    mean_global_loss = 0.0
    est_final_loss = 1.0

    # with tqdm(range(args.epochs)) as pbar:
    for epoch in range(args.epochs + nf_epochs):
        if epoch > args.epochs:
            train_nf = True
        mean_loss = 0.0
        mean_classifier_loss = 0.0
        mean_flow_loss = 0.0
        mean_acc = 0.0

        mean_logprob = 0.0
        mean_logdet = 0.0

        train_feature_stats = feature_stats_dict.copy()
        model.train()
        epoch_labels = []
        for i, data in enumerate(tqdm(trainloader)):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data
            epoch_labels.extend(list(labels))
            inputs = inputs.to(device)  # .float()
            labels = labels.to(device)

            # zero the parameter gradients
            if train_nf:
                flow_optimizer.zero_grad()
            else:
                dnn_optimizer.zero_grad()

            # forward + backward + optimize
            # y, loss_flow, features = model(inputs, return_features=True)  # y, logpz, logdet, z
            y, logpz, logdet, z, features = model(inputs, return_features=True)
            # print("train function: ", features.min(), features.mean(), features.max())

            loss_classifier = criterion(y, labels)
            loss_flow = compute_nf_loss(logpz, logdet)
            # loss = compute_dhm_loss(loss_classifier, loss_flow, lamb=get_lambda(est_final_loss))
            if train_nf:
                loss = loss_flow
            else:
                loss = loss_classifier
            loss.backward()

            # clip gradients...
            # max_norm = 10.0
            # utils.clip_grad_norm_(model.parameters(), max_norm)

            if train_nf:
                flow_optimizer.step()
            else:
                dnn_optimizer.step()

            # nfl.utils.update_lipschitz(model, n_iterations=50)
            update_lipschitz(model, n_iterations=args.n_lipschitz_iters)

            # calculate statistics
            if torch.isnan(loss):
                print(f"classifier loss: {loss_classifier}, flow loss: {loss_flow}")
                print("Loss is NaN! Breaking...")
                break
            mean_loss = running_average(loss.item(), mean_loss, i)
            mean_classifier_loss = running_average(loss_classifier.item(), mean_classifier_loss, i)
            mean_flow_loss = running_average(loss_flow.item(), mean_flow_loss, i)
            mean_acc = running_average(torch.sum(torch.argmax(y, dim=1) == labels) / labels.shape[0], mean_acc, i)

            train_feature_stats = update_feature_stats(train_feature_stats, features.detach().cpu().numpy(), i)
            mean_logprob = running_average(logpz.mean().item(), mean_logprob, i)
            mean_logdet = running_average(logdet.mean().item(), mean_logdet, i)

        val_loss = 0.0
        val_classifier_loss = 0.0
        val_flow_loss = 0.0
        val_acc = 0.0
        val_feature_stats = feature_stats_dict.copy()
        model.eval()
        with torch.no_grad():
            for j, data in enumerate(testloader, 0):
                inputs, labels = data
                inputs = inputs.to(device)  # .float()
                labels = labels.to(device)
                if train_nf:
                    flow_optimizer.zero_grad()
                else:
                    dnn_optimizer.zero_grad()
                y, logpz, logdet, z, features = model(inputs, return_features=True)
                loss_classifier = criterion(y, labels)
                loss_flow = compute_nf_loss(logpz, logdet)
                if train_nf:
                    loss = loss_flow
                else:
                    loss = loss_classifier
                # loss = compute_dhm_loss(loss_classifier, loss_flow, lamb=get_lambda(est_final_loss))

                val_loss = running_average(loss.item(), val_loss, j)
                val_classifier_loss = running_average(loss_classifier.item(), val_classifier_loss, j)
                val_flow_loss = running_average(loss_flow.item(), val_flow_loss, j)
                val_acc = running_average(torch.sum(torch.argmax(y, dim=1) == labels) / labels.shape[0], val_acc, j)
                val_feature_stats = update_feature_stats(val_feature_stats, features.detach().cpu().numpy(), j)
            if args.test_every_epoch:
                test_results = generate_histograms(model, batch_size=args.batch, tgt_name=f"test", test_unimodal=True)
                #test_results = compute_auroc_scores(dhm, testloader, [OODloader])

        # update final loss estimate
        mean_global_loss = running_average(mean_flow_loss, mean_global_loss, epoch)
        if epoch == 0:
            first_loss = mean_global_loss
        elif epoch == 1:
            diff = abs(first_loss - mean_global_loss)
            est_final_loss = diff
        else:
            if mean_global_loss < (first_loss - est_final_loss):
                est_final_loss *= 2

        if args.save_checkpoints and val_acc > best_val_acc:
            model_dict = {
                'state_dict': model.state_dict(),
                'args': vars(args)
            }
            name = wandb.run.name
            torch.save(model_dict, join(args.dirpath, f"{name}_best-acc.pth"))
            best_val_acc = val_acc
        if args.save_checkpoints and val_loss < best_val_loss:
            model_dict = {
                'state_dict': model.state_dict(),
                'args': vars(args)
            }
            name = wandb.run.name
            torch.save(model_dict, join(args.dirpath, f"{name}_best-loss.pth"))
            best_val_loss = val_loss

        if scheduler:
            scheduler.step()
            # flow_scheduler.step()

        tqdm.write(
            f"epoch {epoch}: train loss: {mean_loss:.5f}, classifier loss: {mean_classifier_loss:.5f}, "
            f"NF loss: {mean_flow_loss:.5f}, train acc: {mean_acc:.5f}, "
            f"val loss: {val_loss:.5f}, val acc: {val_acc:.5f}, logprob: {mean_logprob}, logdet: {mean_logdet}"
        )
        print(train_feature_stats, val_feature_stats)
        print(f"epoch label counts: {np.bincount(np.asarray(epoch_labels))}")
        results = {
            "epoch": epoch,
            "train_loss": mean_loss,
            "train_classifier_loss": mean_classifier_loss,
            "train_flow_loss": mean_flow_loss,
            "train_acc": mean_acc,
            "train_mean_logprob": mean_logprob,
            "train_mean_logdet": mean_logdet,
            "val_loss": val_loss,
            "val_classifier_loss": val_classifier_loss,
            "val_flow_loss": val_flow_loss,
            "val_acc": val_acc,
            "train_feature_stats": train_feature_stats,
            "val_feature_stats": val_feature_stats,
            "lr": scheduler.get_last_lr()[0] if scheduler else args.lr,
            "est_final_loss": est_final_loss,
            "lambda": get_lambda(est_final_loss)
        }
        if args.test_every_epoch:
            results.update(test_results)
        wandb.log(results)

        print(f"Best validation accuracy: {best_val_acc}")
    print('Finished Training')
    return results


if __name__ == "__main__":
    args = parser.parse_args()
    seed = args.seed
    if seed == -1:
        seed = np.random.randint(2 ** 32 - 1)
    set_seed(seed)
    args.seed = seed
    model_code = f"DHM-{args.N * 6 + 4}-{args.k}-{args.n_blocks}-{len(args.dims.split('-'))}-{args.dims.split('-')[0]}"
    configs = vars(args)
    configs['model_code'] = model_code
    # for i in range(5):
    wandb.init(
        project=args.project_name,
        config=configs,
        mode=args.mode,
        group=args.group_name
    )

    # initialise directory if necessary
    if not os.path.exists(args.dirpath):
        os.mkdir(args.dirpath)

    # --- PREPARE DATA --- #

    if args.dataset == 'stripes':
        print("preparing stripes dataset...")
        # striped images data
        datamodel = StripedImages(max_width=4)
        trainloader, testloader = prepare_striped_data(datamodel)
        datashape = datamodel.get_shape()
        n_classes = datamodel.n_classes

        OODmodel = StripedOODImages(max_width=4, max_deviation=5)
        OODtrainloader, OODtestloader = prepare_striped_data(OODmodel, num_samples=500, num_test_samples=None)
    elif args.dataset == 'stink-bugs':
        print("preparing stink bugs dataset...")
        # stink bug dataset
        size = 64
        trainloader, testloader, OODloader = prepare_stinkbug_data(size=size)  # should try a few different sizes...
        datashape = (size, size, 3)
        n_classes = trainloader.dataset.num_classes
    else:
        print("preparing CIFAR10 dataset...")
        # CIFAR 10 data
        trainloader, testloader = prepare_CIFAR10_data()
        OODloader = prepare_CIFAR100_OOD_data()
        datashape = (32, 32, 3)
        n_classes = 10

    # --- DEFINE DHM --- #


    dhm = create_ires_dhm(
        input_size=datashape,
        n_classes=n_classes,
        bottleneck=args.bottleneck,
        N=args.N,
        k=args.k,
        common_features=args.common_features,
        normalise_features=args.normalise_features,
        flatten=args.flatten,
        sn=args.sn,
        n_power_iter=args.n_power_iter,
        dnn_coeff=args.dnn_coeff,
        n_blocks=args.n_blocks,
        dims=args.dims,
        actnorm=args.actnorm,
        act=args.act,
        n_dist=args.n_dist,
        n_power_series=args.n_power_series,
        exact_trace=args.exact_trace,
        brute_force=args.brute_force,
        n_samples=args.n_samples,
        batchnorm=args.batchnorm,
        vnorms=args.vnorms,
        learn_p=args.learn_p,
        mixed=args.mixed,
        nf_coeff=args.nf_coeff,
        n_lipschitz_iters=args.n_lipschitz_iters,
        atol=args.atol,
        rtol=args.atol,
        init_layer=args.init_layer,
        src_name=args.src_name,
        norm_ord=args.norm_ord,
    )

    dhm.to(device)

    criterion = nn.CrossEntropyLoss()

    flow_optimizer = optim.Adam(
        dhm.flow.parameters(),
        lr=args.lr,  # 1e-4,
        weight_decay=16e-4
    )

    dnn_optimizer = optim.SGD(
        # dhm.parameters(),  #
        [{'params': dhm.dnn.parameters()}, {'params': dhm.fc.parameters()}],
        nesterov=True,
        lr=0.05,
        momentum=0.9,
        weight_decay=5e-4
    )
    schedule = [int(milestone) for milestone in args.lr_schedule.split('-')]
    print(f"Dropping learning rates at epoch milestones: {schedule}")
    scheduler = MultiStepLR(dnn_optimizer, milestones=schedule, gamma=0.2)

    final_results = train(args, dhm, flow_optimizer, dnn_optimizer, trainloader, testloader, scheduler)

    model_dict = {
        'state_dict': dhm.state_dict(),
        'args': vars(args)
    }

    if not args.save_checkpoints:
        torch.save(model_dict, join(args.dirpath, "test.pth"))
    else:
        name = wandb.run.name
        torch.save(model_dict, join(args.dirpath, f"{name}_e{args.epochs}.pth"))

    output_string = f"{final_results['val_acc']};{final_results['val_loss']};"
    if args.test_model:
        tgt_name = "test"
        if args.mode != "disabled":
            tgt_name = wandb.run.name
        dhm.eval()
        test_results = generate_histograms(dhm, batch_size=max(args.batch // 2, 1),
                                           tgt_name=f"{tgt_name}_e{args.epochs}", )

        wandb.log(test_results)

        fig, tsne_results, data_table, histogram = generate_multidata_tsne_plots(dhm, batch_size=max(args.batch // 2, 1),
                                                                      return_table=True, plot_classes=False,
                                                                      max_size=30, use_metropolis=False)

        if args.mode == 'disabled':
            # fig.show()
            # histogram.show()
            pass
        if fig:
            wandb.log({'feature tsne plot': fig})
            wandb.log({'logprob histogram': histogram})
            wandb.log({
                'tsne_data': wandb.Table(dataframe=data_table),
                'tsne_results': tsne_results
            })
        output_string += f"{test_results['CIFAR100_AUROC']};{test_results['SVHN_AUROC']};"

    wandb.finish()
    print(output_string)
