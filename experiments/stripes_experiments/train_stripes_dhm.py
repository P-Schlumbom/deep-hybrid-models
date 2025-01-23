import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from numpy import trapz
import wandb
import pandas as pd
from sklearn.datasets import make_moons, make_blobs
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import matplotlib.pyplot as plt
from tqdm import tqdm
from torch.nn.functional import softmax

import os
import sys
# Get the absolute path of the directory containing your script
script_directory = os.path.dirname(os.path.abspath(__file__))
# Append the directory containing the modules to the sys.path list
module_directory = os.path.join(script_directory, "../..")
sys.path.append(module_directory)

from experiments.stripes_experiments.stripes_dhm import StripesDHM
from architectures.normalising_flows.residual_flows.layers import base as base_layers
from helpers.utils import running_average, set_seed
from helpers import matplotlib_setup

from datasets.synthetic_data import CustomDataset
from datasets.data_classes import generate_striped_image, generate_diagonally_striped_images, StripedImages, StripedOODImages

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


#
# HELPERS
#


def compute_logpx(logp, logdet):
    return logp - logdet


def compute_nf_loss(logpz, logdet):
    return (logdet - logpz).mean()


def experimental_nf_loss(logpz, logdet, labels):
    logprob = compute_logpx(logpz, logdet)
    mean_logprob = torch.mean(logprob)
    # compute per-class probability and overall equivalence score
    class_probs = []
    for c in range(2):
        class_probs.append(torch.mean(logprob[labels == c]))
    class_diff = max(class_probs) - min(class_probs)
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


def update_lipschitz(model, n_iterations):
    for m in model.modules():
        if isinstance(m, base_layers.SpectralNormConv2d) or isinstance(m, base_layers.SpectralNormLinear):
            m.compute_weight(update=True, n_iterations=n_iterations)
        if isinstance(m, base_layers.InducedNormConv2d) or isinstance(m, base_layers.InducedNormLinear):
            m.compute_weight(update=True, n_iterations=n_iterations)


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


def compute_AUROC(baseline_data, ood_data, n_samples=500):
    # bmin, bmax = np.min(baseline_data), np.max(baseline_data)
    bmin, bmax = min(np.min(baseline_data), np.min(ood_data)), max(np.max(baseline_data), np.max(ood_data))
    brange = bmax - bmin
    bstep = brange / n_samples

    TPR, FPR = [], []
    for i in range(n_samples + 1):
        threshold = bmin + (i * bstep)
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
    for i in range(n_samples + 1):
        threshold = bmin + (i * bstep)
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
    bmin, bmax = min(np.min(baseline_data), np.min(comparison_data)), max(np.max(baseline_data),
                                                                          np.max(comparison_data))
    thresholds = np.linspace(bmin, bmax, n_samples)

    FPR, FNR = [], []
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


def get_AUROC_score(dhm, ID_dataloader, OOD_dataloader):
    ID_probs = []
    OOD_probs = []
    ID_labels = []

    dhm.eval()
    for i, data in enumerate(ID_dataloader):
        inputs, labels = data
        inputs = inputs.permute(0, 3, 1, 2).to(device)
        y, logpz, logdet, z = model(inputs)
        logpx = compute_logpx(logpz, logdet)

        ID_probs.extend(logpx.detach().cpu().numpy().tolist())
        ID_labels.extend(labels.detach().cpu().numpy().tolist())
    for i, data in enumerate(OOD_dataloader):
        inputs, _ = data
        inputs = inputs.permute(0, 3, 1, 2).to(device)
        y, logpz, logdet, z = model(inputs)
        logpx = compute_logpx(logpz, logdet)

        OOD_probs.extend(logpx.detach().cpu().numpy().tolist())
    ID_labels = np.asarray(ID_labels)

    # create AUROC curve
    tpr, fpr = compute_AUROC(ID_probs, OOD_probs)
    area = trapz(tpr, x=fpr) * -1

    horizontal_probs = np.asarray(ID_probs)[ID_labels == 0]
    vertical_probs = np.asarray(ID_probs)[ID_labels == 1]
    h_meanprob = np.mean(horizontal_probs)
    v_meanprob = np.mean(vertical_probs)

    metrics = {
        'AUROC': area,
        'horizontal_meanprob': h_meanprob,
        'vertical_meanprob': v_meanprob
    }

    return metrics


def generate_features(dhm, ID_dataloader, OOD_dataloader):
    ID_feature_set = []
    OOD_feature_set = []
    ID_labels = []
    dhm.eval()
    for i, data in enumerate(ID_dataloader):
        inputs, labels = data
        inputs = inputs.permute(0, 3, 1, 2).to(device)
        y, _, _, _ = model(inputs)
        ID_feature_set.extend(y.squeeze().detach().cpu().numpy())
        ID_labels.extend(labels.detach().cpu().numpy().tolist())
    for i, data in enumerate(OOD_dataloader):
        inputs, _ = data
        inputs = inputs.permute(0, 3, 1, 2).to(device)
        y, _, _, _ = model(inputs)
        OOD_feature_set.extend(y.squeeze().detach().cpu().numpy())
    ID_features = np.stack(ID_feature_set, axis=0)
    OOD_features = np.stack(OOD_feature_set, axis=0)
    ID_labels = np.asarray(ID_labels)
    return ID_features, OOD_features, ID_labels


def compute_OOD_scores(dhm, ID_dataloader, OOD_dataloader, show_plots=False, custom_prefix=''):
    ID_probs = []
    OOD_probs = []
    ID_feature_set = []
    OOD_feature_set = []
    ID_labels = []

    dhm.eval()
    print("computing ID probabilities...")
    for i, data in enumerate(tqdm(ID_dataloader)):
        inputs, labels = data
        inputs = inputs.permute(0, 3, 1, 2).to(device)
        y, logpz, logdet, z = model(inputs)
        logpx = compute_logpx(logpz, logdet)

        ID_probs.extend(logpx.detach().cpu().numpy().tolist())
        ID_feature_set.extend(y.squeeze().detach().cpu().numpy())
        ID_labels.extend(labels.detach().cpu().numpy().tolist())
    for i, data in enumerate(tqdm(OOD_dataloader)):
        inputs, _ = data
        inputs = inputs.permute(0, 3, 1, 2).to(device)
        y, logpz, logdet, z = model(inputs)
        logpx = compute_logpx(logpz, logdet)

        OOD_probs.extend(logpx.detach().cpu().numpy().tolist())
        OOD_feature_set.extend(y.squeeze().detach().cpu().numpy())
    ID_features = np.stack(ID_feature_set, axis=0)
    OOD_features = np.stack(OOD_feature_set, axis=0)
    ID_labels = np.asarray(ID_labels)

    print(len(ID_probs))
    ID_meanprob = np.mean(ID_probs)
    OOD_meanprob = np.mean(OOD_probs)
    print(f"ID mean prob: {ID_meanprob:.5g}, std: {np.std(ID_probs):.5g}")
    results = {
        'ID_meanprob': ID_meanprob
    }

    print(f"OOD set mean prob: {OOD_meanprob:.5g}, std: {np.std(OOD_probs):.5g}")
    results[f'OOD_meanprob'] = OOD_meanprob

    print(
        f"(mean(C10) - std(C10)) - (mean(C100) + std(C100)): {(ID_meanprob - np.std(ID_probs)) - (OOD_meanprob + np.std(OOD_probs))}")

    # create AUROC curve
    tpr, fpr = compute_AUROC(ID_probs, OOD_probs)
    area = trapz(tpr, x=fpr) * -1
    print(f"OOD set AUROC: {area:.5g}")
    results[f'OOD_AUROC'] = area

    # compute AUPR scores
    # Cifar100 in and out AUPR scores
    OOD_in_precision, OOD_recall = compute_AUPR(ID_probs, OOD_probs)
    OOD_in_area = trapz(OOD_in_precision, x=OOD_in_precision) * -1
    # print results
    print(f"AUPR-in: {OOD_in_area}")
    # compute AUTC scores
    OOD_autc = compute_AUTC(ID_probs, OOD_probs)
    # print results
    print(f"CIFAR100 AUTC: {OOD_autc}")

    horizontal_probs = np.asarray(ID_probs)[ID_labels==0]
    vertical_probs = np.asarray(ID_probs)[ID_labels==1]
    h_meanprob = np.mean(horizontal_probs)
    v_meanprob = np.mean(vertical_probs)
    print("---")
    print(f"horizontal meanprob: {h_meanprob:.5g}\nvertical meanprob: {v_meanprob:.5g}\ndifference: {h_meanprob - v_meanprob:.5g}")
    print("---")

    if show_plots:
        ID_probs, ID_edges = create_online_hist(ID_probs, n_bins=100)
        OOD_probs, OOD_edges = create_online_hist(OOD_probs, n_bins=100)
        plt.bar(ID_edges[:-1], ID_probs, width=np.diff(ID_edges), alpha=0.5, label='ID Log Probabilities', align='edge')
        plt.bar(OOD_edges[:-1], OOD_probs, width=np.diff(OOD_edges), alpha=0.5, label='OOD  Log Probabilities', align='edge')
        plt.title("Stripes ID and OOD Log Probabilities")
        plt.legend()
        dir_path = 'figures'
        file_name = f'dhm{custom_prefix}_{n_blocks}-{dims}_lamb{lamb}_histogram_{epochs}.png'
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
        plt.savefig(os.path.join(dir_path, file_name))
        plt.show()

        # plot feature
        horizontal_features = ID_features[ID_labels==0]
        vertical_features = ID_features[ID_labels==1]
        plt.scatter(horizontal_features[:,0], horizontal_features[:,1], label='Horizontal Features')
        plt.scatter(vertical_features[:, 0], vertical_features[:, 1], label='Vertical Features')
        plt.scatter(OOD_features[:, 0], OOD_features[:, 1], label='OOD Features')
        plt.title("Stripes ID and OOD Features")
        plt.legend()
        plt.show()

    return results


def get_grid_volume(x_vals, y_vals, z_vals):
    total_volume = 0
    for i, y in enumerate(y_vals[:-1]):
        for j, x in enumerate(x_vals[:-1]):
            diffx = np.abs(x_vals[j+1] - x)
            diffy = np.abs(y_vals[i+1] - y)
            meanZ = (z_vals[i, j] + z_vals[i+1, j] + z_vals[i, j+1] + z_vals[i+1, j+1]) / 4
            grid_square_volume = diffx * diffy * meanZ
            total_volume += grid_square_volume
    return total_volume


def plot_heatmap(dhm, iteration, ID_loader, OOD_loader, plot_dim=400, plt_min=-0.1, plt_max=1.0, custom_prefix='', show_plot=True):
    # Perform grid search and generate heatmap

    x_values = np.linspace(plt_min, plt_max, plot_dim)
    y_values = np.linspace(plt_min, plt_max, plot_dim)

    #heatmap = np.zeros((len(x_values), len(y_values)))
    heatmap = np.zeros((len(y_values), len(x_values)))

    for j, y_val in enumerate(y_values):
        # Create a vector of x values for this row
        x_vector = x_values

        # Repeat the y_val to match the x_vector's shape and combine them
        y_vector = np.full_like(x_vector, y_val)
        xy_pairs = np.stack([x_vector, y_vector], axis=1)

        # Convert to a PyTorch tensor
        xy_pairs_tensor = torch.tensor(xy_pairs, dtype=torch.float32).to(device)

        # Pass the tensor through your model
        # Ensure your model can handle a batch of inputs if you pass multiple (x, y) pairs at once
        # row_output = model(xy_pairs_tensor).detach().numpy()  # You might need to modify this depending on your model's output
        with torch.no_grad():
            model.eval()
            output_probs = torch.exp(dhm.feature_logprob(xy_pairs_tensor)).squeeze().detach().cpu().numpy()  # normflow prob

        # Insert the computed row into the heatmap matrix
        heatmap[j, :] = output_probs

    # calculate heatmap volume
    flow_volume = get_grid_volume(x_values, y_values, heatmap)
    print(f"flow approx. volume: {flow_volume}")

    # Plot the heatmap
    #vmax_threshold = 0.05 if use_DHM or dhm_target == 'p(x)' else 1.0
    plt.figure(figsize=(12, 12))
    vmax_threshold = 0.05
    plt.imshow(heatmap, extent=(plt_min, plt_max, plt_min, plt_max), cmap='viridis', origin='lower', alpha=0.7, vmin=0, vmax=max(np.max(heatmap), vmax_threshold))
    plt.colorbar(label='Feature Probability')
    plt.title('DHM Probability Heatmap')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')

    # Plot the data points on top of the heatmap
    ID_features, OOD_features, ID_labels = generate_features(dhm, ID_loader, OOD_loader)
    horizontal_features = ID_features[ID_labels == 0]
    vertical_features = ID_features[ID_labels == 1]
    plt.scatter(horizontal_features[:, 0], horizontal_features[:, 1], alpha=0.6, s=50, label='Horizontal Features')
    plt.scatter(vertical_features[:, 0], vertical_features[:, 1], alpha=0.6, s=50, label='Vertical Features')
    plt.scatter(OOD_features[:, 0], OOD_features[:, 1], alpha=0.6, s=50, label='OOD Features')

    dir_path = 'figures'
    file_name = f'dhm{custom_prefix}_{n_blocks}-{dims}_lamb{lamb}_result_{iteration}.png'
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    plt.savefig(os.path.join(dir_path, file_name))

    if show_plot:
        plt.show()
    plt.clf()


#
# DATA PREPARATION
#


def prepare_striped_data(datamodel, batch=128, num_samples=15000, num_test_samples=500):
    trainset = CustomDataset(datamodel, num_samples)
    train_dataloader = DataLoader(trainset, batch_size=batch, shuffle=True)
    if num_test_samples is not None:
        testset = CustomDataset(datamodel, num_test_samples)
        test_dataloader = DataLoader(testset, batch_size=batch, shuffle=True)
        return train_dataloader, test_dataloader
    return train_dataloader, None


def train(model, lamb, epochs, criterion, dnn_optimiser, flow_optimiser, trainloader, testloader, OODloader, n_lipschitz_iters, use_exp_loss=False):
    figure_saving_checkpoints = [10, 20, 40, 80, 160, 320, 640]

    for epoch in range(epochs):
        mean_loss = 0.0
        mean_classifier_loss = 0.0
        mean_flow_loss = 0.0
        mean_acc = 0.0

        mean_logprob = 0.0
        mean_logdet = 0.0

        model.train()

        for i, data in enumerate(trainloader):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data
            inputs = inputs.permute(0, 3, 1, 2).to(device)  # .float()
            labels = labels.to(device)

            # zero the parameter gradients
            flow_optimiser.zero_grad()
            dnn_optimiser.zero_grad()

            # forward + backward + optimize
            y, logpz, logdet, z = model(inputs)

            loss_classifier = criterion(y, labels)
            if use_exp_loss:
                loss_flow = experimental_nf_loss(logpz, logdet, labels)
            else:
                loss_flow = compute_nf_loss(logpz, logdet)
            loss = compute_dhm_loss(loss_classifier, loss_flow, lamb=lamb)
            loss.backward()

            flow_optimiser.step()
            dnn_optimiser.step()

            # nfl.utils.update_lipschitz(model, n_iterations=50)
            update_lipschitz(model, n_iterations=n_lipschitz_iters)

            mean_loss = running_average(loss.item(), mean_loss, i)
            mean_classifier_loss = running_average(loss_classifier.item(), mean_classifier_loss, i)
            mean_flow_loss = running_average(loss_flow.item(), mean_flow_loss, i)
            mean_acc = running_average(torch.sum(torch.argmax(y, dim=1) == labels) / labels.shape[0], mean_acc, i)

            #train_feature_stats = update_feature_stats(train_feature_stats, features.detach().cpu().numpy(), i)
            mean_logprob = running_average(logpz.mean().item(), mean_logprob, i)
            mean_logdet = running_average(logdet.mean().item(), mean_logdet, i)

        val_loss = 0.0
        val_classifier_loss = 0.0
        val_flow_loss = 0.0
        val_acc = 0.0
        val_logprob = 0.0
        val_logdet = 0.0
        model.eval()
        with torch.no_grad():
            for j, data in enumerate(testloader, 0):
                inputs, labels = data
                inputs = inputs.permute(0, 3, 1, 2).to(device)  # .float()
                labels = labels.to(device)
                flow_optimiser.zero_grad()
                dnn_optimiser.zero_grad()
                y, logpz, logdet, z = model(inputs)

                loss_classifier = criterion(y, labels)
                if use_exp_loss:
                    loss_flow = experimental_nf_loss(logpz, logdet, labels)
                else:
                    loss_flow = compute_nf_loss(logpz, logdet)
                loss = compute_dhm_loss(loss_classifier, loss_flow, lamb=lamb)

                val_loss = running_average(loss.item(), val_loss, j)
                val_classifier_loss = running_average(loss_classifier.item(), val_classifier_loss, j)
                val_flow_loss = running_average(loss_flow.item(), val_flow_loss, j)
                val_acc = running_average(torch.sum(torch.argmax(y, dim=1) == labels) / labels.shape[0], val_acc, j)
                #val_feature_stats = update_feature_stats(val_feature_stats, features.detach().cpu().numpy(), j)
                val_logprob = running_average(logpz.mean().item(), val_logprob, j)
                val_logdet = running_average(logdet.mean().item(), val_logdet, j)

        val_metrics = get_AUROC_score(model, testloader, OODloader)
        # Print training progress
        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch + 1}/{epochs}], Loss: {val_loss:.4f}, Accuracy: {val_acc}, logdet: {mean_logdet}, AUROC: {val_metrics["AUROC"]}')

        results = {
            'epoch': epoch,
            'training': {
                'loss': mean_loss,
                'flow_loss': mean_flow_loss,
                'classifier_loss': mean_classifier_loss,
                'accuracy': mean_acc,
                'logdet': mean_logdet,
                'logprob': mean_logprob
            },
            'validation': {
                'loss': val_loss,
                'accuracy': val_acc,
                'flow_loss': val_flow_loss,
                'classifier_loss': val_classifier_loss,
                'logdet': val_logdet,
                'logprob': val_logprob
            },
            "class_metrics": val_metrics
        }
        wandb.log(results)
        """if (epoch + 1) in figure_saving_checkpoints or (epoch + 1) % 160 == 0:
            plot_heatmap(epoch + 1, plot_dim=heatmap_dim)
            
            plot_features(epoch + 1)
            # plot_feature_transformation_map(epoch + 1, plot_dim=20)
            plot_feature_transformation_grid(epoch + 1, plot_dim=40)
            compute_testset_probability()"""
    results = compute_OOD_scores(model, testloader, OODloader, show_plots=True, custom_prefix='-expl' if use_exp_loss else '')


if __name__ == "__main__":

    use_exp_loss = False
    lamb = 0.01  #0.05#0.001
    epochs = 160
    batch = 256#64
    heatmap_dim = 200
    lr = 1e-3
    weight_decay = 1e-5#16e-4

    sn = True
    coeff = 0.9

    n_lipschitz_iters = 5
    n_blocks = 50  # 50
    dims = '32'  # 128-128

    num_samples = 2500
    num_test_samples = 500

    seed = -1
    if seed == -1:
        seed = np.random.randint(2 ** 16 - 1)
    set_seed(seed)

    configs = {'use_exp_loss': use_exp_loss, 'lambda': lamb, 'epochs': epochs, 'batch': batch, 'lr': lr,
               'weight_decay': weight_decay, 'sn': sn, 'coeff': coeff, 'n_lipschitz_iters': n_lipschitz_iters,
               'n_blocks': n_blocks, 'dims': dims, 'num_samples': num_samples, 'seed': seed}
    wandb.init(
        project="stripes-dhm",
        config=configs,
        mode="disabled",
        group="main"
    )

    datamodel = StripedImages(max_width=4)
    trainloader, testloader = prepare_striped_data(datamodel, batch=batch, num_samples=num_samples, num_test_samples=num_test_samples)
    datashape = datamodel.get_shape()
    n_classes = datamodel.n_classes

    OODmodel = StripedOODImages(max_width=4, max_deviation=30)
    OODtrainloader, OODtestloader = prepare_striped_data(OODmodel, batch=batch, num_samples=num_test_samples, num_test_samples=None)

    model = StripesDHM(dims=dims, n_blocks=n_blocks, dnn_coeff=coeff)
    model.to(device)

    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()

    flow_optimiser = optim.Adam(
        model.flow.parameters(),
        lr=lr,
        weight_decay=weight_decay
    )
    dnn_optimiser = optim.SGD(model.dnn.parameters(), lr=0.01)

    # Initialize ActNorm
    x_np = np.random.randn(64, 1, 9, 9)
    x = torch.tensor(x_np).float().to(device)
    _, _, _, _ = model(x)


    # train model
    train(model, lamb=lamb, epochs=epochs, criterion=criterion, dnn_optimiser=dnn_optimiser,
          flow_optimiser=flow_optimiser, trainloader=trainloader, testloader=testloader,
          OODloader=OODtrainloader, n_lipschitz_iters=n_lipschitz_iters, use_exp_loss=use_exp_loss)

    plot_heatmap(model, epochs, trainloader, OODtrainloader, plt_max=0.7, custom_prefix='-expl' if use_exp_loss else '')

    print(model.dnn.conv.weight)

