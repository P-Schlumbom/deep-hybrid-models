import io
import base64
from tqdm import tqdm
from math import log
import numpy as np
from numpy import trapz
from os.path import join
import matplotlib as mpl
from matplotlib import pyplot as plt
import plotly.express as px
import plotly.graph_objs as go
from dash import Dash, dcc, html, Input, Output, no_update, callback
import pandas as pd

from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import MultiStepLR
import torchvision
import torchvision.transforms as transforms

from sklearn.manifold import TSNE

from datasets.dataset_labels import c10_id2label, c100_id2label

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Helper functions

def untransform_image(image):
    return image * 0.25 + 0.5


def np_image_to_base64(im_matrix):
    #print(im_matrix.shape)
    im = Image.fromarray(im_matrix)
    buffer = io.BytesIO()
    im.save(buffer, format="jpeg")
    encoded_image = base64.b64encode(buffer.getvalue()).decode()
    im_url = "data:image/jpeg;base64, " + encoded_image
    return im_url


def generate_tsne_with_images(dhm, batch_size=128, max_samples=1000, return_table=True, plot_classes=False, max_size=20):

    feature_set = []
    dataset_labels = []
    class_labels = []
    logprobs = []
    images = []
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
        images.extend(inputs.numpy())
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
        images.extend(inputs.numpy())
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
        images.extend(inputs.numpy())
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
    images = [image.transpose() for image in images]
    print(len(images), images[0].shape)

    # create tsne plot
    tsne = TSNE(n_components=2, random_state=0)
    tsne_data = tsne.fit_transform(features)

    display_sizes = [x for x in logprobs]
    minp, maxp = min(display_sizes), max(display_sizes)
    marker_sizes = [(x - minp) / (maxp - minp) * max_size for x in display_sizes]
    plt_data = pd.DataFrame(
        {'x': tsne_data[:, 0], 'y': tsne_data[:, 1], 'dataset': dataset_labels, 'class': class_labels,
         'logprob': logprobs, 'images': images})

    # dash app?
    fig = px.scatter(plt_data, x='x', y='y', color='dataset', size=marker_sizes)
    fig.update_traces(
        hoverinfo="none",
        hovertemplate=None,
    )

    app = Dash(__name__)

    app.layout = html.Div(
        className="container",
        children=[
            dcc.Graph(id="graph", figure=fig, clear_on_unhover=True),
            dcc.Tooltip(id="graph-tooltip", direction="bottom"),
        ],
    )

    @callback(
        Output("graph-tooltip", "show"),
        Output("graph-tooltip", "bbox"),
        Output("graph-tooltip", "children"),
        Input("graph", "hoverData"),
    )
    def display_hover(hoverData):
        if hoverData is None:
            return False, no_update, no_update

        hover_data = hoverData["points"][0]
        bbox = hover_data["bbox"]
        num = hover_data["pointNumber"]

        im_matrix = (untransform_image(images[num]) * 255).astype(np.uint8)
        im_url = np_image_to_base64(im_matrix)
        children = [
            html.Div([
                html.Img(
                    src=im_url,
                    style={"width": "50px", 'display': 'block', 'margin': '0 auto'},
                ),
                html.P(f"{num}: {dataset_labels[num]} {class_labels[num]}", style={'font-weight': 'bold'})
            ])
        ]

        return True, bbox, children

    app.run(debug=False)

