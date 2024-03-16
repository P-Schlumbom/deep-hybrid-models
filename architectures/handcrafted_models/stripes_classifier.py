import numpy as np
import torch
import torch.nn as nn


class StripesDetector(nn.Module):
    def __init__(self, in_channels=1):
        super().__init__()

        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=2, kernel_size=3, bias=False)
        print(self.conv.weight.shape)
        # prepare conv weights
        vert_filter = np.asarray([[-1, 0, 1],
                                  [-1, 0, 1],
                                  [-1, 0, 1]])
        hor_filter = np.asarray([[-1, -1, -1],
                                 [0, 0, 0],
                                 [1, 1, 1]])
        filter = np.stack((vert_filter, hor_filter))
        filter = torch.unsqueeze(torch.from_numpy(filter), 1).float()
        self.conv.weight = nn.Parameter(filter, requires_grad=False)
        self.maxpool = nn.AdaptiveMaxPool2d((1, 1))
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x):
        x = self.conv(x)
        x = torch.abs(x)
        #x = torch.nn.functional.relu(x)
        x = self.maxpool(x)
        #x = self.avgpool(x)
        y = x.view(x.size(0), -1)
        return y


if __name__ == "__main__":
    from datasets.data_classes import generate_striped_image, generate_diagonally_striped_images
    from torch.nn.functional import softmax
    from matplotlib import pyplot as plt

    from helpers import matplotlib_setup

    model = StripesDetector()
    model.to('cuda')

    for i in range(1):
        horizontal = i % 2 == 0
        test_im = generate_striped_image(horizontal, max_width=4)
        test_im = torch.unsqueeze(torch.from_numpy(test_im.transpose()), 0).to('cuda').float()
        pred = model(test_im)
        pred = softmax(pred)
        print(f"{i}: horizontal: {horizontal}, prediction: {pred[0][0].item()}, {pred[0][1].item()}")
        print(f"prediction result: {torch.argmax(pred) == (1 - horizontal)}")

    # plot feature vectors
    h_feats, v_feats, d_feats = [], [], []
    for i in range(300):
        type = i % 3
        if type == 0:
            test_im = generate_striped_image(True, max_width=4)
            label = "horizontal"
        elif type == 1:
            test_im = generate_striped_image(False, max_width=4)
            label = "vertical"
        else:
            test_im = generate_diagonally_striped_images(np.random.choice([True, False]), max_deviation=30, max_width=4)
            label = "diagonal"
        test_im = torch.unsqueeze(torch.from_numpy(test_im.transpose()), 0).to('cuda').float()

        feature = model(test_im)
        feature = feature.squeeze().cpu().numpy()
        #print(feature)
        if label == "horizontal":
            h_feats.append(feature)
        elif label == "vertical":
            v_feats.append(feature)
        elif label == "diagonal":
            d_feats.append(feature)
    h_feats = np.asarray(h_feats)
    v_feats = np.asarray(v_feats)
    d_feats = np.asarray(d_feats)
    plt.scatter(x=h_feats[:, 0], y=h_feats[:, 1], label="horizontal (ID)")
    plt.scatter(x=v_feats[:, 0], y=v_feats[:, 1], label="vertical (ID)")
    plt.scatter(x=d_feats[:, 0], y=d_feats[:, 1], label="diagonal (OOD)")
    plt.title("Hand Crafted Stripes Features")
    plt.legend()
    plt.savefig("../../experiments/stripes_experiments/figures/handcrafted_features.png")
    plt.show()

