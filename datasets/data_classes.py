import numpy as np
from numpy.random import multivariate_normal as multinorm_sample
from scipy.stats import norm, multivariate_normal
import torch


class BaseDataClass:
    def __init__(self):
        self.n_classes = None

    def sample(self, N=None):
        pass

    def get_shape(self):
        pass


def generate_striped_image(horizontal=True, max_width=1, image_width=9):
    image = np.zeros((image_width, image_width))
    i = 0
    #colour = np.random.choice([0, 1])
    colour = np.random.uniform()
    while i < image_width:
        width = np.random.randint(1, max_width + 1)
        if horizontal:
            image[i:i+width, :] = colour
        else:
            image[:, i:i+width] = colour
        #colour = 1 - colour
        colour = np.random.uniform()
        i += width
    #image = image[np.newaxis, :, :]
    image = image[:, :, np.newaxis]
    return image


def generate_diagonally_striped_images(positive=True, max_deviation=0, max_width=1, image_width=9):
    """
    generate image arrays with diagonal stripes
    :param positive: bool, set to True for lines with a positive gradient
    :param max_deviation: maximum allowed deviation of gradient (in degrees) from 45 degrees
    :param max_width: int, the maximum number of pixels a line's width may have
    :param image_width: int, width and height of the image (i.e. will always be square)
    :return:
    """
    baseline_angle = 45 * (np.pi / 180)
    min_angle, max_angle = baseline_angle - (max_deviation * np.pi / 180), baseline_angle + (max_deviation * np.pi / 180)
    target_angle = min_angle + np.random.rand() * (max_angle - min_angle)
    target_gradient = np.around(np.tan(target_angle), 2)
    if positive:
        target_gradient *= -1

    image = np.zeros((image_width, image_width))
    y_offset = int(-target_gradient * image_width)
    y_start = min(0, y_offset)
    y_end = max(image_width, image_width + y_offset)
    i = y_start
    #colour = np.random.choice([0, 1])
    colour = np.random.uniform()
    width = np.random.randint(1, max_width + 1)
    while i < y_end:
        if width < 1:
            width = np.random.randint(1, max_width + 1)
            #colour = 1 - colour
            colour = np.random.uniform()
        for x in range(image_width):
            y = int(target_gradient*x) + i
            if 0 <= y < image_width:
                image[y, x] = colour
        width -= 1
        i += 1

    image = image[:, :, np.newaxis]
    return image


class StripedImages(BaseDataClass):
    """
    Creates a dataset of 9x9 images with either horizontal or vertical stripes
    """
    def __init__(self, max_width=1):
        super().__init__()
        self.max_width = max_width
        self.demo_im = generate_striped_image().transpose()
        self.n_classes = 2

    def sample(self, N=None):
        horizontal = np.random.choice([True, False])
        label = 0 if horizontal else 1
        max_width = np.random.randint(1, self.max_width + 1)
        return torch.from_numpy(generate_striped_image(horizontal=horizontal, max_width=max_width)).float(), label

    def get_shape(self):
        return self.demo_im.shape


class StripedOODImages(StripedImages):
    def __init__(self, max_width=1, max_deviation=0):
        super().__init__(max_width)
        self.max_deviation = max_deviation

    def sample(self, N=None):
        positive = np.random.choice([True, False])
        return torch.from_numpy(generate_diagonally_striped_images(positive, max_deviation=self.max_deviation, max_width=self.max_width)).float(), positive


if __name__ == "__main__":
    from matplotlib import pyplot as plt
    for i in range(15):
        if np.random.choice([True, False]):
            positive = 0==i % 2
            test = generate_diagonally_striped_images(max_width=4, positive=positive, max_deviation=15)
            title = f"diagonal, positive={positive}"
        else:
            horizontal = 0==i % 2
            test = generate_striped_image(horizontal=horizontal, max_width=4)
            title = f"orthogonal, horizontal={horizontal}"
        plt.imshow(test)
        plt.title(title)
        plt.show()

