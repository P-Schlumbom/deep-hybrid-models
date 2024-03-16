import numpy as np
import torch
import random


def conv_output_size(width, kernel, padding=0, stride=1):
    #print(np.asarray(((width - kernel + 2*padding)/stride) + 1).astype(int))
    return np.asarray(((width - kernel + 2*padding)/stride) + 1).astype(int)

def running_average(val, mean, count):
    """
    Based on Welford's method:
    M_1 = x_1
    M_k = M_{k-1} + (x_k - M_{k-1}) / (k)
    :param val:
    :param mean:
    :param count:
    :return:
    """
    return mean + (val - mean) / (count + 1)


def get_model_params(model):
    pp = 0
    for p in list(model.parameters()):
        nn = 1
        for s in list(p.size()):
            nn = nn * s
        pp += nn
    return pp


def print_model_params(model):
    print("Number of parameters:")
    pp = 0
    for p in list(model.parameters()):
        nn = 1
        for s in list(p.size()):
            nn = nn * s
        pp += nn
    print(pp)


def print_model_architecture(model):
    # Iterate over the model's parameters and print the information
    total_params = 0
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(f"Layer: {name}, Parameters: {param.numel()}")
            total_params += param.numel()
    #for name, module in model.named_children():
    #    print(f"module: {module}")
    for child in model.children():
        print(child)

    print(f"Total Trainable Parameters: {total_params}")


def set_seed(seed: int) -> None:
    """
    Sets the seeds at a certain value.
    :param seed: the value to be set
    """
    print("Setting seeds ...... \n")
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

