from Model.U_net import UNet

import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np

if __name__ == "__main__":
    layer_num = 4
    in_channels = 3
    num_classes = 1
    conv_kernel_size = 3
    convT_kernel_size = 2
    pool_kernel_size = 2

    model = UNet(
        layer_num = layer_num,
        in_channels = in_channels,
        num_classes = num_classes,
        conv_kernel_size = conv_kernel_size,
        convT_kernel_size = convT_kernel_size,
        pool_kernel_size = pool_kernel_size,
    )