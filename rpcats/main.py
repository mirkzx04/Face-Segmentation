from Model.U_net import UNet
from torch.optim import Adam

import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
import wandb as wb

# Set Weight and Biases module
def set_wb(
        epochs, 
        layers, 
        lr, 
        schedule_name,
        params_schedule,
        opti_name,
        weight_decay, 
        dropout, 
        name_dataset
    ):

    return wb.init(
        entity="Mirkzx",
        project="Face Seg",
        config={
            'Architecture' : 'U-Net',
            'dataset' : name_dataset,
            'Learning rate' : lr,
            'lr_schedule' : {
                'schedule_name' : schedule_name, 
                'params_schedule' : params_schedule
                },
            'optimizer' : opti_name,
            'epochs' : epochs,
            'layers' : layers,
            'weight decay' : weight_decay,
            'dropout' : dropout,
        }
    )

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

    epochs = 150
    lr = 1e-4
    dropout = None
    optimizer = 'Adam'
    scheduler_name = 'Reduce On Plateau'
    scheduler_params = {'factor' : 0.3, 'patience' : 5, 'min_lr' : 1e-7}
    weight_decay = 1e-5

    wb_ = set_wb(
        epochs=epochs,
        layers=layer_num,
        lr = lr,
        schedule_name=scheduler_name,
        scheduler_params = scheduler_params,
        opti_name=optimizer,
        weight_decay=weight_decay,
        dropout=dropout
    )


