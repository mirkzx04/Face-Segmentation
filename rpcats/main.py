from Model.U_net import UNet
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from tqdm import tqdm

import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
import wandb as wb

from Data.CelebAMaskDataset import CelebAMaskDataset
from Data.visualize_batch import visualize_batch

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
        entity="mirkzx-sapienza-universit-di-roma",
        project="Face_Seg",
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

# Dice coefficient function, with this we can evaluate the segmentation of model

if __name__ == "__main__":

    # Defines hyperparameters of the U-Net
    layer_num = 4
    in_channels = 3
    num_classes = 2
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

    # Defines training parameters
    epochs = 150
    lr = 1e-4
    dropout = None
    optimizer = 'Adam'
    scheduler_name = 'Reduce On Plateau'
    scheduler_params = {'factor' : 0.3, 'patience' : 5, 'min_lr' : 1e-7}
    weight_decay = 1e-5
    batch_size = 4

    dataset_root = 'rpcats/Data/Dataset/CelebAMask-HQ'

    # Create weight and biases logger
    wb_ = set_wb(
        epochs=epochs,
        layers=layer_num,
        lr = lr,
        schedule_name=scheduler_name,
        params_schedule = scheduler_params,
        opti_name=optimizer,
        weight_decay=weight_decay,
        dropout=dropout,
        name_dataset='CelebAMask'
    )

    # Instance the dataset and Loader
    celeb_dataset = CelebAMaskDataset(
        root_dir=dataset_root,
        img_size=512,
        transform=False,
        split='train'
    )
    train_loader = DataLoader(
        dataset=celeb_dataset,
        shuffle=True,
        batch_size=batch_size
    )
    # Set model to train mode and defines optimizer and lr scheduler
    model.train()
    optim = Adam(model.parameters(), betas=(0.99, 0.98), eps=1e-6)
    scheduler = ReduceLROnPlateau(
        optimizer=optim,
        mode='min',
        factor=0.5,
        patience=5
    )

    # Defines loss function (Cross entropy)
    criterion = nn.CrossEntropyLoss()

    # Training Loop
    for epoch in tqdm(range(epochs)):
        # Reset model to train mode and reset train and validation loss
        model.train()
        train_loss = 0
        val_loss = 0

        for idx, batch in enumerate(tqdm(train_loader)):
            # Extract data of batch (images as input and masks as target)
            images = batch['image']
            # [batch_size, channel, height, width] height, width = 512, class = 1
            masks = batch['mask']

            # Ligts_out is the output of the model, is the final feature map 
            # logits_out = model(images)

            # Rempve channels size from masks
            masks_without_channels = masks.squeeze(1).long()

            # Compute loss and update optimizer and lr_scheduler
            loss = criterion(logits_out, masks_without_channels)

            optim.step()
            scheduler.step()

            train_loss += loss.item()

        train_loss_avg = train_loss / (idx + 1)
            
            # Debug logits_out and masks dimension
            # print(f'logits dimension : {logits_out.shape}')
            print(f'masks dimension : {masks.shape}')



