from Model.U_net import UNet
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from torchmetrics.classification import  BinaryF1Score
from torchmetrics.segmentation import DiceScore
from tqdm import tqdm

import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
import wandb as wb

from Data.CelebAMaskDataset import CelebAMaskDataset
from Data.visualize_batch import visualize_batch

# Set device on which computing all tensor operations
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# This method is used for visualize some results of the U-Net, as :
# generated mask and, compared with target mask, and mask applied to original 
# image compared with original image
def log_seg_results(model, images, masks):
    """
    Run inference on a batch and log the results to wandb
    
    Args:
        model: PyTorch model
        images: Batch of images [B, C, H, W]
        masks: Batch of ground truth masks [B, 1, H, W]
        wb_logger: Weights & Biases logger
        step: Current step (epoch or batch number)
    """
    # Get predictions from the model
    model.eval()
    with torch.no_grad():
        logits = model(images)
        preds = torch.sigmoid(logits)
    
    # Visualize and log to wandb
    visualize_batch(
        img=images[:4], 
        mask=masks[:4],
        pred=preds[:4],
        num_samples=4,
        apply_mask=True,
        use_wandb=True
    )
    
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

# DICE coefficient function, with this we can evaluate the segmentation of model

# Compute norm of params of model with Euclid norm
def compute_params_norm(model):
    grad_norm = 0
    
    for p in model.parameters():
        if p.grad is not None:
            params_norm = p.grad.detach().data.norm(2).to('cpu')
            grad_norm += params_norm.item() ** 2

    return grad_norm ** 0.5

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
    accumulation_steps = 4

    dataset_root = 'rpcats/Data/Dataset/CelebAMask-HQ'

    # Create weight and biases logger
    wb_logger = set_wb(
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

    # Instance the train and validation dataset and Loader
    train_celeb_dataset = CelebAMaskDataset(
        root_dir=dataset_root,
        img_size=512,
        transform=False,
        split='train'
    )
    train_loader = DataLoader(
        dataset=train_celeb_dataset,
        shuffle=True,
        batch_size=batch_size
    )

    val_celeb_dataset = CelebAMaskDataset(
        root_dir=dataset_root,
        img_size=512,
        transform=False,
        split='validation'
    )
    val_loader = DataLoader(
        dataset=val_celeb_dataset,
        shuffle=True,
        batch_size=batch_size
    )

    # Set model to train mode and defines optimizer and lr scheduler
    model.train().to(device)
    optim = Adam(model.parameters(), betas=(0.99, 0.98), eps=1e-6, weight_decay=weight_decay)
    scheduler = ReduceLROnPlateau(
        optimizer=optim,
        mode='min',
        factor=0.5,
        patience=5
    )

    # Defines loss function (Cross entropy) and F1 Score and Dice Score
    criterion = nn.CrossEntropyLoss()
    f1_score = BinaryF1Score(
        threshold=0.5,
        validate_args=True,
        multidim_average='global',
        ignore_index=None,
        zero_division=1.0
    )
    dice_score = DiceScore(
        num_classes=2,         
        average='micro',        
        ignore_index=None,      
        zero_division=1.0,      
        threshold=0.5,          
    )

    # Training Loop
    for epoch in tqdm(range(epochs)):
        # Reset model to train mode and reset all metrics for training
        model.train()
        train_loss = 0
        val_loss = 0

        dice_score.reset()
        f1_score.reset()

        for idx, batch in enumerate(tqdm(train_loader)):
            # Extract data of batch (images as input and masks as target)
            # masks shape : [batch_size, channel, height, width] height, width = 512, channels = 1
            images = batch['image'].to(device)
            masks = batch['mask'].to(device)

            # Ligts_out is the output of the model, is the final feature map 
            logits_out = model(images)
            probs = torch.sigmoid(logits_out)

            # Rempve channels size from masks
            masks_without_channels = masks.squeeze(1).long()

            # Compute loss and its gradient compute F1, Dice Scores
            loss = criterion(logits_out, masks_without_channels)
            loss /= accumulation_steps
            loss.backward()
            if (idx + 1) % accumulation_steps == 0 or (idx + 1) == len(train_loader):
                optim.step()
                optim.zero_grad()

            train_loss += loss.item()

            dice = dice_score.update(probs, masks)
            f1 = f1_score.update(probs, masks)

        dice = dice_score.compute()
        f1 = f1_score.compute()

        # update optimizer and lr_scheduler, 
        scheduler.step()

        # Compute train loss avarage
        train_loss_avg = train_loss / (len(train_loader))
        
        # Defines model parameters for wandb log
        current_lr = optim.param_groups[0]['lr']
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        trainable_params_norm = compute_params_norm(model)

        # Set model to eval mode
        model.eval()
        val_loss = 0
        # Validation loop
        with torch.no_grad():
            for idx, batch in enumerate(tqdm(val_loader)):
                images = batch['image'].to(device)
                masks = batch['mask'].to(device)

                logits_out = model(images)

                masks_without_channels = masks.squeeze(1).long()
                loss = criterion(logits_out, masks_without_channels)

                val_loss += loss.item()

        val_loss_avg = val_loss / (len(val_loader))

        scheduler.step(val_loss_avg)

        # Visualize some example of masks generated by U-Net, also they applied to the original IMG
        log_seg_results(
            model = model,
            images = images,
            masks = masks,
        )

        wb_logger.log({
            # Log of training loop
            f'TRAIN -> epoch / {epochs}' : epoch+1,
            'TRAIN -> tran_loss' : train_loss_avg,
            'TRAIN -> learning rate' : current_lr,
            'TRAIN -> gradient params' : trainable_params_norm,
            'TRAIN -> F1 Score' : f1,
            'TRAIN -> Dice Score' : dice,

            # Log of model params
            'MODEL -> trainable params' : trainable_params,
            'MODEL -> total param' : total_params,

            # Log of validation loop
            'VALIDATION -> validation loss' : val_loss_avg
        })



