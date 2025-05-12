import matplotlib.pyplot as plt
import numpy as np
import torch

def visualize_batch(img, mask, pred=None, num_samples=1, apply_mask=True, use_wandb=False):
    """
    Visualize one batch of the image, its ground truth mask and predicted mask.
    Can also apply the mask to the original image.
    
    Args:
        img (torch.Tensor): Batch of images [B, C, H, W]
        mask (torch.Tensor): Batch of ground truth masks [B, 1, H, W]
        pred (torch.Tensor, optional): Batch of predicted masks [B, C, H, W]. 
                                      For binary segmentation C=1, for multi-class C=num_classes
        num_samples (int): Number of samples to visualize
        apply_mask (bool): Whether to show the image with the mask applied
        use_wandb (bool): Whether to log the visualization to wandb
        
    Returns:
        plt.Figure: The figure object (if not using wandb)
    """
    # Get images and masks
    imgs = img
    masks = mask
    
    # Make sure num_samples doesn't exceed the batch size
    num_samples = min(num_samples, len(imgs))
    
    # Determine the number of rows based on what we're visualizing
    num_rows = 2  # Default: image and ground truth mask
    if pred is not None:
        num_rows += 1  # Add row for predicted mask
    if apply_mask:
        num_rows += 1  # Add row for masked image
    
    fig, axes = plt.subplots(num_rows, num_samples, figsize=(num_samples * 4, num_rows * 4))
    
    # Handle the case where num_samples=1
    if num_samples == 1:
        axes = axes.reshape(num_rows, 1)
    
    for i in range(num_samples):
        row_idx = 0
        
        # Process image
        if torch.is_tensor(imgs[i]):
            img_np = imgs[i].detach().cpu().numpy()
            if img_np.shape[0] == 3:  # RGB image
                img_np = np.transpose(img_np, (1, 2, 0))
            elif img_np.shape[0] == 1:  # Grayscale image
                img_np = np.squeeze(img_np, axis=0)
        else:
            img_np = imgs[i]
        
        # Display image
        if len(img_np.shape) == 3 and img_np.shape[2] == 3:  # RGB image
            axes[row_idx, i].imshow(img_np)
        else:  # Grayscale image
            axes[row_idx, i].imshow(img_np, cmap='gray')
        axes[row_idx, i].set_title(f'Image {i+1}')
        axes[row_idx, i].axis('off')
        row_idx += 1
        
        # Process ground truth mask
        if torch.is_tensor(masks[i]):
            mask_np = masks[i].detach().cpu().numpy()
            if mask_np.shape[0] > 1:  # multi-channel mask
                mask_np = np.transpose(mask_np, (1, 2, 0))
            else:  # binary mask
                mask_np = np.squeeze(mask_np, axis=0)
        else:
            mask_np = masks[i]
        
        # Display ground truth mask
        axes[row_idx, i].imshow(mask_np, cmap='gray')
        axes[row_idx, i].set_title(f'Ground Truth Mask {i+1}')
        axes[row_idx, i].axis('off')
        row_idx += 1
        
        # Process and display predicted mask if provided
        if pred is not None:
            if torch.is_tensor(pred[i]):
                # Handle different prediction formats
                pred_np = pred[i].detach().cpu().numpy()
                
                # If multi-class predictions, get the class with highest probability
                if pred_np.shape[0] > 1:  # [C, H, W] where C is number of classes
                    pred_np = np.argmax(pred_np, axis=0)
                else:  # Binary prediction [1, H, W]
                    pred_np = np.squeeze(pred_np, axis=0)
                    # Apply threshold for binary prediction
                    pred_np = (pred_np > 0.5).astype(np.float32)
            else:
                pred_np = pred[i]
            
            # Display predicted mask
            axes[row_idx, i].imshow(pred_np, cmap='gray')
            axes[row_idx, i].set_title(f'Predicted Mask {i+1}')
            axes[row_idx, i].axis('off')
            row_idx += 1
        
        # Apply mask to image if requested
        if apply_mask:
            # Choose which mask to apply (ground truth or predicted)
            applied_mask = pred_np if pred is not None else mask_np
            
            # Create masked image
            masked_img = img_np.copy()
            
            # Handle different image formats
            if len(masked_img.shape) == 3:  # RGB
                # Create a colored overlay (red for visibility)
                mask_overlay = np.zeros_like(masked_img)
                mask_overlay[:, :, 0] = applied_mask * 1.0  # Red channel
                
                # Blend original image with the mask
                alpha = 0.5  # Transparency factor
                masked_img = (1-alpha) * masked_img + alpha * mask_overlay
            else:  # Grayscale
                # Just highlight the mask area
                mask_highlight = applied_mask * 0.7  # Value to highlight mask
                masked_img = np.maximum(masked_img, mask_highlight)
            
            # Display masked image
            axes[row_idx, i].imshow(masked_img)
            axes[row_idx, i].set_title(f'Masked Image {i+1}')
            axes[row_idx, i].axis('off')
    
    plt.tight_layout()
    
    if use_wandb:
        import wandb
        # Log the plot to wandb
        wandb.log({"Segmentation Visualization": wandb.Image(plt)})
        plt.close(fig)  # Close the figure to free memory
    else:
        plt.show()
        return fig