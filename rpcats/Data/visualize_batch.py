import matplotlib.pyplot as plt
import numpy as np
import torch

def visualize_batch(img, mask, num_samples=1):
    """
    Visualize one batch of the img and its mask
    """
    # Get images and its mask
    imgs = img
    masks = mask
    
    # Make sure num_samples doesn't exceed the batch size
    num_samples = min(num_samples, len(imgs))
    
    fig, axes = plt.subplots(2, num_samples, figsize=(num_samples * 4, 8))
    
    # Handle the case where num_samples=1 (axes would be 1D)
    if num_samples == 1:
        axes = np.array([[axes[0]], [axes[1]]])
    
    for i in range(num_samples):
        # Convert PyTorch tensor to numpy and transpose if needed
        if torch.is_tensor(imgs[i]):
            img_np = imgs[i].detach().cpu().numpy()
            # If image has shape [C, H, W], transpose to [H, W, C] for RGB
            if img_np.shape[0] == 3:
                img_np = np.transpose(img_np, (1, 2, 0))
            # If grayscale with shape [1, H, W], squeeze to [H, W]
            elif img_np.shape[0] == 1:
                img_np = np.squeeze(img_np, axis=0)
        else:
            img_np = imgs[i]
        
        # Same for mask
        if torch.is_tensor(masks[i]):
            mask_np = masks[i].detach().cpu().numpy()
            if mask_np.shape[0] > 1:  # multi-channel mask
                mask_np = np.transpose(mask_np, (1, 2, 0))
            else:  # binary mask
                mask_np = np.squeeze(mask_np, axis=0)
        else:
            mask_np = masks[i]
        
        # Display image
        if len(img_np.shape) == 3 and img_np.shape[2] == 3:  # RGB image
            axes[0, i].imshow(img_np)
        else:  # Grayscale image
            axes[0, i].imshow(img_np, cmap='gray')
        axes[0, i].set_title(f'Image {i+1}')
        axes[0, i].axis('off')
        
        # Display mask
        axes[1, i].imshow(mask_np, cmap='gray')
        axes[1, i].set_title(f'Mask {i+1}')
        axes[1, i].axis('off')
    
    plt.tight_layout()
    plt.show()