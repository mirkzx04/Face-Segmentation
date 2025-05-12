import os
import numpy as np
import cv2
import torch
import random
import matplotlib.pyplot as plt

from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
from glob import glob

class CelebAMaskDataset(Dataset):
    """
    Personalized Dataset for CelebAMask-HQ
    """

    def __init__(self,
                 root_dir,
                 img_size,
                 transform,
                 split):
        super().__init__()

        """
        Args:
            root_dir : directory of the CelebAMask-HQ dataset
            img_size : Dimension of the image
            split : Train, val and test percentual of the dataset
            transforms : applicable transformations if desired
        """

        self.root_dir = root_dir
        self.img_size = img_size
        self.transform = transform
        self.split = split

        # Defines face parts to convert like one face
        self.face_parts = ['skin', 'nose', 'eye_g', 'l_eye', 'r_eye', 
                          'l_brow', 'r_brow', 'l_ear', 'r_ear', 'mouth', 
                          'u_lip', 'l_lip']
        
        # Create img and mask path list
        self.imgs_path = os.path.join(root_dir, 'CelebA-HQ-img')
        self.masks_path = os.path.join(root_dir, 'CelebAMask-HQ-mask-anno')

        # Get all imgs file
        self.img_files = sorted(glob(os.path.join(
            self.imgs_path, '*.jpg'
        )))

        # Get total imgs number
        total_images = len(self.img_files)

        # Create train, val and test split based on seed for reproducibility
        random.seed(42)
        indices = list(range(total_images))
        random.shuffle(indices)
        
        if split == 'train':
            self.indices = indices[:int(0.8 * total_images)]
        elif split == 'val':
            self.indices = indices[int(0.8 * total_images):int(0.9 * total_images)]
        else:  # test
            self.indices = indices[int(0.9 * total_images):]
        
        self.image_files = [self.img_files[idx] for idx in self.indices]
        print(f'Loaded {len(self.image_files)} images for {split} split')

    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        """
        Return one img and its mask
        """

        # Load img
        img_path = self.image_files[idx]
        img_id = os.path.basename(img_path).split('.')[0]

        # Read img
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (self.img_size, self.img_size))

        # Create binary mask of the face
        binary_mask = self.create_mask(img_id)

        # Transform image in to tensor
        image = transforms.ToTensor()(image)
        binary_mask = torch.from_numpy(binary_mask).float().unsqueeze(0)

        return {'image' : image, 'mask' : binary_mask}
    
    def create_mask(self, img_id):
        """
        Create a binary mask that unites all face parts in to one
        """

        binary_mask = np.zeros((self.img_size, self.img_size), dtype=np.uint8)
        img_id_int = int(img_id)

        # Get mask id from file with dataset format
        folder_idx = img_id_int // 2000
        mask_folder = os.path.join(self.masks_path, str(folder_idx))
        file_id = f'{img_id_int:05d}'

        # Unites all mask in to one part : The face
        for part in self.face_parts:
            mask_path = os.path.join(mask_folder, f'{file_id}_{part}.png')

            if os.path.exists(mask_path):
                # Read mask of the parts of the face
                part_mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
                part_mask = cv2.resize(part_mask, (self.img_size, self.img_size))

                # Unites all binary mask (All pixels are > 0 are face)
                binary_mask = np.logical_or(binary_mask, part_mask > 0).astype(np.uint8)
        
        # Normalize in 0-1
        return binary_mask * 255