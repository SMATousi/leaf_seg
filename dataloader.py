import os
import torch
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np


class leaf_segmentation_dataset(Dataset):
    def __init__(self, image_folder, label_folder, transform=None, threshold = 0):
        self.image_folder = image_folder
        self.label_folder = label_folder
        self.threshold = threshold
        self.transform = transform
        self.image_filenames = sorted([f for f in os.listdir(image_folder) if f.endswith(".jpg")])
        self.label_filenames = [f.replace(".jpg", "_mask.jpg") for f in self.image_filenames]

    def __len__(self):
        return len(self.image_filenames)

    def __getitem__(self, idx):
        image_filename = self.image_filenames[idx]
        label_filename = self.label_filenames[idx]

        image_path = os.path.join(self.image_folder, image_filename)
        label_path = os.path.join(self.label_folder, label_filename)

        image = Image.open(image_path).convert("RGB")
        label = Image.open(label_path).convert("L")
        
        

        if self.transform:
            image = self.transform(image)
            label = self.transform(label)
        
        label = np.array(label)
        label = (label < self.threshold).astype(np.float32)
        

        return image, label