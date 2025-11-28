import os
import pandas as pd
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import torch

class DrivingDataset(Dataset):
    def __init__(self, data_dir, transform=None, sequence_length=1):
        """
        Args:
            data_dir (string): Directory with all the images and CSV.
            transform (callable, optional): Optional transform to be applied on a sample.
            sequence_length (int): Number of frames to stack (for temporal models).
        """
        self.data_dir = data_dir
        self.csv_path = os.path.join(data_dir, 'driving_log.csv')
        self.img_dir = os.path.join(data_dir, 'IMG')
        self.transform = transform
        self.sequence_length = sequence_length
        
        # Read CSV
        # Columns: center,left,right,steering,throttle,brake,speed
        self.data = pd.read_csv(self.csv_path)
        
        # Clean paths in CSV (sometimes they are absolute paths from the recorder machine)
        self.data['center'] = self.data['center'].apply(lambda x: os.path.basename(x))
        self.data['left'] = self.data['left'].apply(lambda x: os.path.basename(x))
        self.data['right'] = self.data['right'].apply(lambda x: os.path.basename(x))

    def __len__(self):
        return len(self.data) - self.sequence_length + 1

    def __getitem__(self, idx):
        # Get sequence of frames
        images = []
        steerings = []
        
        for i in range(self.sequence_length):
            current_idx = idx + i
            row = self.data.iloc[current_idx]
            
            # Load center image
            img_name = row['center']
            img_path = os.path.join(self.img_dir, img_name)
            image = Image.open(img_path).convert('RGB')
            
            if self.transform:
                image = self.transform(image)
            
            images.append(image)
            steerings.append(row['steering'])
            
        # Stack images: (Seq, C, H, W)
        images = torch.stack(images)
        
        # Target: usually we want to predict the steering of the LAST frame in the sequence
        # or the next frame. Let's predict the steering of the last frame.
        target_steering = torch.tensor(steerings[-1], dtype=torch.float32)
        
        return images, target_steering

def get_transforms():
    return transforms.Compose([
        transforms.Resize((66, 200)), # Nvidia architecture size
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
