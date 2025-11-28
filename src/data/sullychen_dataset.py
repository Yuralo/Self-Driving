import os
import pandas as pd
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import torch

class SullyChenDataset(Dataset):
    def __init__(self, data_dir, transform=None, sequence_length=1):
        """
        Args:
            data_dir (string): Directory with all the images and data.txt.
            transform (callable, optional): Optional transform to be applied on a sample.
            sequence_length (int): Number of frames to stack (for temporal models).
        """
        self.data_dir = data_dir
        self.transform = transform
        self.sequence_length = sequence_length
        
        # Read data.txt
        # Format: filename.jpg angle
        data_file = os.path.join(data_dir, 'data.txt')
        if not os.path.exists(data_file):
            # Try finding it recursively or check if it's named differently
            # For now, assume strict structure
            # raise FileNotFoundError(f"data.txt not found in {data_dir}")
            print(f"Warning: data.txt not found in {data_dir}. Dataset will be empty.")
            self.data = []
            return
        
        self.data = []
        with open(data_file, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 2:
                    filename = parts[0]
                    angle_degrees = float(parts[1])
                    angle_normalized = angle_degrees / 25.0
                    self.data.append((filename, angle_normalized))

    def __len__(self):
        if len(self.data) < self.sequence_length:
            return 0
        return len(self.data) - self.sequence_length + 1

    def __getitem__(self, idx):
        images = []
        steerings = []
        
        for i in range(self.sequence_length):
            current_idx = idx + i
            filename, steering = self.data[current_idx]
            
            img_path = os.path.join(self.data_dir, filename)
            if not os.path.exists(img_path):
                image = Image.new('RGB', (200, 66), color='black')
            else:
                image = Image.open(img_path).convert('RGB')
            
            if self.transform:
                image = self.transform(image)
            
            images.append(image)
            steerings.append(steering)
            
        images = torch.stack(images)
        target_steering = torch.tensor(steerings[-1], dtype=torch.float32)
        
        return images, target_steering
