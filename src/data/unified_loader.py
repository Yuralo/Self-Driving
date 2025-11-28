import torch
from torch.utils.data import ConcatDataset

class UnifiedLoader:
    def __init__(self, datasets):
        """
        Args:
            datasets (list): List of Dataset objects.
        """
        self.dataset = ConcatDataset(datasets)
        
    def get_dataset(self):
        return self.dataset
