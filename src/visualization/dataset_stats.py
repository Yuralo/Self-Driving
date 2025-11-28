import matplotlib.pyplot as plt
import numpy as np
import os
import sys

# Add src to path
sys.path.append(os.getcwd())

from src.data.udacity_dataset import DrivingDataset
from src.data.comma_dataset import CommaDataset

def plot_dataset_stats():
    print("Loading datasets...")
    
    # Udacity
    udacity = DrivingDataset('data/data', sequence_length=1)
    udacity_steerings = udacity.data['steering'].values
    print(f"Udacity: {len(udacity_steerings)} samples")
    
    # Comma2k19
    comma = CommaDataset('data/comma2k19', sequence_length=1)
    # CommaDataset stores samples as dicts with 'steerings' (list of values)
    # We want the last value of each sample, or just all values?
    # The samples list has {steerings: [s1, s2...]}
    # Let's extract the target steering (last one)
    comma_steerings = [s['steerings'][-1] for s in comma.samples]
    print(f"Comma2k19: {len(comma_steerings)} samples")
    
    # Plot
    plt.figure(figsize=(12, 6))
    
    plt.hist(udacity_steerings, bins=50, alpha=0.5, label='Udacity', density=True)
    plt.hist(comma_steerings, bins=50, alpha=0.5, label='Comma2k19', density=True)
    
    plt.title('Steering Angle Distribution (Normalized)')
    plt.xlabel('Steering Angle')
    plt.ylabel('Density')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.savefig('visualization/dataset_distribution.png')
    print("Saved plot to visualization/dataset_distribution.png")

if __name__ == '__main__':
    plot_dataset_stats()
