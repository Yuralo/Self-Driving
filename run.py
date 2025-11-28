import sys
import os
import torch
from torch.utils.data import DataLoader

# Add current directory to path so we can import 'src'
sys.path.append(os.getcwd())

from src.data.udacity_dataset import DrivingDataset, get_transforms
from src.data.comma_dataset import CommaDataset
from src.train_multi import train_multi_dataset

def inspect_dataset(name, dataset, num_samples=3):
    print(f"\n--- Inspecting {name} Dataset ---")
    print(f"Total Samples: {len(dataset)}")
    
    if len(dataset) == 0:
        print("Dataset is empty.")
        return

    for i in range(min(num_samples, len(dataset))):
        try:
            images, target = dataset[i]
            print(f"Sample {i}:")
            print(f"  Input Shape: {images.shape} (Seq, C, H, W)")
            print(f"  Target Steering: {target.item():.4f}")
            
            # For Comma dataset, print video source if possible
            if hasattr(dataset, 'samples'):
                print(f"  Source: {dataset.samples[i].get('video_path', 'N/A')} (Frame {dataset.samples[i].get('frame_idx', 'N/A')})")
                
        except Exception as e:
            print(f"  Error loading sample {i}: {e}")

def main():
    print("Initializing Datasets for Inspection...")
    
    # Transforms
    transform = get_transforms()
    sequence_length = 5
    
    # 1. Udacity
    try:
        udacity_ds = DrivingDataset('data/data', transform=transform, sequence_length=sequence_length)
        inspect_dataset("Udacity", udacity_ds)
    except Exception as e:
        print(f"Failed to load Udacity dataset: {e}")

    # 2. Comma2k19
    try:
        comma_ds = CommaDataset('data/comma2k19', transform=transform, sequence_length=sequence_length)
        inspect_dataset("Comma2k19", comma_ds)
    except Exception as e:
        print(f"Failed to load Comma2k19 dataset: {e}")

    print("\n" + "="*50 + "\n")
    print("Starting Training...")
    train_multi_dataset()

if __name__ == "__main__":
    main()
